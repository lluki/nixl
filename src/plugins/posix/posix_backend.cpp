/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cmath>
#include <errno.h>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include "posix_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include "file/file_utils.h"

namespace {
bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (remote_agent != local_agent) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Remote agent must match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
        return false;
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != FILE_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be FILE_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Mismatch in descriptor counts - local: %d, remote: %d",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

nixlPosixBackendReqH &
castPosixHandle(nixlBackendReqH *handle) {
    if (!handle) {
        throw nixlPosixBackendReqH::exception("received null handle", NIXL_ERR_INVALID_PARAM);
    }
    return static_cast<nixlPosixBackendReqH &>(*handle);
}

static std::string_view
getIoQueueType(const nixl_b_params_t *custom_params) {
    if (nixl::getBackendParamDefaulted(custom_params, "use_aio", false)) {
        return "AIO";
    }

    if (nixl::getBackendParamDefaulted(custom_params, "use_uring", false)) {
        return "URING";
    }

    if (nixl::getBackendParamDefaulted(custom_params, "use_posix_aio", false)) {
        return "POSIXAIO";
    }

    return nixlPosixIOQueue::getDefaultIoQueueType();
}

// Log completion percentage at regular intervals (every log_percent_step percent)
void
logOnPercentStep(unsigned int completed, unsigned int total) {
    constexpr unsigned int default_log_percent_step = 10;
    static_assert(default_log_percent_step >= 1 && default_log_percent_step <= 100,
                  "log_percent_step must be in [1, 100]");
    unsigned int log_percent_step = total < 10 ? 1 : default_log_percent_step;

    if (total == 0) {
        NIXL_ERROR << "Tried to log completion percentage with total == 0";
        return;
    }
    // Only log at each percentage step
    if (completed % (total / log_percent_step) == 0) {
        NIXL_DEBUG << absl::StrFormat("Queue progress: %.1f%% complete",
                                      (completed * 100.0 / total));
    }
}
} // namespace

// -----------------------------------------------------------------------------
// POSIX Backend Request Handle Implementation
// -----------------------------------------------------------------------------

// NOTE: we initialize num_confirmed_ios_ to the number of descriptors, so if checkXfer is called
// before postXfer, it will return NIXL_SUCCESS immediately.
nixlPosixBackendReqH::nixlPosixBackendReqH(const nixl_xfer_op_t &op,
                                           const nixl_meta_dlist_t &loc,
                                           const nixl_meta_dlist_t &rem,
                                           nixlPosixIOQueue &io_queue,
                                           size_t worker_index)
    : operation(op),
      local(loc),
      remote(rem),
      queue_depth_(loc.descCount()),
      num_confirmed_ios_(queue_depth_),
      io_queue_(io_queue),
      worker_index_(worker_index) {
    NIXL_ASSERT(local.descCount());
    NIXL_ASSERT(remote.descCount());
}

void
nixlPosixBackendReqH::ioDone(uint32_t data_size, int error) {
    num_confirmed_ios_++;
    logOnPercentStep(num_confirmed_ios_, queue_depth_);
}

void
nixlPosixBackendReqH::ioDoneClb(void *ctx, uint32_t data_size, int error) {
    nixlPosixBackendReqH *self = static_cast<nixlPosixBackendReqH *>(ctx);
    self->ioDone(data_size, error);
}

nixl_status_t
nixlPosixBackendReqH::prepXfer() {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixBackendReqH::checkXfer() {
    if (post_status_ < 0) {
        return post_status_;
    }
    if (num_confirmed_ios_ == queue_depth_) {
        return NIXL_SUCCESS;
    }

    nixl_status_t status = io_queue_.poll();
    if (status < 0) {
        return status;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlPosixBackendReqH::postXfer() {
    num_confirmed_ios_ = 0;

    for (auto [local_it, remote_it] = std::make_pair(local.begin(), remote.begin());
         local_it != local.end() && remote_it != remote.end();
         ++local_it, ++remote_it) {
        int fd = static_cast<nixlPosixFileMD *>(remote_it->metadataP)->file_fd.fd();
        nixl_status_t status = io_queue_.enqueue(fd,
                                                 reinterpret_cast<void *>(local_it->addr),
                                                 remote_it->len,
                                                 remote_it->addr,
                                                 operation == NIXL_READ,
                                                 ioDoneClb,
                                                 this);

        if (status != NIXL_SUCCESS) {
            // Currently we do not support partial submissions, so it's all or nothing
            NIXL_ERROR << absl::StrFormat("Error preparing I/O operation: %d", status);
            return status;
        }
    }

    return io_queue_.post();
}

void
nixlPosixBackendReqH::postXferOnWorker() noexcept {
    try {
        post_status_ = postXfer();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Unexpected error posting POSIX transfer on worker: " << e.what();
        post_status_ = NIXL_ERR_BACKEND;
    }
    catch (...) {
        NIXL_ERROR << "Unknown error posting POSIX transfer on worker";
        post_status_ = NIXL_ERR_BACKEND;
    }
}

// -----------------------------------------------------------------------------
// POSIX Engine Implementation
// -----------------------------------------------------------------------------

nixlPosixEngine::nixlPosixEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      io_queue_type_(getIoQueueType(init_params->customParams)),
      ios_pool_size_(
          nixl::getBackendParamDefaulted(init_params->customParams, "ios_pool_size", 0u)),
      kernel_queue_size_(
          nixl::getBackendParamDefaulted(init_params->customParams, "kernel_queue_size", 0u)),
      thread_num_(
          nixl::getBackendParamDefaulted(init_params->customParams, "thread_num", size_t{1})),
      thread_unpin_(
          nixl::getBackendParamDefaulted(init_params->customParams, "thread_unpin", false)),
      io_queue_(
          thread_num_ == 1 ?
              nixlPosixIOQueue::instantiate(io_queue_type_, ios_pool_size_, kernel_queue_size_) :
              nullptr) {
    if (thread_num_ == 0) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend - thread_num must be at least 1";
        return;
    }
    if (io_queue_type_.empty()) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend - no supported io queue type found";
        return;
    }

    try {
        if (thread_num_ == 1) {
            if (!io_queue_) {
                throw std::runtime_error(
                    absl::StrFormat("unavailable io queue type requested: %s", io_queue_type_));
            }
        } else {
            worker_io_queues_.reserve(thread_num_);
            for (size_t i = 0; i < thread_num_; ++i) {
                auto io_queue = nixlPosixIOQueue::instantiate(
                    io_queue_type_, ios_pool_size_, kernel_queue_size_);
                if (!io_queue) {
                    throw std::runtime_error(
                        absl::StrFormat("unavailable io queue type requested: %s", io_queue_type_));
                }
                worker_io_queues_.push_back(std::move(io_queue));
            }
            worker_pool_ = std::make_unique<nixlPosixWorkerPool>(thread_num_, thread_unpin_);
        }
    }
    catch (const std::exception &e) {
        initErr = true;
        NIXL_ERROR << "Failed to initialize POSIX backend: " << e.what();
        return;
    }

    NIXL_INFO << absl::StrFormat(
        "POSIX backend initialized using io queue type: %s, thread_num: %zu, thread_unpin: %s",
        io_queue_type_,
        thread_num_,
        thread_unpin_ ? "true" : "false");
}

size_t
nixlPosixEngine::selectWorker() const {
    return worker_pool_ ? worker_pool_->nextWorker() : 0;
}

nixlPosixIOQueue &
nixlPosixEngine::queueForWorker(size_t worker_index) const {
    return worker_pool_ ? *worker_io_queues_.at(worker_index) : *io_queue_;
}

void
nixlPosixEngine::invokeWorker(size_t worker_index, std::function<void()> task) const {
    if (worker_pool_) {
        worker_pool_->invoke(worker_index, std::move(task));
    } else {
        task();
    }
}

nixl_status_t
nixlPosixEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) != supported_mems.end()) {
        out = nullptr;
        if (nixl_mem == FILE_SEG) {
            auto resv = path_mode_devids_.reserve(mem.devId, mem.metaInfo);
            if (!resv.ok()) {
                NIXL_ERROR << "POSIX path-mode requires a unique devId per file (devId="
                           << mem.devId << " already registered)";
                return NIXL_ERR_INVALID_PARAM;
            }
            try {
                std::unique_ptr<nixlPosixFileMD> file_md;
                if (worker_pool_ && nixl::parsePathMeta(mem.metaInfo)) {
                    const size_t worker_index = selectWorker();
                    invokeWorker(worker_index, [&]() {
                        file_md = std::make_unique<nixlPosixFileMD>(mem.devId, mem.metaInfo);
                    });
                } else {
                    file_md = std::make_unique<nixlPosixFileMD>(mem.devId, mem.metaInfo);
                }
                out = file_md.release();
            }
            catch (const std::system_error &e) {
                NIXL_ERROR << "POSIX path-mode open failed: " << e.what();
                return NIXL_ERR_BACKEND;
            }
            catch (const std::exception &e) {
                NIXL_ERROR << "POSIX path-mode worker failed: " << e.what();
                return NIXL_ERR_BACKEND;
            }
            resv.commit();
        }
        return NIXL_SUCCESS;
    }

    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlPosixEngine::deregisterMem(nixlBackendMD *meta) {
    // non-null meta is always a file MD. Release the path-mode reservation (path() empty in
    // fd-mode)
    if (meta) {
        auto *file_md = static_cast<nixlPosixFileMD *>(meta);
        if (!file_md->file_fd.path().empty()) {
            path_mode_devids_.release(file_md->devId);
        }
    }
    delete meta;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    try {
        const size_t worker_index = selectWorker();
        auto posix_handle = std::make_unique<nixlPosixBackendReqH>(
            operation, local, remote, queueForWorker(worker_index), worker_index);
        nixl_status_t status = NIXL_SUCCESS;
        invokeWorker(worker_index, [&]() { status = posix_handle->prepXfer(); });
        if (status != NIXL_SUCCESS) {
            return status;
        }

        handle = posix_handle.release();
        return NIXL_SUCCESS;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << absl::StrFormat("Error: %s", e.what());
        return e.code();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << absl::StrFormat("Unexpected error: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlPosixEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        if (worker_pool_) {
            if (!worker_pool_->submit(posix_handle.workerIndex(),
                                      [&posix_handle]() { posix_handle.postXferOnWorker(); })) {
                NIXL_ERROR << "POSIX worker pool rejected transfer submission";
                return NIXL_ERR_BACKEND;
            }
            return NIXL_IN_PROG;
        }

        nixl_status_t status = posix_handle.postXfer();
        if (status != NIXL_IN_PROG) {
            NIXL_ERROR << "Error in submitting queue";
        }
        return status;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Unexpected error submitting POSIX transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlPosixEngine::checkXfer(nixlBackendReqH *handle) const {
    try {
        auto &posix_handle = castPosixHandle(handle);
        nixl_status_t status = NIXL_SUCCESS;
        invokeWorker(posix_handle.workerIndex(), [&]() { status = posix_handle.checkXfer(); });
        return status;
    }
    catch (const nixlPosixBackendReqH::exception &e) {
        NIXL_ERROR << e.what();
        return e.code();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Unexpected error checking POSIX transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlPosixEngine::releaseReqH(nixlBackendReqH *handle) const {
    NIXL_ASSERT(handle != nullptr);
    delete handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i) {
        metadata[i] = descs[i].metaInfo;
    }

    return nixl::queryFileInfoList(metadata, resp);
}
