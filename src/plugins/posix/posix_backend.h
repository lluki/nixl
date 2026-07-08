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

#ifndef NIXL_SRC_PLUGINS_POSIX_POSIX_BACKEND_H
#define NIXL_SRC_PLUGINS_POSIX_POSIX_BACKEND_H

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "backend/backend_engine.h"
#include "file/file_path_mode.h"
#include "io_queue.h"
#include "posix_worker_pool.h"

// POSIX reuses the shared owned-fd base (path-mode devId stored for dereg).
using nixlPosixFileMD = nixlFilePathMD;

class nixlPosixBackendReqH : public nixlBackendReqH {
private:
    const nixl_xfer_op_t &operation; // The transfer operation (read/write)
    const nixl_meta_dlist_t &local; // Local memory descriptor list
    const nixl_meta_dlist_t &remote; // Remote memory descriptor list
    const int queue_depth_; // Queue depth for async I/O
    int num_confirmed_ios_; // Number of confirmed IOs
    nixlPosixIOQueue &io_queue_; // Async I/O queue assigned to this request
    const size_t worker_index_;
    nixl_status_t post_status_ = NIXL_SUCCESS;

    void
    ioDone(uint32_t data_size, int error);
    static void
    ioDoneClb(void *ctx, uint32_t data_size, int error);

public:
    nixlPosixBackendReqH(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         nixlPosixIOQueue &io_queue,
                         size_t worker_index);
    ~nixlPosixBackendReqH(){};

    nixl_status_t
    postXfer();
    void
    postXferOnWorker() noexcept;
    nixl_status_t
    prepXfer();
    nixl_status_t
    checkXfer();

    size_t
    workerIndex() const noexcept {
        return worker_index_;
    }

    // Exception classes
    class exception : public std::exception {
    private:
        const nixl_status_t code_;

    public:
        exception(const std::string &msg, nixl_status_t code) : std::exception(), code_(code) {}

        nixl_status_t
        code() const noexcept {
            return code_;
        }
    };
};

class nixlPosixEngine : public nixlBackendEngine {
private:
    std::string_view io_queue_type_;
    uint32_t ios_pool_size_;
    uint32_t kernel_queue_size_;
    size_t thread_num_;
    bool thread_unpin_;
    mutable std::unique_ptr<nixlPosixIOQueue> io_queue_;
    mutable std::vector<std::unique_ptr<nixlPosixIOQueue>> worker_io_queues_;
    std::unique_ptr<nixlPosixWorkerPool> worker_pool_;
    nixl::PathModeDevIdRegistry path_mode_devids_;

    size_t
    selectWorker() const;
    nixlPosixIOQueue &
    queueForWorker(size_t worker_index) const;
    void
    invokeWorker(size_t worker_index, std::function<void()> task) const;

public:
    nixlPosixEngine(const nixlBackendInitParams *init_params);
    virtual ~nixlPosixEngine() = default;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {FILE_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }
};

#endif
