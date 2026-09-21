/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gcs_backend.h"
#include "common/backend.h"
#include "common/nixl_log.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <thread>

namespace {

std::size_t
getNumThreads(nixl_b_params_t *params) {
    const auto fallback = std::max(1u, std::thread::hardware_concurrency() / 2);
    const auto num_threads = nixl::getBackendParamDefaulted(params, "num_threads", fallback);
    if (num_threads == 0) {
        throw std::invalid_argument("GCS backend requires num_threads to be greater than zero");
    }
    return num_threads;
}

std::string
getBucket(nixl_b_params_t *params) {
    auto bucket = nixl::getBackendParamDefaulted(params, "bucket", std::string());
    if (bucket.empty()) {
        if (const auto *env = std::getenv("NIXL_GCS_BUCKET")) {
            bucket = env;
        }
    }
    if (bucket.empty()) {
        throw std::runtime_error(
            "GCS backend requires custom parameter 'bucket' or NIXL_GCS_BUCKET");
    }
    return bucket;
}

class GcsReqH : public nixlBackendReqH {
public:
    std::vector<std::future<nixl_status_t>> futures;

    nixl_status_t
    status() {
        auto future = futures.begin();
        while (future != futures.end()) {
            if (future->wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                ++future;
                continue;
            }

            try {
                const auto status = future->get();
                if (result_ == NIXL_SUCCESS && status != NIXL_SUCCESS) {
                    result_ = status;
                }
            }
            catch (const std::exception &e) {
                NIXL_ERROR << "GCS worker failed: " << e.what();
                result_ = NIXL_ERR_BACKEND;
            }
            future = futures.erase(future);
        }

        return futures.empty() ? result_ : NIXL_IN_PROG;
    }

private:
    nixl_status_t result_ = NIXL_SUCCESS;
};

class GcsMD : public nixlBackendMD {
public:
    GcsMD(uint64_t dev_id, std::string name)
        : nixlBackendMD(true),
          devId(dev_id),
          objectName(std::move(name)) {}

    uint64_t devId;
    std::string objectName;
};

bool
validXfer(const nixl_xfer_op_t &op,
          const nixl_meta_dlist_t &local,
          const nixl_meta_dlist_t &remote) {
    if (op != NIXL_WRITE && op != NIXL_READ) {
        return false;
    }
    if (local.getType() != DRAM_SEG || remote.getType() != OBJ_SEG) {
        return false;
    }
    if (local.descCount() != remote.descCount()) {
        return false;
    }
    return true;
}

} // namespace

nixlGcsEngine::nixlGcsEngine(const nixlBackendInitParams *params)
    : nixlBackendEngine(params),
      executor_(std::make_shared<asio::thread_pool>(getNumThreads(params->customParams))),
      client_(nullptr),
      bucket_(getBucket(params->customParams)) {
    client_ = nixl_gcs_client_create();
    if (!client_) {
        throw std::runtime_error("GCS client initialization failed");
    }
    NIXL_INFO << "GCS backend initialized for bucket " << bucket_;
}

nixlGcsEngine::~nixlGcsEngine() {
    executor_->wait();
    nixl_gcs_client_destroy(client_);
}

nixl_status_t
nixlGcsEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &type, nixlBackendMD *&out) {
    if (type != DRAM_SEG && type != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (type == DRAM_SEG) {
        out = nullptr;
        return NIXL_SUCCESS;
    }
    auto name = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    devIdToObjectName_[mem.devId] = name;
    out = new GcsMD(mem.devId, std::move(name));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGcsEngine::deregisterMem(nixlBackendMD *meta) {
    auto *const md = static_cast<GcsMD *>(meta);
    if (md) {
        devIdToObjectName_.erase(md->devId);
        delete md;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGcsEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    resp.assign(descs.descCount(), std::nullopt);
    for (int i = 0; i < descs.descCount(); ++i) {
        const auto object =
            descs[i].metaInfo.empty() ? std::to_string(descs[i].devId) : descs[i].metaInfo;
        const auto exists = nixl_gcs_exists(client_, bucket_.c_str(), object.c_str());
        if (exists == 1) {
            resp[i] = nixl_query_resp_t{nixl_b_params_t{}};
        } else if (exists < 0) {
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGcsEngine::prepXfer(const nixl_xfer_op_t &op,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *) const {
    if (!validXfer(op, local, remote)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    handle = new GcsReqH();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGcsEngine::postXfer(const nixl_xfer_op_t &op,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *) const {
    auto *const req = static_cast<GcsReqH *>(handle);

    for (int i = 0; i < local.descCount(); ++i) {
        if (devIdToObjectName_.find(remote[i].devId) == devIdToObjectName_.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (op == NIXL_WRITE && remote[i].addr != 0) {
            NIXL_ERROR << "GCS backend only supports whole-object writes (offset 0)";
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }

    for (int i = 0; i < local.descCount(); ++i) {
        const auto l = local[i];
        const auto r = remote[i];
        const auto found = devIdToObjectName_.find(r.devId);

        const auto promise = std::make_shared<std::promise<nixl_status_t>>();
        req->futures.push_back(promise->get_future());
        auto *const client = client_;
        const auto bucket = bucket_;
        const auto object = found->second;

        try {
            asio::post(*executor_, [client, bucket, object, l, r, op, promise]() {
                try {
                    if (op == NIXL_WRITE) {
                        if (!nixl_gcs_put(client,
                                          bucket.c_str(),
                                          object.c_str(),
                                          reinterpret_cast<const void *>(l.addr),
                                          l.len,
                                          r.addr)) {
                            promise->set_value(NIXL_ERR_BACKEND);
                            return;
                        }
                    } else {
                        if (!nixl_gcs_get(client,
                                          bucket.c_str(),
                                          object.c_str(),
                                          reinterpret_cast<void *>(l.addr),
                                          l.len,
                                          r.addr)) {
                            promise->set_value(NIXL_ERR_BACKEND);
                            return;
                        }
                    }
                    promise->set_value(NIXL_SUCCESS);
                }
                catch (const std::exception &e) {
                    NIXL_ERROR << "GCS transfer failed: " << e.what();
                    promise->set_value(NIXL_ERR_BACKEND);
                }
            });
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to schedule GCS transfer: " << e.what();
            promise->set_value(NIXL_ERR_BACKEND);
        }
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlGcsEngine::checkXfer(nixlBackendReqH *handle) const {
    return static_cast<GcsReqH *>(handle)->status();
}

nixl_status_t
nixlGcsEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *const req = static_cast<GcsReqH *>(handle);
    if (req->status() == NIXL_IN_PROG) {
        return NIXL_ERR_BACKEND;
    }
    delete req;
    return NIXL_SUCCESS;
}
