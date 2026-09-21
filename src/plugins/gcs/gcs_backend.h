/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_GCS_GCS_BACKEND_H
#define NIXL_SRC_PLUGINS_GCS_GCS_BACKEND_H

#include "backend/backend_engine.h"
#include "gcs_sdk_wrapper.h"
#include <asio.hpp>
#include <memory>
#include <string>
#include <unordered_map>

class nixlGcsEngine : public nixlBackendEngine {
public:
    explicit nixlGcsEngine(const nixlBackendInitParams *init_params);
    ~nixlGcsEngine() override;

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
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &, const nixl_mem_t &, nixlBackendMD *&) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *) override;
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &, std::vector<nixl_query_resp_t> &) const override;

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &,
             const nixl_meta_dlist_t &,
             const nixl_meta_dlist_t &,
             const std::string &,
             nixlBackendReqH *&,
             const nixl_opt_b_args_t * = nullptr) const override;
    nixl_status_t
    postXfer(const nixl_xfer_op_t &,
             const nixl_meta_dlist_t &,
             const nixl_meta_dlist_t &,
             const std::string &,
             nixlBackendReqH *&,
             const nixl_opt_b_args_t * = nullptr) const override;
    nixl_status_t
    checkXfer(nixlBackendReqH *) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *) const override;

private:
    std::shared_ptr<asio::thread_pool> executor_;
    nixl_gcs_client *client_;
    std::string bucket_;
    std::unordered_map<uint64_t, std::string> devIdToObjectName_;
};

#endif // NIXL_SRC_PLUGINS_GCS_GCS_BACKEND_H
