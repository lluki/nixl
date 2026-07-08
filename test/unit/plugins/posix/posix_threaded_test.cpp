/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kThreadNum = 4;
constexpr size_t kRequestNum = 8;
constexpr size_t kTransferSize = 4096;

struct Request {
    std::string path;
    void *buffer = nullptr;
    std::unique_ptr<nixl_reg_dlist_t> file_reg;
    std::unique_ptr<nixl_reg_dlist_t> dram_reg;
    std::unique_ptr<nixl_xfer_dlist_t> file_xfer;
    std::unique_ptr<nixl_xfer_dlist_t> dram_xfer;
    nixlXferReqH *handle = nullptr;
};

int
fail(const std::string &message) {
    std::cerr << message << std::endl;
    return 1;
}

bool
waitAll(nixlAgent &agent, std::vector<Request> &requests) {
    size_t remaining = requests.size();
    std::vector<bool> done(requests.size(), false);
    while (remaining != 0) {
        for (size_t i = 0; i < requests.size(); ++i) {
            if (done[i]) {
                continue;
            }
            const nixl_status_t status = agent.getXferStatus(requests[i].handle);
            if (status < 0) {
                return false;
            }
            if (status == NIXL_SUCCESS) {
                done[i] = true;
                --remaining;
            }
        }
    }
    return true;
}

bool
prepareFilesAndMemory(nixlAgent &agent, std::vector<Request> &requests) {
    requests.reserve(kRequestNum);
    for (size_t i = 0; i < kRequestNum; ++i) {
        Request request;
        request.path = "/tmp/nixl_posix_threaded_" + std::to_string(getpid()) + "_" +
            std::to_string(i) + ".bin";
        std::remove(request.path.c_str());
        if (auto *file = std::fopen(request.path.c_str(), "wb")) {
            std::fseek(file, kTransferSize - 1, SEEK_SET);
            std::fputc(0, file);
            std::fclose(file);
        } else {
            return false;
        }

        if (posix_memalign(&request.buffer, sysconf(_SC_PAGESIZE), kTransferSize) != 0) {
            std::remove(request.path.c_str());
            return false;
        }
        std::memset(request.buffer, static_cast<int>(i + 1), kTransferSize);

        request.file_reg = std::make_unique<nixl_reg_dlist_t>(FILE_SEG);
        nixlBlobDesc file_desc;
        file_desc.addr = 0;
        file_desc.len = kTransferSize;
        file_desc.devId = i;
        file_desc.metaInfo = std::string("rw:") + request.path;
        request.file_reg->addDesc(file_desc);

        request.dram_reg = std::make_unique<nixl_reg_dlist_t>(DRAM_SEG);
        nixlBlobDesc dram_desc;
        dram_desc.addr = reinterpret_cast<uintptr_t>(request.buffer);
        dram_desc.len = kTransferSize;
        dram_desc.devId = 0;
        request.dram_reg->addDesc(dram_desc);

        if (agent.registerMem(*request.file_reg) != NIXL_SUCCESS) {
            std::free(request.buffer);
            std::remove(request.path.c_str());
            return false;
        }
        if (agent.registerMem(*request.dram_reg) != NIXL_SUCCESS) {
            agent.deregisterMem(*request.file_reg);
            std::free(request.buffer);
            std::remove(request.path.c_str());
            return false;
        }

        request.file_xfer = std::make_unique<nixl_xfer_dlist_t>(request.file_reg->trim());
        request.dram_xfer = std::make_unique<nixl_xfer_dlist_t>(request.dram_reg->trim());
        requests.push_back(std::move(request));
    }
    return true;
}

bool
runDirection(nixlAgent &agent, std::vector<Request> &requests, nixl_xfer_op_t operation) {
    for (auto &request : requests) {
        if (agent.createXferReq(operation,
                                *request.dram_xfer,
                                *request.file_xfer,
                                "POSIXThreadedTest",
                                request.handle) != NIXL_SUCCESS) {
            return false;
        }
        if (agent.postXferReq(request.handle) < 0) {
            return false;
        }
    }

    if (!waitAll(agent, requests)) {
        return false;
    }

    for (auto &request : requests) {
        agent.releaseXferReq(request.handle);
        request.handle = nullptr;
    }
    return true;
}

void
cleanup(nixlAgent &agent, std::vector<Request> &requests) {
    for (auto &request : requests) {
        if (request.handle) {
            agent.releaseXferReq(request.handle);
        }
        if (request.file_reg) {
            agent.deregisterMem(*request.file_reg);
        }
        if (request.dram_reg) {
            agent.deregisterMem(*request.dram_reg);
        }
        std::free(request.buffer);
        std::remove(request.path.c_str());
    }
}

bool
invalidThreadNumRejected(bool use_uring) {
    nixlAgentConfig cfg;
    nixlAgent agent("POSIXInvalidThreadNum", cfg);
    nixl_b_params_t params;
    params["thread_num"] = "0";
    params[use_uring ? "use_uring" : "use_aio"] = "true";
    nixlBackendH *backend = nullptr;
    return agent.createBackend("POSIX", params, backend) != NIXL_SUCCESS;
}

} // namespace

int
main(int argc, char **argv) {
    bool use_uring = false;
    bool thread_unpin = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--uring") {
            use_uring = true;
        } else if (arg == "--unpin") {
            thread_unpin = true;
        } else {
            return fail("usage: posix_threaded_test [--uring] [--unpin]");
        }
    }

    if (!invalidThreadNumRejected(use_uring)) {
        return fail("thread_num=0 did not reject backend creation");
    }

    nixlAgentConfig cfg;
    nixlAgent agent("POSIXThreadedTest", cfg);
    nixl_b_params_t params;
    params["thread_num"] = std::to_string(kThreadNum);
    params["thread_unpin"] = thread_unpin ? "true" : "false";
    if (use_uring) {
        params["use_uring"] = "true";
        params["use_aio"] = "false";
    } else {
        params["use_aio"] = "true";
        params["use_uring"] = "false";
    }

    nixlBackendH *backend = nullptr;
    if (agent.createBackend("POSIX", params, backend) != NIXL_SUCCESS || !backend) {
        return fail("failed to create threaded POSIX backend");
    }

    std::vector<Request> requests;
    if (!prepareFilesAndMemory(agent, requests)) {
        cleanup(agent, requests);
        return fail("threaded path-mode registration failed");
    }

    if (!runDirection(agent, requests, NIXL_WRITE)) {
        cleanup(agent, requests);
        return fail("parallel writes failed");
    }

    for (auto &request : requests) {
        std::memset(request.buffer, 0, kTransferSize);
    }

    if (!runDirection(agent, requests, NIXL_READ)) {
        cleanup(agent, requests);
        return fail("parallel reads failed");
    }

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto *data = static_cast<const unsigned char *>(requests[i].buffer);
        for (size_t j = 0; j < kTransferSize; ++j) {
            if (data[j] != static_cast<unsigned char>(i + 1)) {
                cleanup(agent, requests);
                return fail("read data validation failed");
            }
        }
    }

    cleanup(agent, requests);
    std::cout << "POSIX threaded path opens and parallel " << (use_uring ? "io_uring" : "Linux AIO")
              << " transfers: OK" << std::endl;
    return 0;
}
