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

#include "io_uring_queue_utils.h"

#include <array>
#include <iostream>
#include <string>

#include "nixl.h"
#include "nixl_params.h"

namespace {

bool
checkSqe(bool read, bool force_async) {
    struct io_uring_sqe sqe {};

    std::array<char, 4096> buffer{};
    nixlPosixPrepareUringSqe(&sqe, -1, buffer.data(), buffer.size(), 0, read, force_async);

    const __u8 expected_opcode = read ? IORING_OP_READ : IORING_OP_WRITE;
    const bool async_is_set = (sqe.flags & IOSQE_ASYNC) != 0;
    return sqe.opcode == expected_opcode && async_is_set == force_async;
}

bool
checkDefaultSqe(bool read) {
    struct io_uring_sqe sqe {};

    std::array<char, 4096> buffer{};
    nixlPosixPrepareUringSqe(&sqe, -1, buffer.data(), buffer.size(), 0, read);

    const __u8 expected_opcode = read ? IORING_OP_READ : IORING_OP_WRITE;
    return sqe.opcode == expected_opcode && (sqe.flags & IOSQE_ASYNC) != 0;
}

bool
createBackend(const std::string &agent_name, nixl_b_params_t params) {
    nixlAgentConfig config;
    nixlAgent agent(agent_name, config);
    nixlBackendH *backend = nullptr;
    return agent.createBackend("POSIX", params, backend) == NIXL_SUCCESS && backend != nullptr;
}

int
fail(const char *message) {
    std::cerr << message << std::endl;
    return 1;
}

} // namespace

int
main() {
    if (!checkDefaultSqe(true) || !checkDefaultSqe(false)) {
        return fail("default io_uring READ/WRITE SQEs did not set IOSQE_ASYNC");
    }
    if (!checkSqe(true, false) || !checkSqe(false, false)) {
        return fail("uring_force_async=false left IOSQE_ASYNC on READ/WRITE SQEs");
    }
    if (!checkSqe(true, true) || !checkSqe(false, true)) {
        return fail("uring_force_async=true did not set IOSQE_ASYNC on READ/WRITE SQEs");
    }

    nixl_b_params_t default_uring;
    default_uring["use_uring"] = "true";
    if (!createBackend("POSIXUringDefaultAsync", default_uring)) {
        return fail("io_uring backend rejected the default uring_force_async=true");
    }

    nixl_b_params_t disabled_uring;
    disabled_uring["use_uring"] = "true";
    disabled_uring["uring_force_async"] = "false";
    if (!createBackend("POSIXUringDisabledAsync", disabled_uring)) {
        return fail("io_uring backend rejected uring_force_async=false");
    }

    nixl_b_params_t forced_uring;
    forced_uring["use_uring"] = "true";
    forced_uring["uring_force_async"] = "true";
    if (!createBackend("POSIXUringForcedAsync", forced_uring)) {
        return fail("io_uring backend rejected uring_force_async=true");
    }

    nixl_b_params_t force_without_uring;
    force_without_uring["uring_force_async"] = "true";
    if (createBackend("POSIXForcedAsyncWithoutUring", force_without_uring)) {
        return fail("uring_force_async was accepted without use_uring=true");
    }

#ifdef HAVE_LINUXAIO
    nixl_b_params_t force_with_aio;
    force_with_aio["use_aio"] = "true";
    force_with_aio["uring_force_async"] = "true";
    if (createBackend("POSIXForcedAsyncWithAIO", force_with_aio)) {
        return fail("uring_force_async was accepted with Linux AIO");
    }
#endif

#ifdef HAVE_POSIXAIO
    nixl_b_params_t force_with_posix_aio;
    force_with_posix_aio["use_posix_aio"] = "true";
    force_with_posix_aio["uring_force_async"] = "true";
    if (createBackend("POSIXForcedAsyncWithPosixAIO", force_with_posix_aio)) {
        return fail("uring_force_async was accepted with POSIX AIO");
    }
#endif

    std::cout << "POSIX uring_force_async SQE flags and configuration: OK" << std::endl;
    return 0;
}
