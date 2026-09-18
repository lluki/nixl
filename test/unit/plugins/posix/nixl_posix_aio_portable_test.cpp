/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>

#include "io_queue.h"

namespace {
struct completionState {
    int completions = 0;
    int errors = 0;
};

void
completionCallback(void *ctx, uint32_t, int error) {
    auto *state = static_cast<completionState *>(ctx);
    state->completions++;
    state->errors += error != 0;
}

#define POSIX_AIO_CHECK(condition)                                                        \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            std::cerr << "POSIX_AIO_CHECK failed at line " << __LINE__ << ": " #condition \
                      << std::endl;                                                       \
            return 1;                                                                     \
        }                                                                                 \
    } while (false)
} // namespace

int
main() {
    constexpr int max_poll_iterations = 2000;
    constexpr auto poll_pause = std::chrono::microseconds(50);
    auto queue = nixlPosixIOQueue::instantiate("POSIXAIO", 64, 16);
    std::array<char, 4096> buffer{};
    completionState failed, succeeded;
    char path[] = "/tmp/nixl_posix_aio_portable_test_XXXXXX";
    int fd = mkstemp(path);

    POSIX_AIO_CHECK(queue);
    POSIX_AIO_CHECK(fd >= 0);
    unlink(path);
    POSIX_AIO_CHECK(
        queue->enqueue(-1, buffer.data(), buffer.size(), 0, false, completionCallback, &failed) ==
        NIXL_SUCCESS);
    POSIX_AIO_CHECK(
        queue->enqueue(
            fd, buffer.data(), buffer.size(), 0, false, completionCallback, &succeeded) ==
        NIXL_SUCCESS);
    nixl_status_t status = queue->post();
    POSIX_AIO_CHECK(status == NIXL_IN_PROG);
    for (int i = 0;
         i < max_poll_iterations && (failed.completions == 0 || succeeded.completions == 0);
         i++) {
        (void)queue->poll();
        std::this_thread::sleep_for(poll_pause);
    }
    POSIX_AIO_CHECK(failed.completions == 1 && failed.errors == 1);
    POSIX_AIO_CHECK(succeeded.completions == 1 && succeeded.errors == 0);
    POSIX_AIO_CHECK(queue->poll() == NIXL_SUCCESS);
    POSIX_AIO_CHECK(failed.completions == 1 && failed.errors == 1);
    POSIX_AIO_CHECK(succeeded.completions == 1 && succeeded.errors == 0);
    close(fd);
    return 0;
}
