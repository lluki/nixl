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

#ifndef NIXL_SRC_PLUGINS_POSIX_POSIX_WORKER_POOL_H
#define NIXL_SRC_PLUGINS_POSIX_POSIX_WORKER_POOL_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

class nixlPosixWorkerPool {
public:
    nixlPosixWorkerPool(size_t thread_num, bool thread_unpin);
    ~nixlPosixWorkerPool();

    nixlPosixWorkerPool(const nixlPosixWorkerPool &) = delete;
    nixlPosixWorkerPool &
    operator=(const nixlPosixWorkerPool &) = delete;

    size_t
    size() const noexcept {
        return workers_.size();
    }

    // Select workers in strict round-robin order. Backend calls are serialized by
    // the agent, but an atomic keeps this helper safe for direct unit tests too.
    size_t
    nextWorker() noexcept {
        return next_worker_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    }

    // Queue work on one specific worker.
    bool
    submit(size_t worker_index, std::function<void()> task);

    // Run work on one specific worker and wait for it to finish. Exceptions from
    // the task are propagated to the caller.
    void
    invoke(size_t worker_index, std::function<void()> task);

private:
    struct Worker;

    void
    stopAndJoin() noexcept;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<size_t> next_worker_{0};
};

#endif // NIXL_SRC_PLUGINS_POSIX_POSIX_WORKER_POOL_H
