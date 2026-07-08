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

#include "posix_worker_pool.h"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

struct nixlPosixWorkerPool::Worker {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> tasks;
    bool stop = false;
    std::thread thread;
};

namespace {

void
unpinThread(std::thread &thread) {
#ifdef __linux__
    const long cpu_count = sysconf(_SC_NPROCESSORS_CONF);
    if (cpu_count <= 0) {
        throw std::runtime_error("failed to determine configured CPU count");
    }

    cpu_set_t *cpus = CPU_ALLOC(static_cast<size_t>(cpu_count));
    if (cpus == nullptr) {
        throw std::bad_alloc();
    }

    const size_t cpus_size = CPU_ALLOC_SIZE(static_cast<size_t>(cpu_count));
    CPU_ZERO_S(cpus_size, cpus);
    for (long cpu = 0; cpu < cpu_count; ++cpu) {
        CPU_SET_S(static_cast<size_t>(cpu), cpus_size, cpus);
    }

    const int status = pthread_setaffinity_np(thread.native_handle(), cpus_size, cpus);
    CPU_FREE(cpus);
    if (status != 0) {
        throw std::system_error(status, std::generic_category(), "pthread_setaffinity_np");
    }
#else
    (void)thread;
    throw std::invalid_argument("POSIX thread_unpin is supported only on Linux");
#endif
}

} // namespace

nixlPosixWorkerPool::nixlPosixWorkerPool(size_t thread_num, bool thread_unpin) {
    if (thread_num == 0) {
        throw std::invalid_argument("POSIX thread_num must be at least 1");
    }

    workers_.reserve(thread_num);
    try {
        for (size_t i = 0; i < thread_num; ++i) {
            auto worker = std::make_unique<Worker>();
            worker->thread = std::thread([worker_ptr = worker.get()]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(worker_ptr->mutex);
                        worker_ptr->cv.wait(lock, [worker_ptr]() {
                            return worker_ptr->stop || !worker_ptr->tasks.empty();
                        });
                        if (worker_ptr->stop && worker_ptr->tasks.empty()) {
                            return;
                        }
                        task = std::move(worker_ptr->tasks.front());
                        worker_ptr->tasks.pop_front();
                    }
                    task();
                }
            });
            workers_.push_back(std::move(worker));
            if (thread_unpin) {
                unpinThread(workers_.back()->thread);
            }
        }
    }
    catch (...) {
        stopAndJoin();
        throw;
    }
}

nixlPosixWorkerPool::~nixlPosixWorkerPool() {
    stopAndJoin();
}

bool
nixlPosixWorkerPool::submit(size_t worker_index, std::function<void()> task) {
    if (worker_index >= workers_.size()) {
        throw std::out_of_range("POSIX worker index is out of range");
    }

    Worker &worker = *workers_[worker_index];
    {
        const std::lock_guard<std::mutex> lock(worker.mutex);
        if (worker.stop) {
            return false;
        }
        worker.tasks.emplace_back(std::move(task));
    }
    worker.cv.notify_one();
    return true;
}

void
nixlPosixWorkerPool::invoke(size_t worker_index, std::function<void()> task) {
    auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
    auto done = packaged->get_future();
    if (!submit(worker_index, [packaged]() { (*packaged)(); })) {
        throw std::runtime_error("POSIX worker pool is stopping");
    }
    done.get();
}

void
nixlPosixWorkerPool::stopAndJoin() noexcept {
    for (auto &worker : workers_) {
        {
            const std::lock_guard<std::mutex> lock(worker->mutex);
            worker->stop = true;
        }
        worker->cv.notify_one();
    }

    for (auto &worker : workers_) {
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
    }
}
