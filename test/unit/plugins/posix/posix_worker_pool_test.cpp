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

#include <sched.h>

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>

namespace {

bool
getAffinity(cpu_set_t &mask) {
    CPU_ZERO(&mask);
    return sched_getaffinity(0, sizeof(mask), &mask) == 0;
}

int
firstCpu(const cpu_set_t &mask) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            return cpu;
        }
    }
    return -1;
}

bool
isOnlyCpu(const cpu_set_t &mask, int cpu) {
    return CPU_COUNT(&mask) == 1 && CPU_ISSET(cpu, &mask);
}

int
fail(const char *message) {
    std::cerr << message << std::endl;
    return 1;
}

} // namespace

int
main() {
    try {
        nixlPosixWorkerPool invalid(0, false);
        return fail("thread_num=0 was accepted");
    }
    catch (const std::invalid_argument &) {
    }

    cpu_set_t original;
    if (!getAffinity(original)) {
        return fail("sched_getaffinity failed");
    }
    const int pinned_cpu = firstCpu(original);
    if (pinned_cpu < 0) {
        return fail("caller has no available CPU");
    }

    cpu_set_t pinned;
    CPU_ZERO(&pinned);
    CPU_SET(pinned_cpu, &pinned);
    if (sched_setaffinity(0, sizeof(pinned), &pinned) != 0) {
        return fail("failed to pin caller for affinity test");
    }

    {
        nixlPosixWorkerPool pool(4, false);
        for (size_t expected = 0; expected < 8; ++expected) {
            if (pool.nextWorker() != expected % 4) {
                sched_setaffinity(0, sizeof(original), &original);
                return fail("round-robin worker selection is not even");
            }
        }

        std::set<std::thread::id> thread_ids;
        for (size_t worker = 0; worker < pool.size(); ++worker) {
            cpu_set_t worker_mask;
            pool.invoke(worker, [&]() {
                thread_ids.insert(std::this_thread::get_id());
                if (!getAffinity(worker_mask)) {
                    throw std::runtime_error("worker sched_getaffinity failed");
                }
            });
            if (!isOnlyCpu(worker_mask, pinned_cpu)) {
                sched_setaffinity(0, sizeof(original), &original);
                return fail("thread_unpin=false did not preserve inherited caller affinity");
            }
        }
        if (thread_ids.size() != pool.size()) {
            sched_setaffinity(0, sizeof(original), &original);
            return fail("worker queues did not execute on distinct threads");
        }
    }

    if (CPU_COUNT(&original) > 1) {
        nixlPosixWorkerPool pool(4, true);
        for (size_t worker = 0; worker < pool.size(); ++worker) {
            cpu_set_t worker_mask;
            pool.invoke(worker, [&]() {
                if (!getAffinity(worker_mask)) {
                    throw std::runtime_error("worker sched_getaffinity failed");
                }
            });
            if (CPU_COUNT(&worker_mask) <= 1) {
                sched_setaffinity(0, sizeof(original), &original);
                return fail("thread_unpin=true left a worker pinned to the caller CPU");
            }
        }
    } else {
        std::cout << "thread_unpin expansion skipped: only one CPU is available" << std::endl;
    }

    if (sched_setaffinity(0, sizeof(original), &original) != 0) {
        return fail("failed to restore caller affinity");
    }

    std::cout << "POSIX worker pool scheduling and affinity: OK" << std::endl;
    return 0;
}
