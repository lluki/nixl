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

#ifndef NIXL_SRC_PLUGINS_POSIX_IO_URING_QUEUE_UTILS_H
#define NIXL_SRC_PLUGINS_POSIX_IO_URING_QUEUE_UTILS_H

#include <liburing.h>

#include <cstddef>
#include <sys/types.h>

inline void
nixlPosixPrepareUringSqe(struct io_uring_sqe *sqe,
                         int fd,
                         void *buf,
                         size_t len,
                         off_t offset,
                         bool read,
                         bool force_async = true) {
    if (read) {
        io_uring_prep_read(sqe, fd, buf, len, offset);
    } else {
        io_uring_prep_write(sqe, fd, buf, len, offset);
    }

    if (force_async) {
        io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
    }
}

#endif // NIXL_SRC_PLUGINS_POSIX_IO_URING_QUEUE_UTILS_H
