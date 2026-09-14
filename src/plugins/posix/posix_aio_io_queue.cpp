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

#include "io_queue.h"
#include "common/nixl_log.h"
#include <aio.h>
#include <cerrno>
#include <cstring>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

struct nixlPosixAioIO {
public:
    void *
    currentBuf() const {
        return static_cast<char *>(buf_) + completed_;
    }

    off_t
    currentOffset() const {
        return offset_ + static_cast<off_t>(completed_);
    }

    size_t
    remaining() const {
        return len_ - completed_;
    }

    void
    advance(size_t completed) {
        NIXL_ASSERT(completed <= remaining());
        completed_ += completed;
    }

    int fd_ = -1;
    void *buf_ = nullptr;
    off_t offset_ = 0;
    size_t len_ = 0;
    size_t completed_ = 0;
    bool read_ = false;
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_ = nullptr;
    bool cancel_requested_ = false;
    struct aiocb aio_{};
};

class nixlPosixIOQueueAIO : public nixlPosixIOQueueImpl<nixlPosixAioIO> {
public:
    nixlPosixIOQueueAIO(uint32_t ios_pool_size, uint32_t kernel_queue_size)
        : nixlPosixIOQueueImpl<nixlPosixAioIO>(ios_pool_size, kernel_queue_size) {}

    virtual nixl_status_t
    post(void) override;
    virtual nixl_status_t
    enqueue(int fd,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) override;
    virtual nixl_status_t
    poll(void) override;
    virtual unsigned
    cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) override;
    virtual ~nixlPosixIOQueueAIO() override;

private:
    nixl_status_t
    doCheckCompleted(void);
    void
    queueForSubmission(nixlPosixAioIO *io);
    void
    finishIO(nixlPosixAioIO *io, bool error);
    bool
    handleIOCompletion(nixlPosixAioIO *io, ssize_t result);
    void
    failQueuedIOs(void *ctx);
    void
    enterTerminalError();

    std::list<nixlPosixAioIO *> ios_in_flight_;
    bool terminal_error_ = false;
};

nixlPosixIOQueueAIO::~nixlPosixIOQueueAIO() {
    failQueuedIOs(nullptr);

    for (nixlPosixAioIO *const io : ios_in_flight_) {
        io->cancel_requested_ = true;
        aio_cancel(io->fd_, &io->aio_);
    }

    while (!ios_in_flight_.empty()) {
        const struct aiocb *const aio = &ios_in_flight_.front()->aio_;
        if (aio_suspend(&aio, 1, nullptr) < 0 && errno != EINTR) {
            NIXL_ERROR << "aio_suspend failed while draining queue: " << nixl_strerror(errno);
        }
        doCheckCompleted();
    }
}

nixl_status_t
nixlPosixIOQueueAIO::enqueue(int fd,
                             void *buf,
                             size_t len,
                             off_t offset,
                             bool read,
                             nixlPosixIOQueueDoneCb clb,
                             void *ctx) {
    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }
    if (free_ios_.empty()) {
        NIXL_ERROR << "No more free blocks available";
        return NIXL_ERR_NOT_ALLOWED;
    }

    nixlPosixAioIO *const io = free_ios_.front();
    free_ios_.pop_front();

    io->fd_ = fd;
    io->buf_ = buf;
    io->offset_ = offset;
    io->len_ = len;
    io->completed_ = 0;
    io->read_ = read;
    io->clb_ = clb;
    io->ctx_ = ctx;
    io->cancel_requested_ = false;

    queueForSubmission(io);

    return NIXL_SUCCESS;
}

void
nixlPosixIOQueueAIO::queueForSubmission(nixlPosixAioIO *io) {
    std::memset(&io->aio_, 0, sizeof(io->aio_));
    io->aio_.aio_fildes = io->fd_;
    io->aio_.aio_buf = io->currentBuf();
    io->aio_.aio_nbytes = io->remaining();
    io->aio_.aio_offset = io->currentOffset();
    ios_to_submit_.push_back(io);
}

// Note: post() must return NIXL_IN_PROG in case of success
nixl_status_t
nixlPosixIOQueueAIO::post(void) {
    if (terminal_error_ || ios_to_submit_.empty()) {
        return NIXL_IN_PROG; // No blocks to submit
    }

    const int num_ios = std::min(MAX_IO_SUBMIT_BATCH_SIZE, (int)ios_to_submit_.size());
    for (int i = 0; i < num_ios; i++) {
        nixlPosixAioIO *const io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        const int ret = io->read_ ? aio_read(&io->aio_) : aio_write(&io->aio_);

        if (ret < 0) {
            const int error = errno;
            if (error == EAGAIN || error == EINTR) {
                ios_to_submit_.push_front(io);
                return NIXL_IN_PROG;
            }

            if (error == ENOSYS) {
                NIXL_ERROR << "aio submission terminal error: " << nixl_strerror(error);
                ios_to_submit_.push_front(io);
                enterTerminalError();
                return NIXL_IN_PROG;
            }

            NIXL_ERROR << "aio submission failed: " << nixl_strerror(error);
            void *const ctx = io->ctx_;
            finishIO(io, true);
            failQueuedIOs(ctx);
            return NIXL_IN_PROG;
        }

        ios_in_flight_.push_back(io);
    }

    return NIXL_IN_PROG;
}

void
nixlPosixIOQueueAIO::finishIO(nixlPosixAioIO *io, bool error) {
    if (io->clb_) {
        if (error) {
            io->clb_(io->ctx_, 0, 1);
        } else {
            io->clb_(io->ctx_, static_cast<uint32_t>(io->len_), 0);
        }
    }
    free_ios_.push_back(io);
}

bool
nixlPosixIOQueueAIO::handleIOCompletion(nixlPosixAioIO *io, ssize_t result) {
    if (terminal_error_) {
        finishIO(io, true);
        return true;
    }

    if (static_cast<size_t>(result) > io->remaining()) {
        finishIO(io, true);
        return true;
    }

    const size_t completed = static_cast<size_t>(result);
    if (completed == 0 && io->remaining() != 0) {
        finishIO(io, true);
        return true;
    }

    io->advance(completed);
    if (io->remaining() == 0) {
        finishIO(io, false);
        return false;
    }

    if (io->cancel_requested_) {
        finishIO(io, true);
        return true;
    }

    NIXL_DEBUG << "POSIX AIO operation completed partially: " << completed << " bytes completed, "
               << io->remaining() << " remaining; resubmitting remainder";
    queueForSubmission(io);
    return false;
}

inline nixl_status_t
nixlPosixIOQueueAIO::doCheckCompleted(void) {
    if (ios_in_flight_.empty()) {
        return free_ios_.size() == ios_pool_size_ ? NIXL_SUCCESS : NIXL_IN_PROG;
    }

    int num_ios = std::min(MAX_IO_CHECK_COMPLETED_BATCH_SIZE, (int)ios_in_flight_.size());
    for (auto it = ios_in_flight_.begin(); it != ios_in_flight_.end() && num_ios > 0;) {
        nixlPosixAioIO *const io = *it;
        const int status = aio_error(&io->aio_);
        if (status == EINPROGRESS) {
            const auto pending = it++;
            ios_in_flight_.splice(ios_in_flight_.end(), ios_in_flight_, pending);
            --num_ios;
            continue;
        }

        const ssize_t result = aio_return(&io->aio_);
        it = ios_in_flight_.erase(it);
        void *const ctx = io->ctx_;
        if (status != 0 || result < 0) {
            NIXL_DEBUG << "POSIX AIO operation failed: " << nixl_strerror(status ? status : errno);
            finishIO(io, true);
            failQueuedIOs(ctx);
        } else if (handleIOCompletion(io, result)) {
            failQueuedIOs(ctx);
        }
        --num_ios;
    }

    return free_ios_.size() == ios_pool_size_ ? NIXL_SUCCESS : NIXL_IN_PROG;
}

void
nixlPosixIOQueueAIO::failQueuedIOs(void *ctx) {
    for (auto it = ios_to_submit_.begin(); it != ios_to_submit_.end();) {
        nixlPosixAioIO *const io = *it;
        if (ctx && io->ctx_ != ctx) {
            ++it;
            continue;
        }

        it = ios_to_submit_.erase(it);
        finishIO(io, true);
    }
}

void
nixlPosixIOQueueAIO::enterTerminalError() {
    terminal_error_ = true;
    failQueuedIOs(nullptr);

    for (nixlPosixAioIO *const io : ios_in_flight_) {
        io->cancel_requested_ = true;
        if (aio_cancel(io->fd_, &io->aio_) == -1) {
            NIXL_DEBUG << "aio_cancel failed after terminal error: " << nixl_strerror(errno);
        }
    }
}

unsigned
nixlPosixIOQueueAIO::cancel(void *ctx, nixlPosixIOQueueCancelDoneCb) {
    if (!ctx) {
        return 0;
    }

    failQueuedIOs(ctx);
    for (nixlPosixAioIO *const io : ios_in_flight_) {
        if (io->ctx_ != ctx) {
            continue;
        }

        io->cancel_requested_ = true;
        const int result = aio_cancel(io->fd_, &io->aio_);
        if (result == -1) {
            NIXL_DEBUG << "aio_cancel failed: " << nixl_strerror(errno);
        } else {
            NIXL_ASSERT(result == AIO_CANCELED || result == AIO_NOTCANCELED ||
                        result == AIO_ALLDONE);
        }
    }

    // aio_cancel() has no separately completing asynchronous cancellation request.
    return 0;
}

nixl_status_t
nixlPosixIOQueueAIO::poll(void) {
    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }

    const nixl_status_t completion_status = doCheckCompleted();
    if (completion_status == NIXL_SUCCESS) {
        return NIXL_SUCCESS;
    }

    return post();
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueAIO>(ios_pool_size, kernel_queue_size);
}
