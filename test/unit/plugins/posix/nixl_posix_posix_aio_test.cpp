/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aio.h>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include "io_queue.h"
#include "posix_backend.h"

namespace {
constexpr size_t block_size = 4096;
constexpr size_t first_partial = 777;
constexpr size_t second_partial = 1234;
constexpr int max_poll_iterations = 2000;
enum class completion_mode_t { PASS, PARTIAL, ZERO, DELAYED, DELAY_64 };

struct submission {
    uintptr_t buffer;
    size_t length;
    off_t offset;
};

struct operation {
    int status;
    ssize_t result;
    bool ready;
};

completion_mode_t completion_mode = completion_mode_t::PASS;
int cancel_result = AIO_NOTCANCELED;
int submission_error = 0;
int submission_errors_remaining = 0;
bool release_delayed = false;
int cancel_completions = 0;
int submission_attempts = 0;
std::vector<submission> submissions;
std::unordered_map<const aiocb *, operation> operations;

struct completionState {
    int completions = 0;
    int errors = 0;
    uint32_t last_size = 0;
};

void
completionCallback(void *ctx, uint32_t size, int error) {
    auto *const state = static_cast<completionState *>(ctx);
    state->completions++;
    state->errors += error != 0;
    state->last_size = size;
}

void
cancelCompletionCallback(void *) {
    cancel_completions++;
}

void
resetMock(completion_mode_t mode) {
    completion_mode = mode;
    cancel_result = AIO_NOTCANCELED;
    submission_error = 0;
    submission_errors_remaining = 0;
    release_delayed = false;
    cancel_completions = 0;
    submission_attempts = 0;
    submissions.clear();
    operations.clear();
}

int
submit(aiocb *control_block, bool read) {
    submission_attempts++;
    if (submission_errors_remaining > 0) {
        submission_errors_remaining--;
        errno = submission_error;
        return -1;
    }

    submissions.push_back({reinterpret_cast<uintptr_t>(control_block->aio_buf),
                           control_block->aio_nbytes,
                           control_block->aio_offset});
    size_t transfer_size = control_block->aio_nbytes;
    if (completion_mode == completion_mode_t::PARTIAL) {
        if (submissions.size() == 1) {
            transfer_size = first_partial;
        } else if (submissions.size() == 2) {
            transfer_size = second_partial;
        }
    } else if (completion_mode == completion_mode_t::ZERO) {
        transfer_size = 0;
    } else if (completion_mode == completion_mode_t::DELAYED && cancel_result == AIO_NOTCANCELED) {
        transfer_size = first_partial;
    }
    ssize_t result = 0;
    if (transfer_size != 0) {
        result = read ? pread(control_block->aio_fildes,
                              const_cast<void *>(control_block->aio_buf),
                              transfer_size,
                              control_block->aio_offset) :
                        pwrite(control_block->aio_fildes,
                               const_cast<const void *>(control_block->aio_buf),
                               transfer_size,
                               control_block->aio_offset);
    }
    const bool delayed = completion_mode == completion_mode_t::DELAYED ||
        (completion_mode == completion_mode_t::DELAY_64 && submissions.size() <= 64);
    operations[control_block] = {result < 0 ? errno : 0, result, !delayed};
    return 0;
}

struct posixAioTest {
    int fd = -1;
    std::vector<std::array<char, block_size>> buffers;
    std::unique_ptr<nixlPosixIOQueue> queue;

    explicit posixAioTest(size_t buffer_count = 128, size_t pool_size = 128)
        : buffers(buffer_count),
          queue(nixlPosixIOQueue::instantiate("POSIXAIO", pool_size, 16)) {
        char path[] = "/tmp/nixl_posix_aio_test_XXXXXX";
        fd = mkstemp(path);
        NIXL_ASSERT_ALWAYS(fd >= 0);
        unlink(path);
    }

    ~posixAioTest() {
        release_delayed = true;
        queue.reset();
        close(fd);
    }

    nixl_status_t
    enqueue(completionState &state, size_t index = 0, bool read = false, off_t offset = 0) {
        return queue->enqueue(
            fd, buffers[index].data(), block_size, offset, read, completionCallback, &state);
    }

    nixl_status_t
    drain() {
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
            status = queue->poll();
        }
        return status;
    }
};

struct posixAioRequest {
    nixl_meta_dlist_t local{DRAM_SEG};
    nixl_meta_dlist_t remote{FILE_SEG};
    nixl_xfer_op_t operation;
    std::unique_ptr<nixlPosixBackendReqH> request;

    posixAioRequest(posixAioTest &test,
                    nixlPosixFileMD &file_md,
                    int first,
                    int count,
                    nixl_xfer_op_t op = NIXL_WRITE,
                    size_t offset_bias = 0)
        : operation(op) {
        for (int i = first; i < first + count; i++) {
            local.addDesc(nixlMetaDesc(
                reinterpret_cast<uintptr_t>(test.buffers[i].data()), block_size, 0, nullptr));
            remote.addDesc(
                nixlMetaDesc(offset_bias + i * block_size, block_size, test.fd, &file_md));
        }
        request = std::make_unique<nixlPosixBackendReqH>(operation, local, remote, test.queue);
    }
};

nixl_status_t
waitFor(nixlPosixBackendReqH &request, nixl_status_t status = NIXL_IN_PROG) {
    for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
        status = request.checkXfer();
    }
    return status;
}

#define POSIX_AIO_CHECK(condition) NIXL_ASSERT_ALWAYS(condition)
} // namespace

extern "C" int
aio_read(struct aiocb *control_block) noexcept {
    return submit(control_block, true);
}

extern "C" int
aio_write(struct aiocb *control_block) noexcept {
    return submit(control_block, false);
}

extern "C" int
aio_error(const struct aiocb *control_block) noexcept {
    operation &op = operations.at(control_block);
    if (!op.ready && release_delayed) {
        op.ready = true;
    }
    return op.ready ? op.status : EINPROGRESS;
}

extern "C" ssize_t
aio_return(struct aiocb *control_block) noexcept {
    const ssize_t result = operations.at(control_block).result;
    operations.erase(control_block);
    return result;
}

extern "C" int
aio_cancel(int, struct aiocb *control_block) noexcept {
    operation &op = operations.at(control_block);
    if (cancel_result == -1) {
        errno = EINVAL;
        op.ready = true;
        return -1;
    }
    if (cancel_result == AIO_CANCELED) {
        op.status = ECANCELED;
        op.result = -1;
    }
    op.ready = true;
    return cancel_result;
}

int
main() {
    for (const bool read : {true, false}) {
        resetMock(completion_mode_t::PARTIAL);
        posixAioTest test(1);
        constexpr size_t offset = 321;
        std::array<char, block_size> expected;
        std::memset(expected.data(), read ? 0x5a : 0x1, expected.size());
        std::memset(test.buffers[0].data(), read ? 0 : 0x1, block_size);
        if (read) {
            POSIX_AIO_CHECK(pwrite(test.fd, expected.data(), block_size, offset) ==
                            static_cast<ssize_t>(block_size));
            std::memset(test.buffers[0].data(), 0, block_size);
        }
        completionState state;
        POSIX_AIO_CHECK(test.enqueue(state, 0, read, offset) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG && test.drain() == NIXL_SUCCESS);
        POSIX_AIO_CHECK(submissions.size() == 3 && state.completions == 1 && state.errors == 0 &&
                        state.last_size == block_size);
        const uintptr_t buffer = reinterpret_cast<uintptr_t>(test.buffers[0].data());
        const std::array<size_t, 3> completed = {0, first_partial, first_partial + second_partial};
        for (size_t i = 0; i < submissions.size(); i++) {
            POSIX_AIO_CHECK(submissions[i].buffer == buffer + completed[i] &&
                            submissions[i].offset == static_cast<off_t>(offset + completed[i]) &&
                            submissions[i].length == block_size - completed[i]);
        }
        std::array<char, block_size> actual;
        if (read) {
            std::memcpy(actual.data(), test.buffers[0].data(), block_size);
        } else {
            POSIX_AIO_CHECK(pread(test.fd, actual.data(), block_size, offset) ==
                            static_cast<ssize_t>(block_size));
        }
        POSIX_AIO_CHECK(std::memcmp(actual.data(), expected.data(), block_size) == 0);
    }
    for (const bool read : {true, false}) {
        resetMock(read ? completion_mode_t::PASS : completion_mode_t::ZERO);
        posixAioTest test(1);
        nixlPosixFileMD file_md(test.fd, "");
        const posixAioRequest request(test, file_md, 0, 1, read ? NIXL_READ : NIXL_WRITE);
        POSIX_AIO_CHECK(request.request->postXfer() == NIXL_IN_PROG);
        POSIX_AIO_CHECK(waitFor(*request.request) == NIXL_ERR_BACKEND && submissions.size() == 1);
    }
    for (const int error : {EAGAIN, EINTR}) {
        resetMock(completion_mode_t::PASS);
        submission_error = error;
        submission_errors_remaining = 1;
        posixAioTest test(1);
        completionState state;
        POSIX_AIO_CHECK(test.enqueue(state) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        POSIX_AIO_CHECK(submission_attempts == 1 && submissions.empty() && state.completions == 0);
        POSIX_AIO_CHECK(test.drain() == NIXL_SUCCESS);
        POSIX_AIO_CHECK(submission_attempts == 2 && submissions.size() == 1 &&
                        state.completions == 1 && state.errors == 0);
    }
    {
        resetMock(completion_mode_t::PASS);
        submission_error = ENOSYS;
        submission_errors_remaining = 1;
        posixAioTest test(3);
        completionState first;
        completionState second;
        completionState rejected;
        POSIX_AIO_CHECK(test.enqueue(first, 0) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.enqueue(second, 1) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        POSIX_AIO_CHECK(submission_attempts == 1 && submissions.empty());
        POSIX_AIO_CHECK(first.completions == 1 && first.errors == 1);
        POSIX_AIO_CHECK(second.completions == 1 && second.errors == 1);
        POSIX_AIO_CHECK(test.queue->enqueue(test.fd,
                                            test.buffers[2].data(),
                                            block_size,
                                            0,
                                            false,
                                            completionCallback,
                                            &rejected) == NIXL_ERR_BACKEND);
        POSIX_AIO_CHECK(rejected.completions == 0);
        POSIX_AIO_CHECK(test.queue->poll() == NIXL_ERR_BACKEND);
    }

    struct cancelCase {
        int result;
        bool read;
        int errors;
    };

    const std::array cases = {cancelCase{AIO_CANCELED, false, 1},
                              cancelCase{AIO_NOTCANCELED, false, 1},
                              cancelCase{AIO_NOTCANCELED, true, 1},
                              cancelCase{AIO_ALLDONE, false, 0},
                              cancelCase{-1, false, 0}};
    for (const cancelCase &test_case : cases) {
        resetMock(completion_mode_t::DELAYED);
        cancel_result = test_case.result;
        posixAioTest test(1);
        POSIX_AIO_CHECK(!test_case.read ||
                        pwrite(test.fd, test.buffers[0].data(), block_size, 0) ==
                            static_cast<ssize_t>(block_size));
        completionState state;
        POSIX_AIO_CHECK(test.enqueue(state, 0, test_case.read) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        POSIX_AIO_CHECK(test.queue->cancel(&state, cancelCompletionCallback) == 0);
        POSIX_AIO_CHECK(state.completions == 0 && cancel_completions == 0);
        POSIX_AIO_CHECK(test.drain() == NIXL_SUCCESS && state.completions == 1);
        POSIX_AIO_CHECK(state.errors == test_case.errors && submissions.size() == 1 &&
                        cancel_completions == 0);
    }
    {
        resetMock(completion_mode_t::DELAY_64);
        posixAioTest test(128, 65);
        nixlPosixFileMD file_md(test.fd, "");
        const posixAioRequest active(test, file_md, 0, 64);
        POSIX_AIO_CHECK(active.request->postXfer() == NIXL_IN_PROG && submissions.size() == 64);
        completionState canceled;
        POSIX_AIO_CHECK(test.enqueue(canceled, 64) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->cancel(&canceled, cancelCompletionCallback) == 0);
        POSIX_AIO_CHECK(canceled.completions == 1 && canceled.errors == 1 &&
                        cancel_completions == 0);
        POSIX_AIO_CHECK(submissions.size() == 64);

        completion_mode = completion_mode_t::PASS;
        completionState unrelated;
        POSIX_AIO_CHECK(test.enqueue(unrelated, 66) == NIXL_SUCCESS);
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        POSIX_AIO_CHECK(test.queue->poll() == NIXL_IN_PROG && unrelated.completions == 0);
        POSIX_AIO_CHECK(test.queue->poll() == NIXL_IN_PROG && unrelated.completions == 1);
        release_delayed = true;
        POSIX_AIO_CHECK(waitFor(*active.request) == NIXL_SUCCESS);

        completionState reusable;
        for (size_t i = 0; i < 65; i++) {
            POSIX_AIO_CHECK(test.enqueue(reusable, i, false, i * block_size) == NIXL_SUCCESS);
        }
        POSIX_AIO_CHECK(test.queue->post() == NIXL_IN_PROG && test.drain() == NIXL_SUCCESS);
        POSIX_AIO_CHECK(reusable.completions == 65 && reusable.errors == 0);
    }
    return 0;
}
