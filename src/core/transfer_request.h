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
#ifndef NIXL_SRC_CORE_TRANSFER_REQUEST_H
#define NIXL_SRC_CORE_TRANSFER_REQUEST_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "nixl_types.h"
#include "backend_engine.h"
#include "telemetry.h"
#include "common/nixl_duration.h"
#include "common/nixl_log.h"
#include "tracing/trace_context.h"

enum nixl_telemetry_stat_status_t {
    NIXL_TELEMETRY_POST = 0,
    NIXL_TELEMETRY_POST_AND_FINISH = 1,
    NIXL_TELEMETRY_FINISH = 2
};

class nixlRemoteSection;

// Weak ref to the owning remote section: expires when that registration generation is
// invalidated or replaced, which is the staleness signal for handles created against it.
using nixl_remote_section_weak_t = std::weak_ptr<nixlRemoteSection>;

// Contains pointers to corresponding backend engine and its handler, and populated
// and verified DescLists, and other state and metadata needed for a NIXL transfer
class nixlXferReqH {
public:
    nixlXferReqH(const std::string &remote_agent,
                 const nixl_xfer_op_t backend_op,
                 const nixl_mem_t local_type,
                 const nixl_mem_t remote_type,
                 const size_t desc_count,
                 const nixl_remote_section_weak_t &remote_section_ref,
                 const nixl::trace::TraceContext &trace_context,
                 const std::uint64_t trace_request_id);

    nixlXferReqH(nixlXferReqH &&) = delete;
    nixlXferReqH(const nixlXferReqH &) = delete;

    void
    operator=(nixlXferReqH &&) = delete;
    void
    operator=(const nixlXferReqH &) = delete;

    // Request lifecycle, derived from `status`:
    //   created   : status == NIXL_ERR_NOT_POSTED (prepXfer done, never posted)
    //   submitted : status == NIXL_IN_PROG (posted, getXferStatus() has not yet reported a
    //               terminal status, even if the backend already finished internally)
    //   terminal  : any other status (success or an error, not reposted since)
    // The backend handle may only be released in the created or terminal state.
    [[nodiscard]] bool
    isSubmitted() const noexcept {
        return status == NIXL_IN_PROG;
    }

    // Logs the client error for releasing a request that is still submitted.
    void
    logReleaseWhileSubmitted() const noexcept {
        NIXL_ERROR << "\n"
                      "################################################################\n"
                      "# NIXL CLIENT ERROR: transfer request released while still in progress\n"
                      "# (submitted state, backend '"
                   << (engine ? engine->getType() : std::string("?"))
                   << "').\n"
                      "# The backend request handle is LEAKED and the user buffers may still\n"
                      "# be read or overwritten by the transfer. Call cancelXferReq() if the\n"
                      "# transfer has to be abandoned, and poll getXferStatus() until it is\n"
                      "# not NIXL_IN_PROG before releasing the request.\n"
                      "################################################################";
    }

    // Must not throw, poll or block. Only called from the destructor and for requests that are
    // not submitted; a submitted request has its backend handle leaked on purpose (the backend
    // may still use it, so freeing it would risk a use-after-free).
    void
    releaseBackendHandle() noexcept {
        if (engine == nullptr || backendHandle == nullptr) {
            return;
        }
        if (isSubmitted()) {
            logReleaseWhileSubmitted();
            backendHandle = nullptr; // leaked on purpose, never touched again
            return;
        }
        try {
            engine->releaseReqH(backendHandle);
        }
        catch (...) {
            NIXL_ERROR << "backend '" << engine->getType()
                       << "' threw while releasing a transfer request handle";
        }
        backendHandle = nullptr; // never release the handle a second time
    }

    ~nixlXferReqH() {
        releaseBackendHandle();
    }

    void
    updateRequestStats(nixlTelemetry *telemetry, nixl_telemetry_stat_status_t stat_status);

    [[nodiscard]] const nixl::trace::TraceContext &
    traceContext() const noexcept {
        return traceContext_;
    }

    /** @brief Trace context of the latest post handed to the backend, which that
     *         post's nixl::xfer.complete carries; traceContext() before any post. */
    [[nodiscard]] const nixl::trace::TraceContext &
    tracePostContext() const noexcept {
        return tracePostContext_;
    }

    [[nodiscard]] std::uint64_t
    traceRequestId() const noexcept {
        return traceRequestId_;
    }

    [[nodiscard]] std::uint64_t
    traceExecution() const noexcept {
        return traceExecution_.load(std::memory_order_relaxed);
    }

    friend class nixlAgent;

private:
    nixlBackendEngine *engine = nullptr;
    nixlBackendReqH *backendHandle = nullptr;

    nixl_meta_dlist_t initiatorDescs;
    nixl_meta_dlist_t targetDescs;

    const std::string remoteAgent;
    const nixl_remote_section_weak_t remoteSection;
    const nixl::trace::TraceContext traceContext_;
    nixl::trace::TraceContext tracePostContext_;
    bool traceRequestSpanIdUsed_ = false;
    const std::uint64_t traceRequestId_;
    std::atomic<std::uint64_t> traceExecution_ = 0;
    nixl_blob_t notifMsg;
    bool hasNotif = false;

    const nixl_xfer_op_t backendOp;
    nixl_status_t status = NIXL_ERR_NOT_POSTED;
    // True once cancelXferReq() succeeded for the current submission; reset on every post.
    std::atomic<bool> cancelRequested = false;

    nixl_xfer_telem_t telemetry;
    nixlTime::nixlDuration timer;
};

struct nixlDlistH {
    using descs_t =
        std::unordered_map<nixlBackendEngine *, std::unique_ptr<nixl_meta_stride_dlist_t>>;

    nixlDlistH(const std::string &remote_agent,
               descs_t &&descs,
               const nixl_remote_section_weak_t &remote_section_ref);

    const std::string remoteAgent; // Empty means "local".
    const descs_t descs;
    // Empty for local descriptor lists.
    const nixl_remote_section_weak_t remoteSectionRef;
};

#endif
