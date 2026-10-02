/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nixlshard {
enum class Status {
    success, not_found, not_ready, busy, no_space, invalid_input,
    io_error, timeout, canceled
};
inline const char *status_name(Status status) noexcept {
    switch (status) {
    case Status::success: return "success";
    case Status::not_found: return "not_found";
    case Status::not_ready: return "not_ready";
    case Status::busy: return "busy";
    case Status::no_space: return "no_space";
    case Status::invalid_input: return "invalid_input";
    case Status::io_error: return "io_error";
    case Status::timeout: return "timeout";
    case Status::canceled: return "canceled";
    }
    return "io_error";
}
struct Segment {
    std::uint64_t registration = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
};
struct Object {
    std::string key;
    std::vector<Segment> segments;
    std::string hint;
};
struct Endpoint {
    std::string host = "127.0.0.1";
    std::uint16_t port = 0;
};
} // namespace nixlshard
