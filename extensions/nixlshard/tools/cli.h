/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <charconv>
#include <chrono>
#include <csignal>
#include <stdexcept>
#include <string>
#include <thread>

namespace nixlshard::cli {
inline volatile std::sig_atomic_t stopped = 0;
inline void stop(int) { stopped = 1; }
inline void wait() {
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    while (!stopped) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
inline std::string value(int &index, int argc, char **argv) {
    if (++index >= argc) throw std::invalid_argument("missing option value");
    return argv[index];
}
template <typename T>
T integer(const std::string &text) {
    T result{};
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("invalid integer: " + text);
    return result;
}
} // namespace nixlshard::cli
