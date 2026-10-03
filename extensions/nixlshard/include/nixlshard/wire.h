/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "nixlshard/types.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>

namespace nixlshard::wire {
constexpr std::size_t max_frame = 16 * 1024 * 1024;
class Timeout : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class Writer {
public:
    std::string data;
    void u8(std::uint8_t value);
    void u32(std::uint32_t value);
    void u64(std::uint64_t value);
    void str(std::string_view value);
};
class Reader {
public:
    explicit Reader(std::string_view value) : data_(value) {}
    std::uint8_t u8();
    std::uint32_t u32();
    std::uint64_t u64();
    std::string str(std::size_t limit = max_frame);
    void finish() const;
private:
    std::string_view data_;
    std::size_t offset_ = 0;
};
class Connection {
public:
    Connection(const Endpoint &, unsigned timeout_ms);
    ~Connection();
    Connection(const Connection &) = delete;
    std::string call(std::string_view message, unsigned timeout_ms);
    bool usable() const noexcept { return fd_ >= 0; }
    void close() noexcept;
private:
    int fd_ = -1;
};
class Server {
public:
    using Handler = std::function<std::string(std::string_view)>;
    Server(Endpoint, Handler, unsigned timeout_ms, std::size_t max_connections = 16);
    ~Server();
    Endpoint endpoint() const;
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
enum Op : std::uint8_t {
    hello = 1, exists = 2, load = 3, cleanup = 4,
    register_owner = 5, announce = 6, lookup = 7, directory = 8,
    load_batch = 9, load_batch_trace = 10
};
} // namespace nixlshard::wire
