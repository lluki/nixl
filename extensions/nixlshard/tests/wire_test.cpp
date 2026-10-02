/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/wire.h"
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <functional>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
using Clock = std::chrono::steady_clock;
void rejects(const std::function<void()> &fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception &) { caught = true; }
    CHECK(caught);
}
struct Socket {
    int fd;
    explicit Socket(const Endpoint &ep) {
        fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        CHECK(fd >= 0);
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(ep.port);
        CHECK(::inet_pton(AF_INET, ep.host.c_str(), &address.sin_addr) == 1);
        if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address))) {
            ::close(fd); throw std::runtime_error("raw test connection failed");
        }
    }
    ~Socket() { ::close(fd); }
    void prefix(uint32_t length) {
        uint32_t encoded = htonl(length);
        CHECK(::send(fd, &encoded, sizeof(encoded), MSG_NOSIGNAL) == sizeof(encoded));
    }
    void closed(unsigned timeout_ms = 1000) {
        pollfd p{fd, POLLIN, 0};
        CHECK(::poll(&p, 1, static_cast<int>(timeout_ms)) > 0);
        char byte;
        auto n = ::recv(fd, &byte, 1, MSG_DONTWAIT);
        CHECK(n == 0 || (n < 0 && errno == ECONNRESET));
    }
};
void encoding_and_endpoints() {
    wire::Writer writer;
    writer.u8(255); writer.u32(0x01020304); writer.u64(0x0102030405060708ULL);
    writer.str(std::string("a\0b", 3));
    CHECK(static_cast<unsigned char>(writer.data[1]) == 1 && static_cast<unsigned char>(writer.data[4]) == 4);
    wire::Reader reader(writer.data);
    CHECK(reader.u8() == 255 && reader.u32() == 0x01020304 && reader.u64() == 0x0102030405060708ULL);
    CHECK(reader.str() == std::string("a\0b", 3)); reader.finish();
    rejects([&] { reader.u8(); });
    wire::Reader extra("xx"); rejects([&] { extra.finish(); });
    auto start = Clock::now();
    rejects([] { wire::Connection connection(Endpoint{"localhost", 12345}, 50); });
    rejects([] { wire::Connection connection(Endpoint{"does-not-exist.invalid", 12345}, 50); });
    rejects([] { wire::Connection connection(Endpoint{"::1", 12345}, 50); });
    rejects([] { wire::Connection connection(Endpoint{"127.0.0.1", 12345}, 0); });
    CHECK(Clock::now() - start < std::chrono::milliseconds(500));
    rejects([] { wire::Server server(Endpoint{}, [](auto) { return "x"; }, 100, 4097); });
}
void malformed_frame_and_idle() {
    std::atomic<unsigned> calls{0};
    wire::Server server(Endpoint{}, [&](auto value) { ++calls; return std::string(value); }, 40);
    for (auto length : {uint32_t(0), uint32_t(wire::max_frame + 1)}) {
        Socket raw(server.endpoint()); raw.prefix(length); raw.closed();
    }
    {
        Socket raw(server.endpoint()); raw.prefix(4);
        CHECK(::send(raw.fd, "a", 1, MSG_NOSIGNAL) == 1);
        raw.closed(); // partial frame expires without invoking the handler
    }
    CHECK(calls == 0);
    wire::Connection idle(server.endpoint(), 1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(idle.call(std::string("a\0b", 3), 1000) == std::string("a\0b", 3));
    CHECK(calls == 1);
}
void deadline_and_server_close() {
    wire::Server delayed(Endpoint{}, [](auto value) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); return std::string(value);
    }, 1000);
    wire::Connection connection(delayed.endpoint(), 1000);
    auto start = Clock::now();
    rejects([&] { connection.call("late", 20); });
    CHECK(!connection.usable() && Clock::now() - start < std::chrono::milliseconds(500));
    delayed.close();
    wire::Server partial(Endpoint{}, [](auto value) { return std::string(value); }, 5000);
    Socket raw(partial.endpoint()); raw.prefix(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    start = Clock::now(); partial.close(); partial.close();
    CHECK(Clock::now() - start < std::chrono::milliseconds(500));
    raw.closed();
}
void connection_limit_and_reuse() {
    wire::Server server(Endpoint{}, [](auto value) { return std::string(value); }, 100, 1);
    {
        wire::Connection first(server.endpoint(), 1000);
        CHECK(first.call("first", 1000) == "first"); // ensure active registration
        wire::Connection excess(server.endpoint(), 1000);
        rejects([&] { excess.call("second", 1000); });
        CHECK(!excess.usable());
    }
    auto deadline = Clock::now() + std::chrono::seconds(2);
    bool recovered = false;
    while (Clock::now() < deadline && !recovered) {
        try {
            wire::Connection next(server.endpoint(), 1000);
            recovered = next.call("next", 1000) == "next";
        } catch (...) {}
        if (!recovered) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(recovered);
}
} // namespace
int main() {
    try {
        encoding_and_endpoints(); malformed_frame_and_idle();
        deadline_and_server_close(); connection_limit_and_reuse();
        std::cout << "wire tests passed (4 suites)\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
