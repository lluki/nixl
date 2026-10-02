/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/wire.h"
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <algorithm>

namespace nixlshard::wire {
namespace {
using Clock = std::chrono::steady_clock;
void wait_fd(int fd, short events, Clock::time_point deadline) {
    for (;;) {
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0) throw Timeout("control deadline exceeded");
        pollfd p{fd, events, 0};
        int rc = ::poll(&p, 1, static_cast<int>(std::min<std::int64_t>(left, 1000)));
        if (rc > 0) {
            if (p.revents & events) return;
            throw std::runtime_error("control connection closed");
        }
        if (rc < 0 && errno != EINTR) throw std::runtime_error("control poll failed");
    }
}
void send_all(int fd, const char *data, std::size_t size, Clock::time_point deadline) {
    while (size) {
        wait_fd(fd, POLLOUT, deadline);
        auto n = ::send(fd, data, size, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) throw std::runtime_error("control send failed");
        data += n; size -= static_cast<std::size_t>(n);
    }
}
void recv_all(int fd, char *data, std::size_t size, Clock::time_point deadline) {
    while (size) {
        wait_fd(fd, POLLIN, deadline);
        auto n = ::recv(fd, data, size, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) throw std::runtime_error("control receive failed");
        data += n; size -= static_cast<std::size_t>(n);
    }
}
void send_frame(int fd, std::string_view value, Clock::time_point deadline) {
    if (value.size() > max_frame) throw std::invalid_argument("control frame too large");
    std::uint32_t length = htonl(static_cast<std::uint32_t>(value.size()));
    send_all(fd, reinterpret_cast<const char *>(&length), sizeof(length), deadline);
    send_all(fd, value.data(), value.size(), deadline);
}
std::string recv_frame(int fd, Clock::time_point deadline) {
    std::uint32_t length;
    recv_all(fd, reinterpret_cast<char *>(&length), sizeof(length), deadline);
    auto size = ntohl(length);
    if (!size || size > max_frame) throw std::invalid_argument("invalid control frame");
    std::string value(size, '\0');
    recv_all(fd, value.data(), size, deadline);
    return value;
}
int connect_socket(const Endpoint &endpoint, unsigned timeout_ms) {
    if (endpoint.host.empty() || !endpoint.port || !timeout_ms)
        throw std::invalid_argument("invalid endpoint or connection timeout");
    // Initial protocol deliberately uses numeric IPv4 endpoints: synchronous DNS
    // resolution would escape the progress thread's connection/close deadline.
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(endpoint.port);
    if (endpoint.host.find('\0') != std::string::npos ||
        ::inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr) != 1)
        throw std::invalid_argument("control endpoint requires a numeric IPv4 address");
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) throw std::runtime_error("control socket creation failed");
    int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    int rc = ::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
    if (rc == 0) return fd;
    if (errno == EINPROGRESS) {
        try {
            wait_fd(fd, POLLOUT, deadline);
            int error = 0; socklen_t size = sizeof(error);
            if (!::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) && !error) return fd;
        } catch (...) {}
    }
    ::close(fd);
    throw std::runtime_error("endpoint connection failed");
}
}
void Writer::u8(std::uint8_t value) { data.push_back(static_cast<char>(value)); }
void Writer::u32(std::uint32_t value) {
    for (int i = 3; i >= 0; --i) u8(static_cast<std::uint8_t>(value >> (i * 8)));
}
void Writer::u64(std::uint64_t value) {
    for (int i = 7; i >= 0; --i) u8(static_cast<std::uint8_t>(value >> (i * 8)));
}
void Writer::str(std::string_view value) {
    if (value.size() > max_frame) throw std::invalid_argument("control string too large");
    u32(static_cast<std::uint32_t>(value.size())); data.append(value);
}
std::uint8_t Reader::u8() {
    if (offset_ >= data_.size()) throw std::invalid_argument("truncated control message");
    return static_cast<std::uint8_t>(data_[offset_++]);
}
std::uint32_t Reader::u32() {
    std::uint32_t v = 0; for (int i = 0; i < 4; ++i) v = (v << 8) | u8(); return v;
}
std::uint64_t Reader::u64() {
    std::uint64_t v = 0; for (int i = 0; i < 8; ++i) v = (v << 8) | u8(); return v;
}
std::string Reader::str(std::size_t limit) {
    auto n = u32();
    if (n > limit || n > data_.size() - offset_) throw std::invalid_argument("invalid control string");
    std::string v(data_.substr(offset_, n)); offset_ += n; return v;
}
void Reader::finish() const {
    if (offset_ != data_.size()) throw std::invalid_argument("extra control bytes");
}
Connection::Connection(const Endpoint &ep, unsigned timeout) : fd_(connect_socket(ep, timeout)) {}
Connection::~Connection() { close(); }
void Connection::close() noexcept {
    if (fd_ >= 0) { ::shutdown(fd_, SHUT_RDWR); ::close(fd_); fd_ = -1; }
}
std::string Connection::call(std::string_view message, unsigned timeout_ms) {
    try {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        send_frame(fd_, message, deadline); return recv_frame(fd_, deadline);
    } catch (...) { close(); throw; }
}
struct Server::Impl {
    Endpoint ep;
    Handler handler;
    unsigned timeout;
    std::size_t limit;
    int listener = -1;
    std::atomic<bool> stop{false};
    std::thread acceptor;
    struct Client { int fd; std::thread worker; std::shared_ptr<std::atomic<bool>> done; };
    std::mutex mutex;
    std::vector<Client> clients;
    Impl(Endpoint endpoint, Handler cb, unsigned timeout_ms, std::size_t max_connections)
        : ep(std::move(endpoint)), handler(std::move(cb)), timeout(timeout_ms), limit(max_connections) {
        if (!timeout || !limit || limit > 4096)
            throw std::invalid_argument("invalid control resource limit");
        clients.reserve(limit);
        listener = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (listener < 0) throw std::runtime_error("listen socket failed");
        int one = 1; ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(ep.port);
        if (::inet_pton(AF_INET, ep.host.c_str(), &address.sin_addr) != 1 ||
            ::bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ||
            ::listen(listener, static_cast<int>(limit))) {
            ::close(listener); listener = -1; throw std::runtime_error("control bind failed");
        }
        socklen_t size = sizeof(address);
        ::getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size);
        ep.port = ntohs(address.sin_port);
        try { acceptor = std::thread([this] { run(); }); }
        catch (...) { ::close(listener); listener = -1; throw; }
    }
    void reap() {
        std::lock_guard lock(mutex);
        for (auto it = clients.begin(); it != clients.end();) {
            if (it->done->load()) { it->worker.join(); ::close(it->fd); it = clients.erase(it); }
            else ++it;
        }
    }
    void run() {
        while (!stop.load()) {
            reap();
            pollfd p{listener, POLLIN, 0};
            if (::poll(&p, 1, 50) <= 0) continue;
            int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (fd < 0) continue;
            int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            try {
                std::lock_guard lock(mutex);
                if (clients.size() >= limit || stop.load()) { ::close(fd); continue; }
                auto done = std::make_shared<std::atomic<bool>>(false);
                // Register a nonjoinable client before starting work. Allocation or
                // thread creation failures can close this socket without terminating
                // the process through destruction of an untracked joinable thread.
                clients.push_back(Client{fd, std::thread{}, done});
                try {
                    clients.back().worker = std::thread([this, fd, done] {
                        try {
                            while (!stop.load()) {
                                // Idle channels stay alive; each nonempty request has a bounded frame deadline.
                                pollfd p{fd, POLLIN, 0};
                                if (::poll(&p, 1, 50) <= 0) continue;
                                if (!(p.revents & POLLIN)) break;
                                auto request = recv_frame(fd, Clock::now() + std::chrono::milliseconds(timeout));
                                auto response = handler(request);
                                send_frame(fd, response, Clock::now() + std::chrono::milliseconds(timeout));
                            }
                        } catch (...) {}
                        done->store(true);
                    });
                } catch (...) { clients.pop_back(); throw; }
            } catch (...) { ::close(fd); }
        }
    }
    void close() {
        if (stop.exchange(true)) return;
        ::shutdown(listener, SHUT_RDWR);
        if (acceptor.joinable()) acceptor.join();
        {
            std::lock_guard lock(mutex);
            for (auto &c : clients) ::shutdown(c.fd, SHUT_RDWR);
        }
        for (auto &c : clients) { if (c.worker.joinable()) c.worker.join(); ::close(c.fd); }
        clients.clear(); ::close(listener); listener = -1;
    }
    ~Impl() { close(); }
};
Server::Server(Endpoint ep, Handler cb, unsigned timeout, std::size_t limit)
    : impl_(std::make_unique<Impl>(std::move(ep), std::move(cb), timeout, limit)) {}
Server::~Server() = default;
Endpoint Server::endpoint() const { return impl_->ep; }
void Server::close() { impl_->close(); }
} // namespace nixlshard::wire
