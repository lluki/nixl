/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
Status status(wire::Connection &connection, const wire::Writer &request) {
    auto response = connection.call(request.data, 1000);
    wire::Reader reader(response);
    auto result = static_cast<Status>(reader.u8());
    reader.finish();
    return result;
}
wire::Writer registration(const std::string &owner, const std::string &incarnation,
                          const std::string &host = "127.0.0.1", uint32_t port = 12345) {
    wire::Writer request;
    request.u8(wire::register_owner); request.str(owner); request.str(incarnation);
    request.str(host); request.u32(port);
    return request;
}
wire::Writer announcement(const std::string &owner, const std::string &incarnation,
                          uint64_t sequence, const std::vector<std::string> &keys) {
    wire::Writer request;
    request.u8(wire::announce); request.str(owner); request.str(incarnation);
    request.u64(sequence); request.u32(static_cast<uint32_t>(keys.size()));
    for (const auto &key : keys) request.str(key);
    return request;
}
using Location = std::pair<std::string, std::string>;
std::vector<Location> lookup(wire::Connection &connection, const std::vector<std::string> &keys) {
    wire::Writer request; request.u8(wire::lookup); request.u32(static_cast<uint32_t>(keys.size()));
    for (const auto &key : keys) request.str(key);
    auto response = connection.call(request.data, 1000);
    wire::Reader reader(response);
    CHECK(static_cast<Status>(reader.u8()) == Status::success);
    CHECK(reader.u32() == keys.size());
    std::vector<Location> result;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto owner = reader.str(), incarnation = reader.str();
        result.emplace_back(std::move(owner), std::move(incarnation));
    }
    reader.finish();
    return result;
}
Status directory(wire::Connection &connection, const std::string &owner,
                 std::string *incarnation = nullptr, Endpoint *endpoint = nullptr) {
    wire::Writer request; request.u8(wire::directory); request.str(owner);
    auto response = connection.call(request.data, 1000);
    wire::Reader reader(response);
    auto result = static_cast<Status>(reader.u8());
    if (result == Status::success) {
        auto inc = reader.str(), host = reader.str();
        auto port = reader.u32();
        if (incarnation) *incarnation = std::move(inc);
        if (endpoint) *endpoint = Endpoint{std::move(host), static_cast<uint16_t>(port)};
    }
    reader.finish();
    return result;
}
void owner_and_sequence() {
    MetadataServer server(Endpoint{});
    wire::Connection connection(server.endpoint(), 1000);
    CHECK(directory(connection, "owner") == Status::not_found);
    CHECK(status(connection, announcement("owner", "one", 1, {"a"})) == Status::not_found);
    CHECK(status(connection, registration("owner", "one")) == Status::success);
    std::string incarnation; Endpoint endpoint;
    CHECK(directory(connection, "owner", &incarnation, &endpoint) == Status::success);
    CHECK(incarnation == "one" && endpoint.host == "127.0.0.1" && endpoint.port == 12345);
    CHECK(status(connection, announcement("owner", "one", 10, {"a", "a"})) == Status::success);
    CHECK(status(connection, announcement("owner", "one", 9, {"stale"})) == Status::success);
    CHECK(status(connection, announcement("owner", "one", 10, {"duplicate"})) == Status::success);
    auto results = lookup(connection, {"a", "stale", "duplicate", "miss"});
    CHECK(results[0] == Location("owner", "one"));
    CHECK(results[1] == Location() && results[2] == Location() && results[3] == Location());
    CHECK(status(connection, registration("owner", "two", "10.0.0.1", 54321)) == Status::success);
    CHECK(lookup(connection, {"a"})[0] == Location());
    CHECK(status(connection, announcement("owner", "one", 11, {"a"})) == Status::not_found);
    CHECK(status(connection, registration("owner", "one")) == Status::not_found);
    CHECK(status(connection, announcement("owner", "two", 0, {"a"})) == Status::success);
    CHECK(lookup(connection, {"a"})[0] == Location("owner", "two"));
    CHECK(directory(connection, "owner", &incarnation, &endpoint) == Status::success);
    CHECK(incarnation == "two" && endpoint.host == "10.0.0.1" && endpoint.port == 54321);
    // An immutable key may have another owner; hints are advisory single locations.
    CHECK(status(connection, registration("other", "three")) == Status::success);
    CHECK(status(connection, announcement("other", "three", 1, {"a"})) == Status::success);
    CHECK(status(connection, registration("owner", "four")) == Status::success);
    CHECK(lookup(connection, {"a"})[0] == Location("other", "three"));
}
void ttl_and_capacity() {
    MetadataServer server(Endpoint{}, 2, 80);
    wire::Connection connection(server.endpoint(), 1000);
    CHECK(status(connection, registration("first", "one")) == Status::success);
    CHECK(status(connection, registration("second", "two")) == Status::success);
    CHECK(status(connection, registration("third", "three")) == Status::no_space);
    CHECK(status(connection, announcement("first", "one", 1, {"a", "b"})) == Status::success);
    CHECK(status(connection, announcement("second", "two", 1, {"c"})) == Status::no_space);
    CHECK(lookup(connection, {"a", "b", "c"})[2] == Location());
    CHECK(status(connection, registration("first", "new")) == Status::success); // old hints reclaimed
    CHECK(status(connection, announcement("second", "two", 1, {"c"})) == Status::success); // failed batch did not advance seq
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (directory(connection, "second") == Status::success) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(lookup(connection, {"c"})[0] == Location());
    CHECK(status(connection, announcement("second", "two", 2, {"c"})) == Status::not_found);
    CHECK(status(connection, registration("third", "three")) == Status::success);
    CHECK(status(connection, announcement("third", "three", 1, {"c", "d"})) == Status::success);
    CHECK(lookup(connection, {"c", "d"})[0] == Location("third", "three"));
}
void malformed_and_batch_bounds() {
    MetadataServer server(Endpoint{});
    wire::Connection connection(server.endpoint(), 1000);
    CHECK(status(connection, registration("", "inc")) == Status::invalid_input);
    CHECK(status(connection, registration("owner", "inc", "127.0.0.1", 0)) == Status::invalid_input);
    CHECK(status(connection, registration("owner", "inc", "127.0.0.1", 65536)) == Status::invalid_input);
    CHECK(status(connection, registration("owner", "inc", std::string("host\0bad", 8))) == Status::invalid_input);
    auto trailing = registration("owner", "inc"); trailing.u8(0);
    CHECK(status(connection, trailing) == Status::invalid_input);
    CHECK(directory(connection, "owner") == Status::not_found);
    CHECK(status(connection, registration("owner", "inc")) == Status::success);
    std::vector<std::string> keys;
    for (int i = 0; i < 128; ++i) keys.push_back("key" + std::to_string(i));
    CHECK(status(connection, announcement("owner", "inc", 1, keys)) == Status::success);
    CHECK(lookup(connection, keys).size() == 128);
    CHECK(lookup(connection, {}).empty());
    keys.push_back("too-many");
    CHECK(status(connection, announcement("owner", "inc", 2, keys)) == Status::invalid_input);
    auto malformed = announcement("owner", "inc", 2, {"partial"}); malformed.u8(255);
    CHECK(status(connection, malformed) == Status::invalid_input);
    CHECK(lookup(connection, {"partial"})[0] == Location());
    CHECK(status(connection, announcement("owner", "inc", 2, {std::string("a\0b", 3)})) == Status::success);
    CHECK(lookup(connection, {std::string("a\0b", 3)})[0] == Location("owner", "inc"));
    CHECK(status(connection, announcement("owner", "inc", 3, {std::string(65537, 'x')})) == Status::invalid_input);
    CHECK(status(connection, announcement("owner", "inc", 3, {""})) == Status::invalid_input);
    wire::Writer truncated; truncated.u8(wire::announce); truncated.str("owner");
    CHECK(status(connection, truncated) == Status::invalid_input);
    wire::Writer unknown; unknown.u8(255);
    CHECK(status(connection, unknown) == Status::invalid_input);
    CHECK(status(connection, announcement("owner", "inc", 3, {"valid"})) == Status::success);
}
void retired_incarnation_bound() {
    MetadataServer server(Endpoint{}, 1);
    wire::Connection connection(server.endpoint(), 1000);
    CHECK(status(connection, registration("owner", "one")) == Status::success);
    CHECK(status(connection, registration("owner", "two")) == Status::success);
    CHECK(status(connection, registration("owner", "one")) == Status::not_found);
    CHECK(status(connection, registration("owner", "three")) == Status::busy);
    CHECK(status(connection, registration("owner", "two")) == Status::success);
    CHECK(status(connection, announcement("owner", "two", 1, {"key"})) == Status::success);
}
void concurrent_connections_and_close() {
    MetadataServer server(Endpoint{});
    std::vector<std::thread> workers;
    std::atomic<bool> valid{true};
    for (int i = 0; i < 8; ++i) workers.emplace_back([&, i] {
        try {
            wire::Connection connection(server.endpoint(), 1000);
            auto name = "owner" + std::to_string(i);
            if (status(connection, registration(name, "inc")) != Status::success ||
                status(connection, announcement(name, "inc", 1, {name})) != Status::success ||
                lookup(connection, {name})[0] != Location(name, "inc")) valid = false;
        } catch (...) { valid = false; }
    });
    for (auto &worker : workers) worker.join();
    CHECK(valid);
    server.close(); server.close();
}
} // namespace
int main() {
    try {
        owner_and_sequence();
        ttl_and_capacity();
        malformed_and_batch_bounds();
        retired_incarnation_bound();
        concurrent_connections_and_close();
        std::cout << "metadata tests passed (5 suites)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
