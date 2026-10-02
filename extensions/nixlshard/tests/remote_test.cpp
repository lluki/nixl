/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
using Clock = std::chrono::steady_clock;
struct File {
    std::string path;
    File() {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-remote-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};
AgentConfig config(const std::string &name) {
    AgentConfig cfg; cfg.name = name; cfg.staging_slots = 2;
    cfg.staging_slot_bytes = 8192; cfg.workers = 1;
    cfg.max_inflight = 4; cfg.timeout_ms = 1500;
    return cfg;
}
void eventually(const std::function<bool()> &predicate, const std::string &description = "operation") {
    auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        if (Clock::now() >= deadline) throw std::runtime_error("timed out waiting for " + description);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
std::vector<Status> wait(Agent &agent, uint64_t handle) {
    std::optional<std::vector<Status>> result;
    eventually([&] { result = agent.poll(handle); return result.has_value(); });
    return *result;
}
Status load(Agent &agent, Object object) {
    auto handle = agent.batch_load({std::move(object)});
    auto result = wait(agent, handle);
    agent.release(handle);
    CHECK(result.size() == 1);
    return result[0];
}
std::string peer_identity(const Endpoint &endpoint) {
    wire::Connection connection(endpoint, 1000);
    wire::Writer request; request.u8(wire::hello);
    auto response = connection.call(request.data, 1000);
    wire::Reader reader(response);
    CHECK(static_cast<Status>(reader.u8()) == Status::success);
    auto incarnation = reader.str(); reader.str(); reader.finish();
    return incarnation;
}
void caller_hint_roundtrip_and_restart() {
    File file;
    auto owner_cfg = config("remote-owner");
    owner_cfg.disks = {{file.path, 16384 + 32 * 4096, 4096, 16384, true}};
    auto owner = std::make_unique<Agent>(owner_cfg);
    auto endpoint = owner->endpoint();
    auto old_incarnation = peer_identity(endpoint);
    std::array<uint8_t, 8192> source{}, destination{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((i * 7 + 13) % 251);
    auto src = owner->register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    Object store{"scatter-key", {{src, 1024, 3000}, {src, 17, 1500}}, {}};
    auto put = owner->batch_store({store});
    CHECK(wait(*owner, put) == std::vector<Status>{Status::success}); owner->release(put);
    CHECK(owner->batch_exists({"scatter-key"})[0]);
    {
        wire::Connection control(endpoint, 1000);
        const std::string request_id = "regression-delayed-load";
        for (int i = 0; i < 2; ++i) {
            wire::Writer cleanup; cleanup.u8(wire::cleanup);
            cleanup.str(old_incarnation); cleanup.str(request_id);
            auto response = control.call(cleanup.data, 1000);
            wire::Reader result(response);
            CHECK(static_cast<Status>(result.u8()) == Status::success && result.u8() == 1);
            result.finish();
        }
        auto before = destination;
        wire::Writer delayed; delayed.u8(wire::load); delayed.str(request_id);
        delayed.str(old_incarnation); delayed.str("scatter-key"); delayed.u64(4500);
        delayed.u64(reinterpret_cast<uintptr_t>(destination.data()));
        delayed.str("bogus-requester"); delayed.str("bogus-metadata");
        auto response = control.call(delayed.data, 1000);
        wire::Reader result(response);
        CHECK(static_cast<Status>(result.u8()) == Status::canceled); result.finish();
        CHECK(destination == before); // repeated cleanup keeps the late-write fence
    }
    auto reader_cfg = config("remote-reader"); reader_cfg.peers[owner_cfg.name] = endpoint;
    Agent reader(reader_cfg);
    auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(destination.data()), destination.size());
    eventually([&] { return reader.stats()["peer_connections"] >= 1; }, "caller-hint peer connection");
    try {
        eventually([&]() -> bool { return reader.batch_exists({"scatter-key"}, {owner_cfg.name})[0]; }, "caller-hint owner existence");
    } catch (...) {
        for (auto &[name, value] : reader.stats()) std::cerr << "reader " << name << "=" << value << '\n';
        throw;
    }
    Object get{"scatter-key", {{dst, 11, 1700}, {dst, 4000, 2800}}, owner_cfg.name};
    CHECK(load(reader, get) == Status::success);
    std::vector<uint8_t> expected(source.begin() + 1024, source.begin() + 4024);
    expected.insert(expected.end(), source.begin() + 17, source.begin() + 1517);
    std::vector<uint8_t> actual(destination.begin() + 11, destination.begin() + 1711);
    actual.insert(actual.end(), destination.begin() + 4000, destination.begin() + 6800);
    CHECK(actual == expected);
    CHECK(destination[0] == 0 && destination[1711] == 0 && destination[3999] == 0 && destination[6800] == 0);
    CHECK(load(reader, {"missing", {{dst, 0, 4500}}, owner_cfg.name}) == Status::not_found);
    auto start = Clock::now();
    CHECK(load(reader, {"scatter-key", {{dst, 0, 4500}}, "unknown-owner"}) == Status::not_ready);
    CHECK(Clock::now() - start < std::chrono::milliseconds(500));
    auto before = destination;
    CHECK(load(reader, {"scatter-key", {{dst, 0, 4499}}, owner_cfg.name}) == Status::invalid_input);
    CHECK(destination == before);
    CHECK(owner->stats()["ucx_write_bytes"] >= 4500);
    CHECK(reader.stats()["remote_read_bytes"] >= 4500);
    owner->deregister_memory(src);
    owner->close(); owner.reset();
    owner_cfg.disks[0].create = false; owner_cfg.listen = endpoint;
    owner = std::make_unique<Agent>(owner_cfg);
    // A live new connection carrying the old incarnation must fail before payload access.
    {
        wire::Connection connection(endpoint, 1000);
        wire::Writer request; request.u8(wire::exists); request.str(old_incarnation);
        request.u32(1); request.str("scatter-key");
        auto response = connection.call(request.data, 1000);
        wire::Reader result(response);
        CHECK(static_cast<Status>(result.u8()) == Status::not_ready); result.finish();
    }
    // Existence can drop the dead control channel without risking a payload target.
    reader.batch_exists({"scatter-key"}, {owner_cfg.name});
    eventually([&]() -> bool { return reader.batch_exists({"scatter-key"}, {owner_cfg.name})[0]; }, "restarted owner existence");
    destination.fill(0);
    CHECK(load(reader, get) == Status::success);
    actual.assign(destination.begin() + 11, destination.begin() + 1711);
    actual.insert(actual.end(), destination.begin() + 4000, destination.begin() + 6800);
    CHECK(actual == expected && peer_identity(endpoint) != old_incarnation);
    reader.deregister_memory(dst); reader.close(); owner->close();
}
void metadata_discovery_roundtrip() {
    File file;
    MetadataServer metadata(Endpoint{}, 128, 10000);
    auto owner_cfg = config("md-owner"); owner_cfg.metadata_endpoint = metadata.endpoint();
    owner_cfg.disks = {{file.path, 16384 + 16 * 4096, 4096, 16384, true}};
    Agent owner(owner_cfg);
    std::array<uint8_t, 256> source{}, destination{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i);
    auto src = owner.register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    auto handle = owner.batch_store({{"md-key", {{src, 0, source.size()}}, {}}});
    CHECK(wait(owner, handle) == std::vector<Status>{Status::success}); owner.release(handle);
    auto reader_cfg = config("md-reader"); reader_cfg.metadata_endpoint = metadata.endpoint();
    Agent reader(reader_cfg);
    auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(destination.data()), destination.size());
    // The foreground only queues discovery; background lookup, directory and hello make it ready.
    eventually([&]() -> bool { return reader.batch_exists({"md-key"})[0]; }, "metadata-discovered key");
    CHECK(load(reader, {"md-key", {{dst, 0, destination.size()}}, {}}) == Status::success);
    CHECK(destination == source);
    CHECK(!reader.batch_exists({"absent"})[0]);
    auto start = Clock::now();
    CHECK(load(reader, {"absent", {{dst, 0, destination.size()}}, {}}) == Status::not_ready);
    CHECK(Clock::now() - start < std::chrono::milliseconds(500));
    reader.deregister_memory(dst); owner.deregister_memory(src);
    reader.close(); owner.close(); metadata.close();
}
} // namespace
int main() {
    try {
        caller_hint_roundtrip_and_restart(); metadata_discovery_roundtrip();
        std::cout << "remote tests passed (2 real Agent TCP suites)\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
