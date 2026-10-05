/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
using Clock = std::chrono::steady_clock;
struct File {
    std::string path;
    File() {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-direct-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};
struct Memory {
    uint8_t *data = nullptr; size_t size;
    explicit Memory(size_t bytes) : size(bytes) { CHECK(!::posix_memalign(reinterpret_cast<void **>(&data), 4096, bytes)); }
    ~Memory() { ::free(data); }
};
AgentConfig config(const std::string &name) {
    AgentConfig cfg; cfg.name = name; cfg.staging_slots = 2;
    cfg.staging_slot_bytes = 32768; cfg.workers = 1;
    cfg.max_inflight = 4; cfg.timeout_ms = 1500; cfg.remote_batch_limit = 8;
    return cfg;
}
void eventually(const std::function<bool()> &predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        CHECK(Clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
std::vector<Status> wait(Agent &agent, uint64_t handle) {
    std::optional<std::vector<Status>> result;
    eventually([&] { result = agent.poll(handle); return result.has_value(); });
    return *result;
}
void rejects(const std::function<void()> &operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception &) { rejected = true; }
    CHECK(rejected);
}
void scatter_roundtrip(bool trace) {
    File file; auto oc = config("direct-owner");
    oc.disks = {{file.path, 16384 + 128 * 4096, 4096, 16384, true}};
    Agent owner(oc); std::array<uint8_t, 8192> source{}, target{}, expected{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = (i * 7 + 13) % 211;
    auto src = owner.register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    auto put = owner.batch_store({{"one", {{src, 0, 64}}, {}}, {"two", {{src, 4096, 3000}}, {}}});
    CHECK(wait(owner, put) == std::vector<Status>({Status::success, Status::success})); owner.release(put);
    auto rc = config("direct-reader"); rc.direct_receive = true; rc.enable_trace = trace;
    rc.peers[oc.name] = owner.endpoint(); Agent reader(rc);
    auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(target.data()), target.size());
    rejects([&] { reader.register_memory(reinterpret_cast<uintptr_t>(target.data()) + 1, 10); });
    eventually([&]() -> bool { return reader.batch_exists({"one"}, {oc.name})[0]; });
    target.fill(201); expected.fill(201);
    auto get = reader.batch_load({
        {"one", {{dst, 13, 17}, {dst, 100, 47}}, oc.name},
        {"missing", {{dst, 200, 32}}, oc.name},
        {"two", {{dst, 512, 1500}, {dst, 4096, 1500}}, oc.name},
        {"one", {{dst, 6144, 64}}, oc.name}});
    auto result = wait(reader, get);
    if (result != std::vector<Status>({Status::success, Status::not_found, Status::success, Status::success})) {
        for (auto status : result) std::cerr << "scatter status=" << static_cast<int>(status) << '\n';
        for (auto &[name, value] : owner.stats()) std::cerr << "owner " << name << "=" << value << '\n';
        for (auto &[name, value] : reader.stats()) std::cerr << "reader " << name << "=" << value << '\n';
    }
    CHECK(result == std::vector<Status>({Status::success, Status::not_found, Status::success, Status::success}));
    eventually([&] { return reader.is_quiescent(get); });
    std::copy_n(source.begin(), 17, expected.begin() + 13);
    std::copy_n(source.begin() + 17, 47, expected.begin() + 100);
    std::copy_n(source.begin() + 4096, 1500, expected.begin() + 512);
    std::copy_n(source.begin() + 5596, 1500, expected.begin() + 4096);
    std::copy_n(source.begin(), 64, expected.begin() + 6144);
    CHECK(target == expected);
    CHECK(reader.stats()["staging_copy_bytes"] == 0);
    CHECK(reader.stats()["direct_receive_bytes"] == 3128);
    CHECK(reader.stats()["direct_receive_segments"] == 5);
    CHECK(owner.stats()["ucx_write_bytes"] == 3128);
    auto events = reader.trace(get); CHECK(events.size() == (trace ? 2 : 0));
    if (trace) {
        const auto &rpc = events.back();
        CHECK(rpc.stage == "remote_rpc" && rpc.direct_receive && rpc.destination_segments == 5);
        CHECK(rpc.bytes == 3128 && rpc.owner_read_bytes == 12288 && rpc.owner_timing_flags == 15);
        CHECK(rpc.owner_posix_ns > 0 && rpc.owner_ucx_ns > 0 && rpc.owner_metadata_ns > 0);
        CHECK(rpc.owner_metadata_bytes > 0);
    }
    // Completion does not release ownership of the caller's registered ranges.
    rejects([&] { reader.deregister_memory(dst); });
    rejects([&] { reader.batch_load({{"one", {{dst, 13, 64}}, oc.name}}); });
    reader.release(get);
    rejects([&] { reader.batch_load({{"one", {{dst, 0, 64}}, oc.name}, {"one", {{dst, 32, 64}}, oc.name}}); });
    reader.deregister_memory(dst);
    // A live owner's imported memory view must follow deregistration/re-registration
    // and added regions, rather than reusing the first request's stale metadata.
    dst = reader.register_memory(reinterpret_cast<uintptr_t>(target.data()), target.size());
    std::array<uint8_t, 64> added{}; auto extra = reader.register_memory(reinterpret_cast<uintptr_t>(added.data()), added.size());
    get = reader.batch_load({{"one", {{extra, 0, 64}}, oc.name}});
    CHECK(wait(reader, get) == std::vector<Status>{Status::success});
    eventually([&] { return reader.is_quiescent(get); }); reader.release(get);
    CHECK(std::equal(added.begin(), added.end(), source.begin()));
    reader.deregister_memory(extra); reader.deregister_memory(dst); owner.deregister_memory(src);
    reader.close(); owner.close();
}
void local_direct_and_padding() {
    File file; Memory source(16384), target(24576); auto cfg = config("direct-local");
    cfg.direct_receive = true; cfg.direct_io = true; cfg.enable_trace = true;
    cfg.disks = {{file.path, 16384 + 128 * 4096, 4096, 16384, true}};
    Agent agent(cfg);
    for (size_t i = 0; i < source.size; ++i) source.data[i] = (i * 3 + 19) % 251;
    std::fill_n(target.data, target.size, 201);
    auto src = agent.register_memory(reinterpret_cast<uintptr_t>(source.data), source.size);
    auto dst = agent.register_memory(reinterpret_cast<uintptr_t>(target.data), target.size);
    auto put = agent.batch_store({{"aligned", {{src, 0, 4096}, {src, 8192, 4096}}, {}},
                                  {"padded", {{src, 0, 37}}, {}}});
    CHECK(wait(agent, put) == std::vector<Status>({Status::success, Status::success})); agent.release(put);
    const auto before = agent.stats()["staging_copy_bytes"];
    auto get = agent.batch_load({{"aligned", {{dst, 4096, 4096}, {dst, 12288, 4096}}, {}}});
    CHECK(wait(agent, get) == std::vector<Status>{Status::success});
    eventually([&] { return agent.is_quiescent(get); });
    auto events = agent.trace(get); CHECK(events.size() == 2);
    CHECK(events.back().stage == "local_posix" && events.back().direct_receive && events.back().destination_segments == 2);
    agent.release(get);
    CHECK(agent.stats()["staging_copy_bytes"] == before);
    CHECK(agent.stats()["direct_local_read_bytes"] == 8192);
    CHECK(std::equal(target.data + 4096, target.data + 8192, source.data));
    CHECK(std::equal(target.data + 12288, target.data + 16384, source.data + 8192));
    CHECK(target.data[4095] == 201 && target.data[8192] == 201 && target.data[16384] == 201);
    get = agent.batch_load({{"padded", {{dst, 20000, 37}}, {}}});
    CHECK(wait(agent, get) == std::vector<Status>{Status::success});
    eventually([&] { return agent.is_quiescent(get); }); agent.release(get);
    CHECK(agent.stats()["local_direct_fallbacks"] == 1 && agent.stats()["staging_copy_bytes"] == before + 37);
    CHECK(std::equal(target.data + 20000, target.data + 20037, source.data));
    CHECK(target.data[19999] == 201 && target.data[20037] == 201);
    agent.deregister_memory(dst); agent.deregister_memory(src); agent.close();
}
void delayed_write_and_incarnation_fence() {
    File file; auto oc = config("late-owner");
    oc.disks = {{file.path, 16384 + 32 * 4096, 4096, 16384, true}};
    Agent owner(oc); auto alternate_cfg = config(oc.name); Agent alternate(alternate_cfg);
    std::array<uint8_t, 64> source{}; std::array<uint8_t, 128> target{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = i + 17;
    auto src = owner.register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    auto put = owner.batch_store({{"one", {{src, 0, 64}}, {}}});
    CHECK(wait(owner, put) == std::vector<Status>{Status::success}); owner.release(put);
    std::atomic<bool> first{true}, allow_write{false}, allow_cleanup{false}, late_done{false}, advertise_new{false}, reset_exists{false};
    wire::Server proxy({}, [&](std::string_view request) {
        wire::Reader r(request); const auto op = r.u8();
        if (op == wire::cleanup && !allow_cleanup.load()) {
            wire::Writer reply; reply.u8(static_cast<uint8_t>(Status::success)); reply.u8(0); return reply.data;
        }
        if (op == wire::exists && reset_exists.exchange(false)) return std::string{};
        const bool delayed = (op == wire::load_scatter || op == wire::load_scatter_trace) && first.exchange(false);
        if (delayed) while (!allow_write.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        wire::Connection connection(op == wire::hello && advertise_new.load() ? alternate.endpoint() : owner.endpoint(), 1500);
        auto reply = connection.call(request, 1500);
        if (delayed) late_done.store(true);
        return reply;
    }, 5000);
    // Unblock the proxy even if a check fails.
    struct Unblock { std::atomic<bool> &flag; ~Unblock() { flag.store(true); } } unblock{allow_write};
    auto rc = config("late-reader"); rc.timeout_ms = 200; rc.direct_receive = true; rc.peers[oc.name] = proxy.endpoint();
    Agent reader(rc); auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(target.data()), target.size());
    eventually([&]() -> bool { return reader.batch_exists({"one"}, {oc.name})[0]; });
    auto get = reader.batch_load({{"one", {{dst, 0, 64}}, oc.name}});
    CHECK(wait(reader, get) == std::vector<Status>{Status::timeout});
    CHECK(!reader.is_quiescent(get));
    auto start = Clock::now(); rejects([&] { reader.release(get); });
    CHECK(Clock::now() - start < std::chrono::milliseconds(100));
    rejects([&] { reader.deregister_memory(dst); });
    rejects([&] { reader.batch_load({{"one", {{dst, 0, 64}}, oc.name}}); });
    eventually([&]() -> bool { return reader.batch_exists({"one"}, {oc.name})[0]; });
    auto healthy = reader.batch_load({{"one", {{dst, 64, 64}}, oc.name}});
    CHECK(wait(reader, healthy) == std::vector<Status>{Status::success});
    eventually([&] { return reader.is_quiescent(healthy); }); reader.release(healthy);
    CHECK(std::equal(target.begin() + 64, target.end(), source.begin()));
    // The timed-out operation really writes its original destination later.
    allow_write.store(true); eventually([&] { return late_done.load(); });
    CHECK(std::equal(target.begin(), target.begin() + 64, source.begin()));
    CHECK(!reader.is_quiescent(get));
    // A genuine different owner incarnation cannot authorize recycling old memory,
    // even though the original owner's operation has finished.
    auto connections = reader.stats()["peer_connections"];
    advertise_new.store(true); reset_exists.store(true); reader.batch_exists({"one"}, {oc.name});
    eventually([&] { return reader.stats()["peer_connections"] > connections; });
    allow_cleanup.store(true); std::this_thread::sleep_for(std::chrono::milliseconds(350));
    CHECK(!reader.is_quiescent(get)); rejects([&] { reader.release(get); });
    connections = reader.stats()["peer_connections"];
    advertise_new.store(false); reset_exists.store(true); reader.batch_exists({"one"}, {oc.name});
    eventually([&] { return reader.stats()["peer_connections"] > connections; });
    eventually([&] { return reader.is_quiescent(get); });
    reader.release(get); reader.deregister_memory(dst); reader.close(); proxy.close();
    owner.deregister_memory(src); owner.close(); alternate.close();
}
} // namespace
int main() {
    try {
        scatter_roundtrip(false); scatter_roundtrip(true);
        local_direct_and_padding(); delayed_write_and_incarnation_fence();
        std::cout << "direct receive tests passed (real scatter, metadata refresh, aligned POSIX, padding, late UCX and incarnation fence)\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
