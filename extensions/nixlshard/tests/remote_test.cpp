/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <libaio.h>
#include <sys/syscall.h>
#include <chrono>
#include <functional>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {
std::atomic<bool> fail_group_read{false};
std::atomic<unsigned> injected_read_errors{0};
}
extern "C" int io_submit(io_context_t ctx, long nr, struct iocb **ios) {
    if (fail_group_read.load() && nr > 0 && ios[0]->aio_lio_opcode == IO_CMD_PREAD) {
        ++injected_read_errors; return -EIO;
    }
    // POSIX plugins load libaio locally; RTLD_NEXT from this executable cannot
    // reliably resolve it. Match libaio's negative-errno syscall convention.
    const auto result = ::syscall(SYS_io_submit, ctx, nr, ios);
    return result < 0 ? -errno : static_cast<int>(result);
}

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
void grouped_loads_preserve_layout_and_statuses(bool tracing = false) {
    File file;
    auto owner_cfg = config("group-owner"); owner_cfg.staging_slot_bytes = 32768;
    owner_cfg.disks = {{file.path, 16384 + 128 * 4096, 4096, 16384, true}};
    Agent owner(owner_cfg);
    std::array<uint8_t, 8192> source{}, destination{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((i * 7 + 13) % 211);
    auto src = owner.register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    auto put = owner.batch_store({{"one", {{src, 0, 64}}, {}}, {"two", {{src, 4096, 3000}}, {}}});
    CHECK(wait(owner, put) == std::vector<Status>({Status::success, Status::success})); owner.release(put);
    auto reader_cfg = config("group-reader"); reader_cfg.staging_slot_bytes = 32768;
    reader_cfg.remote_batch_limit = 8; reader_cfg.peers[owner_cfg.name] = owner.endpoint();
    reader_cfg.enable_trace = tracing;
    Agent reader(reader_cfg);
    auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(destination.data()), destination.size());
    eventually([&] { return reader.batch_exists({"one", "two"}, {owner_cfg.name, owner_cfg.name}) == std::vector<bool>({true, true}); });
    destination.fill(201);
    const auto before_get = Clock::now();
    auto get = reader.batch_load({
        {"one", {{dst, 13, 17}, {dst, 100, 47}}, owner_cfg.name},
        {"missing", {{dst, 200, 32}}, owner_cfg.name},
        {"two", {{dst, 512, 3000}}, owner_cfg.name},
        {"one", {{dst, 4000, 64}}, owner_cfg.name},
        {"one", {{dst, 4200, 63}}, owner_cfg.name}});
    CHECK(wait(reader, get) == std::vector<Status>({Status::success, Status::not_found,
          Status::success, Status::success, Status::invalid_input}));
    auto events = reader.trace(get);
    CHECK(events.size() == (tracing ? 5 : 0));
    if (tracing) {
        auto ns = [](Clock::time_point t) { return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()); };
        CHECK(events.front().stage == "queue" && events.front().start_ns >= ns(before_get));
        const auto &rpc = events.at(1);
        CHECK(rpc.stage == "remote_rpc" && !rpc.request_id.empty());
        CHECK(rpc.owner_timing_flags == 3 && rpc.bytes == 3128 && rpc.owner_read_bytes == 12288);
        CHECK(rpc.owner_posix_ns > 0 && rpc.owner_ucx_ns > 0);
        CHECK(rpc.owner_posix_ns + rpc.owner_ucx_ns <= rpc.end_ns - rpc.start_ns);
        CHECK(events.at(2).first_object == 0 && events.at(3).first_object == 2 && events.at(4).first_object == 3);
        uint64_t copied = 0;
        for (size_t i = 0; i < events.size(); ++i) {
            CHECK(events[i].end_ns >= events[i].start_ns);
            CHECK(events[i].end_ns <= ns(Clock::now()));
            if (i) CHECK(events[i].start_ns >= events[i - 1].end_ns);
            if (events[i].stage == "staging_copy") copied += events[i].bytes;
        }
        CHECK(copied == 3128 && events.size() <= 1 + 3 * 5);
    }
    reader.release(get);
    bool expired_trace = false;
    try { reader.trace(get); } catch (const std::invalid_argument &) { expired_trace = true; }
    CHECK(expired_trace);
    auto expected = std::array<uint8_t, 8192>{}; expected.fill(201);
    std::copy_n(source.begin(), 17, expected.begin() + 13);
    std::copy_n(source.begin() + 17, 47, expected.begin() + 100);
    std::copy_n(source.begin() + 4096, 3000, expected.begin() + 512);
    std::copy_n(source.begin(), 64, expected.begin() + 4000);
    CHECK(destination == expected);
    // A bulk read error cannot fabricate successes or issue UCX writes.
    auto write_before = owner.stats()["ucx_write_bytes"];
    destination.fill(201); fail_group_read.store(true);
    get = reader.batch_load({{"one", {{dst, 0, 64}}, owner_cfg.name},
                             {"two", {{dst, 512, 3000}}, owner_cfg.name}});
    auto failed = wait(reader, get); fail_group_read.store(false);
    if (tracing) {
        events = reader.trace(get);
        CHECK(events.size() == 2 && events.back().stage == "remote_rpc");
        CHECK(events.back().bytes == 0 && events.back().owner_timing_flags == 0);
    }
    reader.release(get);
    CHECK(failed == std::vector<Status>({Status::io_error, Status::io_error}));
    CHECK(injected_read_errors.load() > 0);
    CHECK(std::all_of(destination.begin(), destination.end(), [](uint8_t x) { return x == 201; }));
    CHECK(owner.stats()["ucx_write_bytes"] == write_before);
    CHECK(reader.stats()["remote_load_batch_requests"] == 2);
    CHECK(reader.stats()["remote_read_bytes"] == 3128);
    CHECK(owner.stats()["posix_read_bytes"] == 3 * 4096);
    CHECK(owner.stats()["ucx_write_bytes"] == 3128);
    // Metadata failure must not erase independently known per-page outcomes.
    {
        wire::Connection control(owner.endpoint(), 1000);
        wire::Writer request; request.u8(wire::load_batch); request.str("bad-group-metadata");
        request.str(peer_identity(owner.endpoint())); request.str("bogus"); request.str("bogus");
        request.u64(1234); request.u32(3); request.u32(1000);
        request.str("one"); request.u64(64); request.str("missing"); request.u64(32);
        request.str("one"); request.u64(63);
        auto response = control.call(request.data, 1000); wire::Reader result(response);
        CHECK(static_cast<Status>(result.u8()) == Status::success && result.u32() == 3);
        CHECK(static_cast<Status>(result.u8()) == Status::not_ready);
        CHECK(static_cast<Status>(result.u8()) == Status::not_found);
        CHECK(static_cast<Status>(result.u8()) == Status::invalid_input); result.finish();
        CHECK(owner.stats()["ucx_write_bytes"] == 3128);
        // Checked target arithmetic rejects overflow before metadata import/I/O.
        wire::Writer overflow; overflow.u8(wire::load_batch); overflow.str("overflow-group");
        overflow.str(peer_identity(owner.endpoint())); overflow.str("bogus"); overflow.str("bogus");
        overflow.u64(UINT64_MAX - 31); overflow.u32(2); overflow.u32(1000);
        overflow.str("one"); overflow.u64(64); overflow.str("one"); overflow.u64(64);
        response = control.call(overflow.data, 1000); wire::Reader rejected(response);
        CHECK(static_cast<Status>(rejected.u8()) == Status::invalid_input); rejected.finish();
        CHECK(owner.stats()["posix_read_bytes"] == 3 * 4096);
    }
    // Caller aliases retain the original per-object publication order.
    destination.fill(201); expected.fill(201);
    get = reader.batch_load({{"one", {{dst, 0, 64}}, owner_cfg.name},
                             {"two", {{dst, 32, 3000}}, owner_cfg.name}});
    CHECK(wait(reader, get) == std::vector<Status>({Status::success, Status::success})); reader.release(get);
    std::copy_n(source.begin(), 64, expected.begin());
    std::copy_n(source.begin() + 4096, 3000, expected.begin() + 32);
    CHECK(destination == expected);
    // The normal one-object tail retains the old wire path; tracing uses its
    // measured group envelope so owner durations are available for tails too.
    CHECK(load(reader, {"one", {{dst, 5000, 64}}, owner_cfg.name}) == Status::success);
    CHECK(reader.stats()["remote_load_batch_requests"] == (tracing ? 4 : 3));
    reader.deregister_memory(dst); owner.deregister_memory(src);
    reader.close(); owner.close();
}
void grouped_padding_falls_back_to_individual_loads() {
    File file;
    auto cfg = config("padding-owner"); cfg.staging_slot_bytes = 4096;
    cfg.disks = {{file.path, 16384 + 32 * 4096, 4096, 16384, true}};
    Agent owner(cfg);
    std::array<uint8_t, 192> source{}, destination{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i);
    auto src = owner.register_memory(reinterpret_cast<uintptr_t>(source.data()), source.size());
    auto put = owner.batch_store({{"one", {{src, 0, 64}}, {}}, {"two", {{src, 64, 64}}, {}},
                                 {"three", {{src, 128, 64}}, {}}});
    CHECK(wait(owner, put) == std::vector<Status>({Status::success, Status::success, Status::success})); owner.release(put);
    auto reader_cfg = config("padding-reader"); reader_cfg.staging_slot_bytes = 4096;
    reader_cfg.remote_batch_limit = 8; reader_cfg.peers[cfg.name] = owner.endpoint();
    Agent reader(reader_cfg);
    auto dst = reader.register_memory(reinterpret_cast<uintptr_t>(destination.data()), destination.size());
    eventually([&] { return reader.batch_exists({"one", "two"}, {cfg.name, cfg.name}) == std::vector<bool>({true, true}); });
    auto get = reader.batch_load({{"one", {{dst, 0, 64}}, cfg.name}, {"two", {{dst, 64, 64}}, cfg.name},
                                 {"three", {{dst, 128, 64}}, cfg.name}});
    CHECK(wait(reader, get) == std::vector<Status>({Status::success, Status::success, Status::success})); reader.release(get);
    CHECK(destination == source);
    CHECK(reader.stats()["remote_group_fallbacks"] == 1);
    CHECK(owner.stats()["posix_read_bytes"] == 12288 && owner.stats()["ucx_write_bytes"] == 192);
    reader.deregister_memory(dst); owner.deregister_memory(src); reader.close(); owner.close();
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
        grouped_loads_preserve_layout_and_statuses();
        grouped_loads_preserve_layout_and_statuses(true);
        grouped_padding_falls_back_to_individual_loads();
        std::cout << "remote tests passed (5 real Agent TCP suites, including traced and untraced grouped loads)\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
