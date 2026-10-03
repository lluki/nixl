/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <thread>
#include <unistd.h>
using namespace nixlshard;
namespace {
using Clock = std::chrono::steady_clock;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)
void eventually(const std::function<bool()> &check) {
    auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!check()) {
        CHECK(Clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
struct File {
    std::string path;
    File() {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-timeout-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};
Status wait(Agent &agent, std::uint64_t handle, std::size_t count = 1) {
    std::optional<std::vector<Status>> result;
    eventually([&] { result = agent.poll(handle); return result.has_value(); });
    CHECK(result->size() == count);
    CHECK(std::all_of(result->begin(), result->end(), [&](Status s) { return s == result->at(0); }));
    return result->at(0);
}
void delayed_load_fenced_and_quarantine_retained(bool grouped = false, bool tracing = false) {
    File file;
    AgentConfig cfg; cfg.name = "timeout-owner"; cfg.staging_slots = 2;
    cfg.staging_slot_bytes = grouped ? 8192 : 4096; cfg.timeout_ms = 2000;
    cfg.disks = {{file.path, 1024 * 1024, 4096, 16384, true}};
    Agent owner(cfg);
    std::array<unsigned char, 128> source{}, destination{};
    source.fill(77); destination.fill(19);
    auto src = owner.register_memory(reinterpret_cast<std::uintptr_t>(source.data()), source.size());
    const auto value_bytes = grouped ? 64 : source.size();
    auto put = owner.batch_store({{"value", {{src, 0, value_bytes}}, {}}});
    CHECK(wait(owner, put) == Status::success); owner.release(put);
    const auto endpoint = owner.endpoint();
    std::atomic<bool> allow_cleanup{false}, late_canceled{false}, delay_load{true}, fence_created{false};
    wire::Server proxy(Endpoint{}, [&](std::string_view message) {
        wire::Reader inspect(message); auto op = inspect.u8();
        if (op == wire::cleanup && !allow_cleanup.load()) {
            wire::Writer reply; reply.u8(static_cast<unsigned>(Status::success)); reply.u8(false);
            return reply.data;
        }
        const bool delayed = (op == wire::load || op == wire::load_batch || op == wire::load_batch_trace) && delay_load.load();
        if (delayed) eventually([&] { return fence_created.load(); });
        wire::Connection upstream(endpoint, 1000);
        auto result = upstream.call(message, 1000);
        if (op == wire::cleanup) {
            wire::Reader reply(result);
            if (static_cast<Status>(reply.u8()) == Status::success && reply.u8())
                fence_created.store(true);
        }
        if (delayed) {
            wire::Reader reply(result);
            late_canceled.store(static_cast<Status>(reply.u8()) == Status::canceled);
        }
        return result;
    }, 2000, 16);
    AgentConfig reader_cfg; reader_cfg.name = "timeout-reader"; reader_cfg.staging_slots = 1;
    reader_cfg.staging_slot_bytes = grouped ? 8192 : 4096; reader_cfg.timeout_ms = 500;
    reader_cfg.remote_batch_limit = grouped ? 2 : 1;
    reader_cfg.enable_trace = tracing;
    reader_cfg.peers["timeout-owner"] = proxy.endpoint();
    Agent reader(reader_cfg);
    auto dst = reader.register_memory(reinterpret_cast<std::uintptr_t>(destination.data()), destination.size());
    CHECK(owner.batch_exists({"value"})[0]);
    try { eventually([&]() -> bool { return reader.batch_exists({"value"}, {"timeout-owner"})[0]; }); }
    catch (...) {
        for (auto &[name, value] : reader.stats()) std::cerr << "preflight " << name << '=' << value << '\n';
        throw;
    }
    Object get{"value", {{dst, 0, value_bytes}}, "timeout-owner"};
    std::vector<Object> gets{get};
    if (grouped) gets.push_back({"value", {{dst, 64, 64}}, "timeout-owner"});
    std::uint64_t handle = 0;
    eventually([&] {
        handle = reader.batch_load(gets);
        auto result = wait(reader, handle, gets.size());
        if (result == Status::not_ready) { reader.release(handle); return false; }
        CHECK(result == Status::timeout);
        return true;
    });
    CHECK(reader.poll(handle)->at(0) == Status::timeout);
    reader.release(handle);
    CHECK(destination[0] == 19 && destination.back() == 19);
    CHECK(reader.stats()["staging_quarantined_slots"] == 1);
    bool retained = false;
    try { reader.poll(handle); } catch (const std::invalid_argument &) { retained = true; }
    CHECK(retained);
    auto blocked = reader.batch_load(gets);
    CHECK(wait(reader, blocked, gets.size()) == Status::busy); reader.release(blocked);
    CHECK(destination[0] == 19 && destination.back() == 19);
    allow_cleanup.store(true);
    eventually([&] { return reader.stats()["staging_quarantined_slots"] == 0; });
    eventually([&] { return late_canceled.load(); });
    delay_load.store(false);
    eventually([&]() -> bool { return reader.batch_exists({"value"}, {"timeout-owner"})[0]; });
    auto good = reader.batch_load(gets);
    CHECK(wait(reader, good, gets.size()) == Status::success); reader.release(good);
    CHECK(destination == source);
    reader.deregister_memory(dst); owner.deregister_memory(src);
    reader.close(); proxy.close(); owner.close();
}
}
int main() {
    try { delayed_load_fenced_and_quarantine_retained();
          delayed_load_fenced_and_quarantine_retained(true);
          delayed_load_fenced_and_quarantine_retained(true, true);
          std::cout << "timeout, quarantine, delayed-load fencing and reuse passed\n"; return 0; }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
