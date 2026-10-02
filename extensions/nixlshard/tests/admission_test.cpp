/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
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
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-admission-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};

AgentConfig config(const std::string &name) {
    AgentConfig cfg;
    cfg.name = name; cfg.staging_slots = 2; cfg.staging_slot_bytes = 4096;
    cfg.workers = 1; cfg.max_inflight = 4; cfg.timeout_ms = 1500;
    return cfg;
}

void eventually(const std::function<bool()> &predicate, const std::string &description) {
    auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        if (Clock::now() >= deadline) throw std::runtime_error("timed out waiting for " + description);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::vector<Status> wait(Agent &agent, std::uint64_t handle) {
    std::optional<std::vector<Status>> result;
    eventually([&] { result = agent.poll(handle); return result.has_value(); }, "transfer");
    agent.release(handle);
    return *result;
}

void normal_and_cleanup_connections_leave_room_for_ninth_reader() {
    File file;
    auto owner_cfg = config("admission-owner");
    owner_cfg.disks = {{file.path, 16384 + 16 * 4096, 4096, 16384, true}};
    Agent owner(owner_cfg);
    std::array<std::uint8_t, 4096> source{}, destination{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 17 + 9) % 251;
    auto src = owner.register_memory(reinterpret_cast<std::uintptr_t>(source.data()), source.size());
    CHECK(wait(owner, owner.batch_store({{"admission-key", {{src, 0, source.size()}}, {}}})) ==
          std::vector<Status>{Status::success});

    // Keep each reader alive: normal and cleanup channels are intentionally persistent.
    std::vector<std::unique_ptr<Agent>> readers;
    std::vector<std::uint64_t> registrations;
    for (unsigned i = 0; i < 9; ++i) {
        auto cfg = config("admission-reader-" + std::to_string(i));
        cfg.peers[owner_cfg.name] = owner.endpoint();
        auto reader = std::make_unique<Agent>(cfg);
        eventually([&] { return reader->stats()["peer_connections"] >= 1; },
                   "reader " + std::to_string(i) + " normal connection");
        auto dst = reader->register_memory(reinterpret_cast<std::uintptr_t>(destination.data()), destination.size());
        auto cleanups_before = owner.stats()["remote_cleanup_requests"];
        destination.fill(0);
        CHECK(wait(*reader, reader->batch_load({
            {"admission-key", {{dst, 0, destination.size()}}, owner_cfg.name}})) ==
              std::vector<Status>{Status::success});
        CHECK(destination == source);
        eventually([&] { return owner.stats()["remote_cleanup_requests"] > cleanups_before; },
                   "reader " + std::to_string(i) + " persistent cleanup connection");
        // The normal channel remains usable while the cleanup channel stays connected.
        eventually([&]() -> bool { return reader->batch_exists({"admission-key"}, {owner_cfg.name})[0]; },
                   "reader " + std::to_string(i) + " owner existence");
        registrations.push_back(dst);
        readers.push_back(std::move(reader));
    }
    CHECK(readers.size() == 9);
    for (std::size_t i = 0; i < readers.size(); ++i) {
        readers[i]->deregister_memory(registrations[i]);
        readers[i]->close();
    }
    owner.deregister_memory(src); owner.close();
}

void exists_uses_one_deadline_and_measures_failed_calls() {
    Agent owner(config("admission-hello-owner"));
    std::atomic<unsigned> stalled_requests{0};
    std::vector<std::unique_ptr<wire::Server>> fake_owners;
    auto reader_cfg = config("admission-deadline-reader");
    reader_cfg.timeout_ms = 500;
    std::vector<std::string> keys, hints;
    for (unsigned i = 0; i < 5; ++i) {
        auto fake = std::make_unique<wire::Server>(Endpoint{},
            [&owner, &stalled_requests](std::string_view request) {
                wire::Reader input(request);
                auto operation = input.u8();
                if (operation == wire::hello) {
                    // Forward a genuine hello, so the reader imports valid native UCX metadata.
                    wire::Connection connection(owner.endpoint(), 500);
                    return connection.call(request, 500);
                }
                if (operation != wire::exists) throw std::invalid_argument("unexpected fake-owner operation");
                input.str(); auto count = input.u32();
                CHECK(count > 0 && count <= 128);
                for (unsigned j = 0; j < count; ++j) input.str();
                input.finish();
                ++stalled_requests;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                wire::Writer response; response.u8(static_cast<unsigned>(Status::success));
                response.u32(count);
                for (unsigned j = 0; j < count; ++j) response.u8(false);
                return response.data;
            }, 1500);
        auto name = "admission-fake-owner-" + std::to_string(i);
        reader_cfg.peers[name] = fake->endpoint();
        keys.push_back("absent-" + std::to_string(i)); hints.push_back(name);
        fake_owners.push_back(std::move(fake));
    }
    Agent reader(reader_cfg);
    eventually([&] { return reader.stats()["peer_connections"] >= fake_owners.size(); },
               "five established fake-owner connections");
    auto stats_before = reader.stats();
    auto start = Clock::now();
    auto results = reader.batch_exists(keys, hints);
    auto duration = Clock::now() - start;
    auto stats_after = reader.stats();
    CHECK(results.size() == keys.size());
    CHECK(std::none_of(results.begin(), results.end(), [](bool found) { return found; }));
    CHECK(stalled_requests.load() >= 1);
    CHECK(duration >= std::chrono::milliseconds(300));
    CHECK(duration < std::chrono::milliseconds(750));
    CHECK(stats_after["exists_control_ns"] - stats_before["exists_control_ns"] >= 300000000);
    reader.close();
    for (auto &fake : fake_owners) fake->close();
    owner.close();
}
} // namespace

int main() {
    try {
        normal_and_cleanup_connections_leave_room_for_ninth_reader();
        exists_uses_one_deadline_and_measures_failed_calls();
        std::cout << "admission tests passed (nine readers and bounded failed-owner exists)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
