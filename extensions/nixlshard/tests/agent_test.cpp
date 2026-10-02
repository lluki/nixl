/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error("check failed: " #expr); } while (false)
namespace {
template <typename Error>
void rejects(const std::function<void()> &action) {
    bool caught = false;
    try { action(); } catch (const Error &) { caught = true; }
    CHECK(caught);
}
std::vector<Status> wait(Agent &agent, std::uint64_t handle) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        auto result = agent.poll(handle);
        if (result) return *result;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("operation did not complete before test deadline");
}
struct File {
    std::string path;
    File() {
        auto name = (std::filesystem::temp_directory_path() / "nixlshard-agent-XXXXXX").string();
        int fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        ::close(fd);
        path = name;
    }
    ~File() { ::unlink(path.c_str()); }
};
} // namespace
int main() {
    try {
        File file;
        AgentConfig cfg;
        cfg.name = "native-local-test";
        cfg.disks = {{file.path, 16384 + 16 * 4096, 4096, 16384, true}};
        cfg.max_inflight = 1;
        cfg.workers = 1;
        cfg.staging_slots = 1;
        cfg.staging_slot_bytes = 4096;
        cfg.timeout_ms = 2000;
        std::array<std::uint8_t, 64> source{}, destination{};
        for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint8_t>(i + 11);
        std::vector<std::uint8_t> expected;
        for (auto [offset, length] : {std::pair{32, 16}, std::pair{4, 8}, std::pair{56, 8}})
            expected.insert(expected.end(), source.begin() + offset, source.begin() + offset + length);
        {
            Agent agent(cfg);
            CHECK(agent.endpoint().port != 0);
            rejects<std::invalid_argument>([&] { agent.register_memory(0, 64); });
            auto src = agent.register_memory(reinterpret_cast<std::uintptr_t>(source.data()), source.size());
            auto dst = agent.register_memory(reinterpret_cast<std::uintptr_t>(destination.data()), destination.size());
            rejects<std::invalid_argument>([&] { agent.batch_store({{"bad", {{src, 60, 8}}, {}}}); });
            rejects<std::invalid_argument>([&] { agent.batch_store({{"", {{src, 0, 8}}, {}}}); });
            rejects<std::invalid_argument>([&] { agent.batch_load({{"bad", {{src + dst + 1000, 0, 8}}, {}}}); });
            Object store{"key", {{src, 32, 16}, {src, 4, 8}, {src, 56, 8}}, {}};
            auto handle = agent.batch_store({store});
            CHECK(wait(agent, handle) == std::vector<Status>{Status::success});
            // Terminal handles retain registrations and admission budget until released.
            rejects<std::runtime_error>([&] { agent.deregister_memory(src); });
            rejects<std::runtime_error>([&] { agent.batch_store({store}); });
            agent.release(handle);
            CHECK(agent.batch_exists({"key", "absent"}) == std::vector<bool>({true, false}));
            Object load{"key", {{dst, 0, 12}, {dst, 24, 20}}, {}};
            auto get = agent.batch_load({load});
            CHECK(wait(agent, get) == std::vector<Status>{Status::success});
            agent.release(get);
            std::vector<std::uint8_t> actual(destination.begin(), destination.begin() + 12);
            actual.insert(actual.end(), destination.begin() + 24, destination.begin() + 44);
            CHECK(actual == expected);
            auto miss = agent.batch_load({{"absent", {{dst, 0, 32}}, {}}});
            CHECK(wait(agent, miss) == std::vector<Status>{Status::not_found});
            agent.release(miss);
            CHECK(agent.checkpoint() == Status::success);
            agent.deregister_memory(src);
            agent.deregister_memory(dst);
            agent.close();
            agent.close();
        }
        // A checkpointed key remains readable through a freshly instantiated agent.
        cfg.disks[0].create = false;
        destination.fill(0);
        {
            Agent restarted(cfg);
            auto dst = restarted.register_memory(reinterpret_cast<std::uintptr_t>(destination.data()), destination.size());
            auto handle = restarted.batch_load({{"key", {{dst, 0, expected.size()}}, {}}});
            CHECK(wait(restarted, handle) == std::vector<Status>{Status::success});
            restarted.release(handle);
            CHECK(std::equal(expected.begin(), expected.end(), destination.begin()));
            restarted.deregister_memory(dst);
            restarted.close();
        }
        // Retain closed agents to exercise Abseil mutex state across NIXL DSOs.
        // Static Abseil copies previously crashed after the eighth retained agent.
        std::vector<std::unique_ptr<File>> files;
        std::vector<std::unique_ptr<Agent>> retained;
        for (unsigned iteration = 0; iteration < 20; ++iteration) {
            files.push_back(std::make_unique<File>());
            cfg.name = "retained-" + std::to_string(iteration);
            cfg.disks[0].path = files.back()->path;
            cfg.disks[0].create = true;
            auto agent = std::make_unique<Agent>(cfg);
            auto token = agent->register_memory(reinterpret_cast<std::uintptr_t>(source.data()), source.size());
            std::vector<Object> batch;
            for (unsigned item = 0; item < 4; ++item)
                batch.push_back({std::to_string(item), {{token, item * 16, 16}}, {}});
            auto handle = agent->batch_store(batch);
            CHECK(wait(*agent, handle) == std::vector<Status>(4, Status::success));
            agent->release(handle);
            agent->deregister_memory(token);
            agent->close();
            retained.push_back(std::move(agent));
        }
        std::cout << "local scatter/gather, validation, handle lifetime, admission, restart, and repeated retained-agent lifetime passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
