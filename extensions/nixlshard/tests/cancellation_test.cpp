/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
using namespace nixlshard;
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<bool> enabled{false}, entered{false}, resume{false};
ino_t target_inode = 0;
dev_t target_device = 0;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)
void eventually(const std::function<bool()> &predicate) {
    auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        CHECK(Clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
std::vector<Status> wait(Agent &agent, std::uint64_t handle) {
    std::optional<std::vector<Status>> status;
    eventually([&] { status = agent.poll(handle); return status.has_value(); });
    return *status;
}
struct File {
    std::string path;
    File() {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-cancel-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};
}
extern "C" int fdatasync(int fd) {
    using Fn = int (*)(int);
    static auto actual = reinterpret_cast<Fn>(::dlsym(RTLD_NEXT, "fdatasync"));
    if (!actual) std::abort();
    if (enabled.load()) {
        struct stat st{};
        if (!::fstat(fd, &st) && st.st_ino == target_inode && st.st_dev == target_device) {
            entered.store(true);
            while (!resume.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return actual(fd);
}
int main() {
    void *source = MAP_FAILED;
    try {
        File file;
        AgentConfig cfg; cfg.name = "store-cancellation"; cfg.workers = 1;
        cfg.staging_slots = 1; cfg.staging_slot_bytes = 4096; cfg.timeout_ms = 150;
        cfg.disks = {{file.path, 16384 + 4096, 4096, 16384, true}};
        Agent agent(cfg);
        struct ResumeOnExit {
            ~ResumeOnExit() { resume.store(true); enabled.store(false); }
        } resume_on_exit;
        source = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(source != MAP_FAILED); std::memset(source, 31, 4096);
        auto token = agent.register_memory(reinterpret_cast<std::uintptr_t>(source), 4096);
        auto first = agent.batch_store({{"old", {{token, 0, 4096}}, {}}});
        CHECK(wait(agent, first) == std::vector<Status>{Status::success}); agent.release(first);
        eventually([&] { return agent.stats()["metadata_checkpoint_ns"] > 0; });
        CHECK(agent.checkpoint() == Status::success);
        struct stat st{}; CHECK(::stat(file.path.c_str(), &st) == 0);
        target_inode = st.st_ino; target_device = st.st_dev; enabled.store(true);
        // The only slot is full. Hold eager reclamation's durability barrier until
        // the submitted store has reported its logical timeout.
        auto second = agent.batch_store({{"new", {{token, 0, 4096}}, {}}});
        eventually([&] { return entered.load(); });
        CHECK(wait(agent, second) == std::vector<Status>{Status::timeout});
        CHECK(agent.poll(second) == std::optional<std::vector<Status>>{{Status::timeout}});
        // Fault injection detects any canceled source packing after the terminal
        // status. The allocation itself remains alive until release.
        CHECK(::mprotect(source, 4096, PROT_NONE) == 0);
        resume.store(true);
        agent.release(second);
        enabled.store(false);
        CHECK(!agent.batch_exists({"new"})[0]);
        CHECK(::mprotect(source, 4096, PROT_READ | PROT_WRITE) == 0);
        agent.deregister_memory(token); agent.close();
        ::munmap(source, 4096); source = MAP_FAILED;
        std::cout << "canceled store never reads source after timeout during reclamation\n";
        return 0;
    } catch (const std::exception &error) {
        resume.store(true); enabled.store(false);
        if (source != MAP_FAILED) ::munmap(source, 4096);
        std::cerr << error.what() << '\n'; return 1;
    }
}
