/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
using namespace nixlshard;
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<int> fault{0};
std::atomic<bool> entered{false}, resume{false};
std::atomic<pthread_t> maintenance_thread{};
std::atomic<unsigned> injected_errors{0};
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
    std::optional<std::vector<Status>> result;
    eventually([&] { result = agent.poll(handle); return result.has_value(); });
    return *result;
}
struct File {
    std::string path;
    File() {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-reclaim-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
    }
    ~File() { ::unlink(path.c_str()); }
};
struct Fixture {
    File first, second;
    std::unique_ptr<Agent> agent;
    void *source = MAP_FAILED;
    std::uint64_t token = 0;
    std::vector<std::uint64_t> handles;
    explicit Fixture(bool two_disks = false, std::uint32_t timeout_ms = 1000,
                     std::size_t staging_bytes = 4096) {
        AgentConfig cfg; cfg.name = "reclamation-" + std::to_string(Clock::now().time_since_epoch().count());
        cfg.workers = 1; cfg.staging_slots = 1; cfg.staging_slot_bytes = staging_bytes;
        cfg.timeout_ms = timeout_ms;
        cfg.disks = {{first.path, 16384 + 4096, 4096, 16384, true}};
        if (two_disks) cfg.disks.push_back({second.path, 16384 + 4096, 4096, 16384, true});
        agent = std::make_unique<Agent>(cfg);
        source = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(source != MAP_FAILED); std::memset(source, 31, 4096);
        token = agent->register_memory(reinterpret_cast<std::uintptr_t>(source), 4096);
        struct stat st{}; CHECK(::stat(first.path.c_str(), &st) == 0);
        target_inode = st.st_ino; target_device = st.st_dev;
    }
    ~Fixture() {
        fault.store(0); resume.store(true);
        if (source != MAP_FAILED) ::mprotect(source, 4096, PROT_READ | PROT_WRITE);
        if (agent) {
            for (auto handle : handles) agent->release(handle);
            agent->deregister_memory(token); agent->close();
        }
        if (source != MAP_FAILED) ::munmap(source, 4096);
    }
    Object object(const std::string &key) { return {key, {{token, 0, 4096}}, {}}; }
    std::uint64_t submit_object(const Object &value) {
        auto handle = agent->batch_store({value}); handles.push_back(handle); return handle;
    }
    std::uint64_t submit(const std::string &key) { return submit_object(object(key)); }
    void release(std::uint64_t handle) {
        agent->release(handle);
        handles.erase(std::find(handles.begin(), handles.end(), handle));
    }
    void seed(const std::string &key) {
        auto handle = submit(key);
        CHECK(wait(*agent, handle) == std::vector<Status>{Status::success}); release(handle);
        eventually([&] { return agent->checkpoint() == Status::success; });
    }
    void identify_maintenance() {
        // Only maintenance checkpoints this idle fixture: identify it so EIO
        // is injected solely into the store worker's eviction commit.
        const auto before = agent->stats()["metadata_checkpoint_ns"];
        maintenance_thread.store(pthread_t{}); injected_errors.store(0); fault.store(3);
        eventually([] { return maintenance_thread.load() != pthread_t{}; });
        // Its completed timer is recorded after releasing the commit lock. The
        // error test needs a worker eviction, rather than healthy-disk fallback
        // during this identification checkpoint's remaining sync barriers.
        eventually([&] { return agent->stats()["metadata_checkpoint_ns"] > before; });
        fault.store(0);
    }
};
struct BlockedCheckpoint {
    std::thread thread;
    Status status = Status::busy;
    explicit BlockedCheckpoint(Agent &agent) {
        entered.store(false); resume.store(false); fault.store(1);
        thread = std::thread([this, &agent] {
            do { status = agent.checkpoint(); } while (status == Status::busy);
        });
        try { eventually([] { return entered.load(); }); }
        catch (...) { unblock(); throw; }
    }
    void unblock() { resume.store(true); if (thread.joinable()) thread.join(); fault.store(0); }
    ~BlockedCheckpoint() { unblock(); }
};
std::string disk_zero_key(const std::string &prefix) {
    for (unsigned i = 0;; ++i) {
        auto key = prefix + std::to_string(i);
        if (std::hash<std::string>{}(key) % 2 == 0) return key;
    }
}
void oversized_empty_disk_declines_without_packing() {
    Fixture f(false, 1000, 8192);
    CHECK(::mprotect(f.source, 4096, PROT_NONE) == 0);
    // Logical size fits staging and each segment fits the registration, but
    // two units cannot fit this empty disk's one-unit payload allocation.
    auto handle = f.submit_object({"oversized", {{f.token, 0, 4096}, {f.token, 0, 4096}}, {}});
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::no_space});
    f.release(handle);
    CHECK(f.agent->stats()["staging_copy_bytes"] == 0);
    CHECK(!f.agent->batch_exists({"oversized"})[0]);
    CHECK(::mprotect(f.source, 4096, PROT_READ | PROT_WRITE) == 0);
}
void contention_retries() {
    Fixture f;
    f.seed("old");
    BlockedCheckpoint checkpoint(*f.agent);
    auto handle = f.submit("replacement");
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CHECK(!f.agent->poll(handle)); // Transient lock contention is not no_space.
    checkpoint.unblock(); CHECK(checkpoint.status == Status::success);
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::success});
    f.release(handle);
    CHECK(f.agent->batch_exists({"replacement", "old"}) == (std::vector<bool>{true, false}));
}
void healthy_disk_wins_before_contended_disk_unblocks() {
    Fixture f(true);
    f.seed(disk_zero_key("old"));
    BlockedCheckpoint checkpoint(*f.agent);
    auto key = disk_zero_key("replacement");
    auto handle = f.submit(key);
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::success});
    f.release(handle);
    CHECK(!resume.load()); // Acceptance used disk two while disk one stays busy.
    CHECK(f.agent->batch_exists({key, disk_zero_key("old")}) == (std::vector<bool>{true, true}));
    checkpoint.unblock();
}
void contention_keeps_deadline_and_cancel_gate() {
    Fixture f(false, 150);
    f.seed("old");
    BlockedCheckpoint checkpoint(*f.agent);
    auto handle = f.submit("replacement");
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::timeout});
    // Deliberately make the retained source inaccessible as a packing-gate fault
    // probe. Ordinary callers keep their source alive through handle release.
    CHECK(::mprotect(f.source, 4096, PROT_NONE) == 0);
    const auto start = Clock::now();
    f.release(handle);
    CHECK(Clock::now() - start < std::chrono::seconds(1));
    CHECK(!resume.load()); // Worker finished while checkpoint still owns the lock.
    checkpoint.unblock();
    CHECK(::mprotect(f.source, 4096, PROT_READ | PROT_WRITE) == 0);
    CHECK(f.agent->batch_exists({"old", "replacement"}) == (std::vector<bool>{true, false}));
}
void eviction_error_preserved() {
    Fixture f;
    f.seed("old");
    f.identify_maintenance();
    fault.store(2);
    auto handle = f.submit("replacement");
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::io_error});
    CHECK(injected_errors.load() > 0);
    f.release(handle); fault.store(0);
    CHECK(!f.agent->batch_exists({"replacement"})[0]);
}
void healthy_disk_wins_after_error() {
    Fixture f(true);
    f.seed(disk_zero_key("old"));
    f.identify_maintenance();
    fault.store(2);
    auto key = disk_zero_key("replacement");
    auto handle = f.submit(key);
    CHECK(wait(*f.agent, handle) == std::vector<Status>{Status::success});
    CHECK(injected_errors.load() > 0);
    f.release(handle); fault.store(0);
    CHECK(f.agent->batch_exists({key})[0]);
}
}
extern "C" int fdatasync(int fd) {
    using Fn = int (*)(int);
    static auto actual = reinterpret_cast<Fn>(::dlsym(RTLD_NEXT, "fdatasync"));
    if (!actual) std::abort();
    auto mode = fault.load();
    if (mode) {
        struct stat st{};
        if (!::fstat(fd, &st) && st.st_ino == target_inode && st.st_dev == target_device) {
            if (mode == 3) { maintenance_thread.store(::pthread_self()); return actual(fd); }
            if (mode == 2 && ::pthread_equal(::pthread_self(), maintenance_thread.load()))
                return actual(fd);
            entered.store(true);
            if (mode == 2) { ++injected_errors; errno = EIO; return -1; }
            while (!resume.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return actual(fd);
}
int main() {
    try {
        contention_retries();
        healthy_disk_wins_before_contended_disk_unblocks();
        contention_keeps_deadline_and_cancel_gate();
        eviction_error_preserved();
        healthy_disk_wins_after_error();
        oversized_empty_disk_declines_without_packing();
        std::cout << "reclamation contention, deadlines, errors and healthy-disk fallback passed\n";
        return 0;
    } catch (const std::exception &error) {
        resume.store(true); fault.store(0);
        std::cerr << error.what() << '\n'; return 1;
    }
}
