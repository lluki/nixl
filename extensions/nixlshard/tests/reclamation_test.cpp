#include "nixlshard/storage.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
constexpr size_t unit = 512;
struct Fixture {
    std::string path; DiskConfig config;
    std::atomic<size_t> reads{0}, writes{0}, written_bytes{0}, barriers{0};
    MetadataIO hook; PersistenceBarrier sync_hook;
    explicit Fixture(size_t count = 8) {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-reclaimer-v2-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
        config.path = path; config.capacity_bytes = 4096 + count * 1024;
        config.unit_bytes = unit; config.metadata_alignment = unit;
        config.min_object_bytes = 1; config.max_object_bytes = unit * 4; config.create = true;
    }
    ~Fixture() { ::unlink(path.c_str()); }
    MetadataIO io() { return [&](int fd, bool write, uint64_t offset, void *buffer, size_t bytes) {
        if (write) { ++writes; written_bytes += bytes; } else ++reads;
        if (hook) { auto result = hook(fd, write, offset, buffer, bytes); if (result != Status::success) return result; }
        size_t done = 0;
        while (done < bytes) {
            ssize_t n = write ? ::pwrite(fd, static_cast<char *>(buffer) + done, bytes - done, offset + done)
                              : ::pread(fd, static_cast<char *>(buffer) + done, bytes - done, offset + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return Status::io_error;
            done += static_cast<size_t>(n);
        }
        return Status::success;
    }; }
    PersistenceBarrier sync() { return [&](int fd) {
        ++barriers; if (sync_hook) return sync_hook(fd);
        return ::fdatasync(fd) ? Status::io_error : Status::success;
    }; }
};
Allocation add(DiskIndex &disk, const std::string &key, size_t bytes = unit) {
    Allocation a; CHECK(disk.reserve(key, bytes, a) == Status::success && disk.publish(a) == Status::success); return a;
}
void whole_object_and_dual_resources() {
    Fixture f(4); DiskIndex disk(f.config, f.io(), f.sync());
    auto big = add(disk, "big", unit * 3); add(disk, "small");
    CHECK(disk.free_slots() == 0 && disk.free_records() == 2);
    CHECK(disk.evict_one() == Status::success);
    CHECK(!disk.exists("big") && disk.exists("small"));
    CHECK(disk.free_slots() == 3 && disk.free_records() == 3);
    Allocation old; CHECK(disk.pin("big", old, big.id) == Status::not_found);
    auto replacement = add(disk, "big", unit * 3);
    CHECK(replacement.id != big.id && disk.abort(big) == Status::not_found);
    CHECK(disk.free_slots() == 0);
}
void pinned_candidate_defers_and_old_identity_is_stale() {
    Fixture f; DiskIndex disk(f.config, f.io(), f.sync());
    auto first = add(disk, "first"); auto second = add(disk, "second");
    Allocation pinned; CHECK(disk.pin("first", pinned) == Status::success);
    CHECK(disk.evict_one() == Status::success);
    CHECK(disk.exists("first") && !disk.exists("second"));
    CHECK(disk.evict_one() == Status::busy);
    CHECK(disk.unpin(pinned) == Status::success && disk.evict_one() == Status::success);
    CHECK(!disk.exists("first"));
    auto newer = add(disk, "first");
    CHECK(newer.id != first.id && newer.id != second.id);
    CHECK(disk.pin("first", pinned, first.id) == Status::not_found);
}
void retirement_hides_lookup_but_holds_both_ledgers() {
    Fixture f(1); DiskIndex disk(f.config, f.io(), f.sync()); auto a = add(disk, "a");
    std::mutex mutex; std::condition_variable cv; bool entered = false, resume = false;
    f.hook = [&](int, bool write, uint64_t offset, void *, size_t) {
        if (write && offset == disk.record_offset(a.record)) {
            CHECK(!disk.exists("a")); // callback runs without index locks
            std::unique_lock guard(mutex); entered = true; cv.notify_all(); cv.wait(guard, [&] { return resume; });
        } return Status::success;
    };
    Status result = Status::io_error;
    std::thread thread([&] { result = disk.evict_one(); });
    { std::unique_lock guard(mutex); cv.wait(guard, [&] { return entered; }); }
    CHECK(!disk.exists("a") && disk.free_slots() == 0 && disk.free_records() == 0);
    Allocation replacement; CHECK(disk.reserve("a", unit, replacement) == Status::no_space);
    CHECK(disk.pin("a", replacement) == Status::not_found);
    CHECK(disk.close(CloseMode::discard) == Status::busy);
    { std::lock_guard guard(mutex); resume = true; } cv.notify_all(); thread.join();
    CHECK(result == Status::success && disk.free_slots() == 1 && disk.free_records() == 1);
    CHECK(disk.reserve("a", unit, replacement) == Status::success);
    CHECK(replacement.id != a.id && replacement.record == a.record && replacement.slots == a.slots);
    CHECK(disk.abort(replacement) == Status::success);
}
void invalidation_failure_never_reuses_and_dirty_recovery_discards() {
    Fixture f(2); uint64_t generation;
    {
        DiskIndex disk(f.config, f.io(), f.sync()); auto a = add(disk, "a"); add(disk, "healthy");
        generation = disk.generation();
        f.hook = [&](int, bool write, uint64_t offset, void *, size_t) {
            return write && offset == disk.record_offset(a.record) ? Status::io_error : Status::success;
        };
        CHECK(disk.evict_one() == Status::io_error && disk.state() == DeviceState::failed);
        CHECK(!disk.exists("a") && !disk.exists("healthy"));
        CHECK(disk.free_slots() == 0 && disk.free_records() == 0);
        Allocation b; CHECK(disk.reserve("b", unit, b) == Status::io_error);
        CHECK(disk.close(CloseMode::clean) == Status::io_error);
        CHECK(disk.close(CloseMode::discard) == Status::success);
    }
    f.hook = {}; auto config = f.config; config.create = false;
    DiskIndex recovered(config, f.io(), f.sync());
    CHECK(recovered.generation() != generation && recovered.enumerate().empty());
    CHECK(recovered.free_slots() == 2 && recovered.free_records() == 2);
}
void eviction_cost_is_per_record_without_eager_barriers() {
    Fixture f(16); DiskIndex disk(f.config, f.io(), f.sync());
    for (int i = 0; i < 12; ++i) add(disk, std::to_string(i));
    auto reads = f.reads.load(), writes = f.writes.load(), bytes = f.written_bytes.load(), barriers = f.barriers.load();
    for (int i = 0; i < 8; ++i) CHECK(disk.evict_one() == Status::success);
    CHECK(f.reads == reads + 8 && f.writes == writes + 8);
    CHECK(f.written_bytes == bytes + 8 * disk.metadata_record_bytes());
    CHECK(f.barriers == barriers);
    CHECK(disk.enumerate().size() == 4);
    CHECK(disk.close(CloseMode::clean) == Status::success);
}
void held_checkpoint_does_not_hold_allocator_or_index_lock() {
    Fixture f; DiskIndex disk(f.config, f.io(), f.sync()); add(disk, "a");
    std::mutex mutex; std::condition_variable cv; bool entered = false, resume = false;
    f.sync_hook = [&](int) {
        std::unique_lock guard(mutex); entered = true; cv.notify_all(); cv.wait(guard, [&] { return resume; });
        return Status::success;
    };
    Status checkpoint = Status::io_error;
    std::thread thread([&] { checkpoint = disk.checkpoint(); });
    { std::unique_lock guard(mutex); cv.wait(guard, [&] { return entered; }); }
    Allocation pending; CHECK(disk.reserve("b", unit, pending) == Status::success);
    CHECK(disk.publish(pending) == Status::success && disk.exists("b"));
    CHECK(disk.evict_one() == Status::busy); // bounded contention status, not capacity loss
    { std::lock_guard guard(mutex); resume = true; } cv.notify_all(); thread.join();
    CHECK(checkpoint == Status::success && disk.evict_one() == Status::success);
}
void stale_policy_and_drain_without_pressure() {
    struct Policy : EvictionPolicy {
        std::vector<uint64_t> candidates{std::numeric_limits<uint64_t>::max()};
        std::vector<Status> results;
        void on_block_added(const AllocationIdentity &id, size_t) noexcept override { candidates.push_back(id.id); }
        void on_block_read(uint64_t) noexcept override {}
        uint64_t nominate_eviction_candidate(uint64_t) noexcept override { return candidates.empty() ? 0 : candidates.front(); }
        void on_eviction_result(uint64_t id, Status result) noexcept override {
            results.push_back(result);
            candidates.erase(std::remove(candidates.begin(), candidates.end(), id), candidates.end());
            if (result == Status::busy) candidates.push_back(id);
        }
        void on_device_retired(uint64_t) noexcept override { candidates.clear(); }
    };
    Fixture f; auto policy = std::make_shared<Policy>();
    DiskIndex disk(f.config, f.io(), f.sync(), {}, policy);
    policy->candidates.push_back(std::numeric_limits<uint64_t>::max());
    add(disk, "a"); add(disk, "b");
    CHECK(disk.set_state(DeviceState::draining) == Status::success);
    Allocation denied; CHECK(disk.reserve("new", unit, denied) == Status::busy);
    CHECK(disk.evict_one() == Status::success && disk.enumerate().size() == 1);
    CHECK(policy->results.size() == 2 && policy->results[0] == Status::not_found &&
          policy->results[1] == Status::success);
    CHECK(disk.evict_one() == Status::success && disk.enumerate().empty());
    CHECK(disk.close(CloseMode::clean) == Status::success);
}
}
int main() {
    const std::pair<const char *, void (*)()> tests[] = {
        {"whole object and dual resources", whole_object_and_dual_resources},
        {"pinned candidate and stale lifetime", pinned_candidate_defers_and_old_identity_is_stale},
        {"RETIRING visibility and reuse fencing", retirement_hides_lookup_but_holds_both_ledgers},
        {"metadata invalidation failure", invalidation_failure_never_reuses_and_dirty_recovery_discards},
        {"per-record IO without eager durability", eviction_cost_is_per_record_without_eager_barriers},
        {"checkpoint contention remains bounded", held_checkpoint_does_not_hold_allocator_or_index_lock},
        {"stale policy and demand-free drain", stale_policy_and_drain_without_pressure},
    };
    try { for (const auto &[name, test] : tests) { test(); std::cout << "PASS " << name << '\n'; } }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
