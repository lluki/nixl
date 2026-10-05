#include "nixlshard/storage.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
constexpr size_t unit = 512, header = 4096;
using Hook = std::function<Status(int, bool, uint64_t, void *, size_t)>;
struct File {
    std::string path;
    DiskConfig config;
    std::atomic<size_t> reads{0}, writes{0}, syncs{0};
    Hook hook;
    std::function<Status(int)> sync_hook;
    explicit File(size_t records = 8, size_t extra = 0) {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-storage-v2-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data()); CHECK(fd >= 0); ::close(fd); path = name.data();
        config.path = path; config.capacity_bytes = header + records * (unit + unit) + extra;
        config.unit_bytes = unit; config.metadata_alignment = unit;
        config.min_object_bytes = 1; config.max_object_bytes = 3 * unit;
        config.namespace_id = "full-model/layout-schema"; config.create = true;
    }
    ~File() { ::unlink(path.c_str()); }
    DiskConfig reopen() const { auto c = config; c.create = false; return c; }
    MetadataIO io() {
        return [this](int fd, bool write, uint64_t offset, void *data, size_t bytes) {
            CHECK(reinterpret_cast<uintptr_t>(data) % config.metadata_alignment == 0);
            CHECK(offset % config.metadata_alignment == 0 && bytes % config.metadata_alignment == 0);
            if (write) ++writes; else ++reads;
            if (hook) { auto s = hook(fd, write, offset, data, bytes); if (s != Status::success) return s; }
            size_t done = 0;
            while (done < bytes) {
                auto n = write ? ::pwrite(fd, static_cast<char *>(data) + done, bytes - done, offset + done)
                               : ::pread(fd, static_cast<char *>(data) + done, bytes - done, offset + done);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) return Status::io_error;
                done += static_cast<size_t>(n);
            }
            return Status::success;
        };
    }
    PersistenceBarrier barrier() {
        return [this](int fd) { ++syncs; if (sync_hook) return sync_hook(fd);
            return ::fdatasync(fd) ? Status::io_error : Status::success; };
    }
};
void expect_throw(const std::function<void()> &fn) {
    bool threw = false; try { fn(); } catch (const std::exception &) { threw = true; } CHECK(threw);
}
Allocation add(DiskIndex &disk, const std::string &key, size_t bytes = unit) {
    Allocation a; CHECK(disk.reserve(key, bytes, a) == Status::success);
    CHECK(disk.publish(a) == Status::success); return a;
}
uint64_t read64(const unsigned char *data, size_t offset) {
    uint64_t value = 0; for (size_t i = 0; i < 8; ++i) value |= uint64_t(data[offset + i]) << (8 * i); return value;
}
void write64(unsigned char *data, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) data[offset + i] = static_cast<unsigned char>(value >> (8 * i));
}
uint64_t digest(const unsigned char *data, size_t size) {
    uint64_t v = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) v = (v ^ data[i]) * 1099511628211ULL;
    return v;
}
void mutate_record(const File &file, uint64_t offset, size_t bytes,
                   const std::function<void(unsigned char *)> &fn) {
    int fd = ::open(file.path.c_str(), O_RDWR); CHECK(fd >= 0);
    std::vector<unsigned char> data(bytes); CHECK(::pread(fd, data.data(), bytes, offset) == static_cast<ssize_t>(bytes));
    fn(data.data()); write64(data.data(), bytes - 8, digest(data.data(), bytes - 8));
    CHECK(::pwrite(fd, data.data(), bytes, offset) == static_cast<ssize_t>(bytes)); CHECK(::fdatasync(fd) == 0); ::close(fd);
}
void clean_restore_and_ssd_reads() {
    File f; Allocation old;
    {
        DiskIndex disk(f.config, f.io(), f.barrier());
        CHECK(disk.metadata_record_bytes() == unit && disk.metadata_record_count() == 8);
        CHECK(disk.payload_base() % unit == 0 && disk.free_records() == 8 && disk.free_slots() == 8);
        old = add(disk, std::string("a\0b", 3), unit + 7);
        CHECK(disk.enumerate().size() == 1 && disk.enumerate()[0].record == old.record);
        auto before = f.reads.load(); Allocation read;
        CHECK(disk.pin(old.key, read) == Status::success && read.slots == old.slots);
        CHECK(f.reads == before + 1); CHECK(disk.unpin(read) == Status::success);
        CHECK(disk.pin(old.key, read, old.id + 1) == Status::not_found);
        CHECK(f.reads == before + 1); // stale upper identity cannot issue SSD/payload access
        CHECK(disk.close(CloseMode::clean) == Status::success);
    }
    {
        DiskIndex disk(f.reopen(), f.io(), f.barrier()); Allocation read;
        CHECK(disk.pin(old.key, read) == Status::success);
        CHECK(read.id == old.id && read.bytes == old.bytes && read.slots == old.slots);
        CHECK(read.generation != old.generation);
        CHECK(disk.unpin(old) == Status::not_found && disk.unpin(read) == Status::success);
        Allocation next = add(disk, "next"); CHECK(next.id != old.id);
        CHECK(disk.close(CloseMode::clean) == Status::success);
    }
}
void dirty_close_and_checkpoint_discard() {
    File f;
    { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "a");
      CHECK(disk.checkpoint() == Status::success); CHECK(disk.close(CloseMode::discard) == Status::success); }
    { DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("a"));
      CHECK(disk.free_slots() == 8 && disk.free_records() == 8); add(disk, "b"); }
    DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("b"));
}
void process_crash_after_reuse_discards_all() {
    File f;
    pid_t child = ::fork(); CHECK(child >= 0);
    if (!child) {
        try { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "old");
              CHECK(disk.evict_one() == Status::success); add(disk, "new");
              CHECK(disk.checkpoint() == Status::success); ::_exit(0); }
        catch (...) { ::_exit(3); }
    }
    int status = 0; CHECK(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("old") && !disk.exists("new"));
    CHECK(disk.free_slots() == 8 && disk.free_records() == 8);
}
void dual_ledger_and_ordered_scatter() {
    {
        File f(4, unit); DiskIndex disk(f.config, f.io(), f.barrier());
        for (int i = 0; i < 4; ++i) add(disk, std::to_string(i));
        CHECK(disk.free_records() == 0 && disk.free_slots() == 1);
        Allocation failed; CHECK(disk.reserve("overflow", unit, failed) == Status::no_space);
        CHECK(!failed.id && disk.free_slots() == 1 && !disk.exists("overflow"));
    }
    {
        File f(4); DiskIndex disk(f.config, f.io(), f.barrier());
        Allocation pending; CHECK(disk.reserve("pending", unit + 1, pending) == Status::success);
        Allocation duplicate; CHECK(disk.reserve("pending", unit + 1, duplicate) == Status::busy);
        CHECK(disk.abort(pending) == Status::success && disk.free_records() == 4 && disk.free_slots() == 4);
        add(disk, "large-a", unit + 1); add(disk, "large-b", unit + 1);
        CHECK(disk.free_slots() == 0 && disk.free_records() == 2);
        CHECK(disk.reserve("too-big", unit, duplicate) == Status::no_space);
        CHECK(disk.free_records() == 2);
    }
    {
        File f; DiskIndex disk(f.config, f.io(), f.barrier());
        add(disk, "a"); auto b = add(disk, "b"); add(disk, "c");
        Allocation pinned; CHECK(disk.pin("b", pinned) == Status::success);
        auto syncs = f.syncs.load();
        CHECK(disk.evict_one() == Status::success && disk.evict_one() == Status::success);
        CHECK(f.syncs == syncs); // no eager deletion durability barriers
        auto scatter = add(disk, "scatter", unit + 1);
        CHECK(scatter.slots == std::vector<uint64_t>({2, 0}));
        CHECK(disk.unpin(pinned) == Status::success);
        CHECK(disk.pin("b", pinned, b.id) == Status::success && disk.unpin(pinned) == Status::success);
        CHECK(disk.close(CloseMode::clean) == Status::success);
    }
}
void metadata_fetch_claim_blocks_reuse_and_close() {
    File f(1); DiskIndex disk(f.config, f.io(), f.barrier()); auto original = add(disk, "a");
    std::mutex mutex; std::condition_variable cv; bool entered = false, released = false;
    f.hook = [&](int, bool write, uint64_t offset, void *, size_t) {
        if (!write && offset == disk.record_offset(original.record)) {
            std::unique_lock lock(mutex); entered = true; cv.notify_all(); cv.wait(lock, [&] { return released; });
        }
        return Status::success;
    };
    Allocation read; Status result = Status::io_error;
    std::thread thread([&] { result = disk.pin("a", read); });
    { std::unique_lock lock(mutex); cv.wait(lock, [&] { return entered; }); }
    CHECK(disk.evict_one() == Status::busy && disk.free_slots() == 0 && disk.free_records() == 0);
    CHECK(disk.close(CloseMode::clean) == Status::busy);
    Allocation rejected; CHECK(disk.reserve("new", unit, rejected) == Status::busy);
    { std::lock_guard lock(mutex); released = true; } cv.notify_all(); thread.join();
    CHECK(result == Status::success && read.id == original.id);
    CHECK(disk.unpin(read) == Status::success && disk.close(CloseMode::clean) == Status::success);
}
void metadata_write_failure_and_record_validation() {
    {
        File f; DiskIndex disk(f.config, f.io(), f.barrier()); Allocation pending;
        CHECK(disk.reserve("a", unit, pending) == Status::success);
        f.hook = [](int, bool write, uint64_t offset, void *, size_t) {
            return write && offset >= header ? Status::io_error : Status::success;
        };
        CHECK(disk.publish(pending) == Status::io_error && !disk.exists("a"));
        CHECK(disk.state() == DeviceState::failed && disk.free_records() == 7);
        CHECK(disk.abort(pending) == Status::success); CHECK(disk.close(CloseMode::discard) == Status::success);
    }
    {
        File f; DiskIndex disk(f.config, f.io(), f.barrier()); auto a = add(disk, "a");
        mutate_record(f, disk.record_offset(a.record), disk.metadata_record_bytes(),
                      [](unsigned char *p) { p[56] = 'b'; });
        Allocation read; CHECK(disk.pin("a", read) == Status::io_error);
        CHECK(!read.id && !disk.exists("a") && disk.state() == DeviceState::failed);
    }
    {
        File f; uint64_t offset; size_t bytes;
        { DiskIndex disk(f.config, f.io(), f.barrier()); auto a = add(disk, "a", unit + 1);
          offset = disk.record_offset(a.record); bytes = disk.metadata_record_bytes();
          CHECK(disk.close(CloseMode::clean) == Status::success); }
        mutate_record(f, offset, bytes, [](unsigned char *p) { write64(p, 56 + 32 + 8, read64(p, 56 + 32)); });
        DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("a"));
        CHECK(disk.free_slots() == 8 && disk.free_records() == 8);
    }
}
void marker_failures_namespace_and_unknown_media() {
    {
        File f; int fd = ::open(f.path.c_str(), O_RDWR); CHECK(fd >= 0);
        CHECK(::ftruncate(fd, f.config.capacity_bytes) == 0);
        unsigned char magic[8]; write64(magic, 0, 0x31544d435248534eULL);
        CHECK(::pwrite(fd, magic, 8, 0) == 8); ::close(fd);
        expect_throw([&] { DiskIndex disk(f.reopen(), f.io(), f.barrier()); });
        auto reset = f.reopen(); reset.reset = true;
        DiskIndex disk(reset, f.io(), f.barrier()); CHECK(disk.enumerate().empty());
    }
    {
        File f;
        { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "a"); CHECK(disk.close(CloseMode::clean) == Status::success); }
        auto different = f.reopen(); different.namespace_id = "another exact domain";
        expect_throw([&] { DiskIndex disk(different, f.io(), f.barrier()); });
        different.reset = true; DiskIndex disk(different, f.io(), f.barrier()); CHECK(!disk.exists("a"));
    }
    {
        File f; int fd = ::open(f.path.c_str(), O_RDWR); CHECK(fd >= 0);
        CHECK(::ftruncate(fd, f.config.capacity_bytes) == 0);
        CHECK(::pwrite(fd, "unknown", 7, 0) == 7); ::close(fd);
        auto c = f.config; c.reset = true;
        expect_throw([&] { DiskIndex disk(c, f.io(), f.barrier()); });
    }
    {
        File f;
        { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "a"); CHECK(disk.close(CloseMode::clean) == Status::success); }
        f.hook = [](int fd, bool write, uint64_t offset, void *data, size_t) {
            if (write && offset == 0) {
                CHECK(::pwrite(fd, data, 128, 0) == 128); CHECK(::fdatasync(fd) == 0);
                return Status::io_error; // torn DIRTY write cannot authorize admission
            }
            return Status::success;
        };
        expect_throw([&] { DiskIndex disk(f.reopen(), f.io(), f.barrier()); });
        f.hook = {};
        DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("a"));
    }
    {
        File f;
        { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "a");
          f.sync_hook = [](int) { return Status::io_error; };
          CHECK(disk.close(CloseMode::clean) == Status::io_error); }
        f.sync_hook = {};
        DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("a"));
    }
    {
        File f;
        { DiskIndex disk(f.config, f.io(), f.barrier()); add(disk, "a");
          f.hook = [](int fd, bool write, uint64_t offset, void *data, size_t) {
              if (write && offset == 0 && read64(static_cast<unsigned char *>(data), 112) != 0) {
                  CHECK(::pwrite(fd, data, 128, 0) == 128); CHECK(::fdatasync(fd) == 0); return Status::io_error;
              } return Status::success;
          };
          CHECK(disk.close(CloseMode::clean) == Status::io_error); }
        f.hook = {};
        DiskIndex disk(f.reopen(), f.io(), f.barrier()); CHECK(!disk.exists("a"));
    }
}
void bounds_drain_and_descriptor_retirement() {
    File f; DiskIndex disk(f.config, f.io(), f.barrier()); Allocation a;
    CHECK(disk.reserve(std::string(33, 'x'), unit, a) == Status::invalid_input);
    CHECK(disk.reserve("", unit, a) == Status::invalid_input);
    CHECK(disk.reserve("zero", 0, a) == Status::invalid_input);
    CHECK(disk.reserve("overflow", std::numeric_limits<size_t>::max(), a) == Status::invalid_input);
    CHECK(disk.reserve("pending", unit, a) == Status::success);
    CHECK(disk.set_state(DeviceState::draining) == Status::success);
    Allocation another; CHECK(disk.reserve("new", unit, another) == Status::busy);
    CHECK(disk.publish(a) == Status::success); // established reservation can finish
    CHECK(disk.set_state(DeviceState::offline) == Status::success);
    CHECK(disk.pin("pending", another) == Status::not_ready && disk.exists("pending"));
    CHECK(disk.set_state(DeviceState::active) == Status::success);
    CHECK(disk.close(CloseMode::clean) == Status::success);
    File fail; int observed_fd = -1; bool retired = false;
    fail.hook = [](int, bool, uint64_t, void *, size_t) { return Status::io_error; };
    expect_throw([&] { DiskIndex bad(fail.config, fail.io(), fail.barrier(), {}, {},
        [&](int fd) { CHECK(::fcntl(fd, F_GETFD) >= 0); observed_fd = fd; retired = true; return Status::success; }); });
    CHECK(retired && observed_fd >= 0 && ::fcntl(observed_fd, F_GETFD) == -1 && errno == EBADF);
    retired = false; observed_fd = -1;
    expect_throw([&] { DiskIndex bad(fail.config, fail.io(), fail.barrier(), {}, {},
        [&](int fd) { observed_fd = fd; retired = true; return Status::io_error; }); });
    CHECK(retired && observed_fd >= 0 && ::fcntl(observed_fd, F_GETFD) >= 0);
    ::close(observed_fd); // integration retains fd until failed deregistration is resolved
    auto cfg = f.reopen(); cfg.max_object_bytes = std::numeric_limits<size_t>::max();
    expect_throw([&] { DiskIndex invalid(cfg, f.io(), f.barrier()); });
}
void identity_bound_events_and_policy() {
    struct Policy : EvictionPolicy {
        std::vector<std::string> events; uint64_t candidate = 0;
        void on_block_added(const AllocationIdentity &a, size_t) noexcept override { events.push_back("add"); candidate = a.id; }
        void on_block_read(uint64_t) noexcept override { events.push_back("read"); }
        uint64_t nominate_eviction_candidate(uint64_t) noexcept override { events.push_back("candidate"); return candidate; }
        void on_eviction_result(uint64_t, Status result) noexcept override { events.push_back("result"); if (result != Status::busy) candidate = 0; }
        void on_device_retired(uint64_t) noexcept override { events.push_back("retired"); candidate = 0; }
    };
    File f; auto policy = std::make_shared<Policy>(); std::vector<IndexEvent> events;
    DiskIndex disk(f.config, f.io(), f.barrier(), [&](const IndexEvent &e) { events.push_back(e); }, policy);
    policy->events.clear(); auto a = add(disk, "a"); Allocation pinned;
    CHECK(disk.pin("a", pinned) == Status::success && disk.evict_one() == Status::busy);
    CHECK(policy->events[0] == "add" && policy->events[1] == "read");
    CHECK(disk.unpin(pinned) == Status::success && disk.evict_one() == Status::success);
    auto newer = add(disk, "a"); CHECK(newer.id != a.id);
    CHECK(events.size() == 3 && events[1].kind == IndexEventKind::evicted &&
          events[1].identity.id == a.id && events[2].identity.id == newer.id);
    CHECK(disk.pin("a", pinned, a.id) == Status::not_found);
    CHECK(disk.set_state(DeviceState::failed) == Status::success);
    CHECK(!disk.exists("a") && disk.enumerate().empty());
    CHECK(events.back().kind == IndexEventKind::device_retired && events.back().identity.id == newer.id);
    CHECK(events[0].sequence < events[1].sequence && events[1].sequence < events[2].sequence);
}
void ordered_events_during_concurrent_replacement() {
    File f; std::mutex mutex; std::condition_variable cv;
    bool first_entered = false, release = false;
    std::vector<IndexEvent> observed;
    DiskIndex disk(f.config, f.io(), f.barrier(), [&](const IndexEvent &event) {
        std::unique_lock guard(mutex);
        observed.push_back(event);
        if (observed.size() == 1) {
            first_entered = true; cv.notify_all(); cv.wait(guard, [&] { return release; });
        }
        (void)disk.snapshot_keys(); // read-only reentrancy must not hold index locks
    });
    Allocation original; CHECK(disk.reserve("a", unit, original) == Status::success);
    Status first = Status::io_error, evicted = Status::io_error, second = Status::io_error;
    std::thread publisher([&] { first = disk.publish(original); });
    { std::unique_lock guard(mutex); cv.wait(guard, [&] { return first_entered; }); }
    std::thread evicter([&] { evicted = disk.evict_one(); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((disk.exists("a") || disk.free_records() != disk.metadata_record_count()) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    Allocation newer; auto reserved = disk.reserve("a", unit, newer);
    std::thread replacement([&] { if (reserved == Status::success) second = disk.publish(newer); });
    { std::lock_guard guard(mutex); release = true; } cv.notify_all();
    publisher.join(); evicter.join(); replacement.join();
    CHECK(first == Status::success && evicted == Status::success && second == Status::success);
    CHECK(observed.size() == 3 && observed[0].kind == IndexEventKind::committed &&
          observed[1].kind == IndexEventKind::evicted && observed[2].kind == IndexEventKind::committed);
    CHECK(observed[0].identity.id == original.id && observed[1].identity.id == original.id &&
          observed[2].identity.id == newer.id && newer.id != original.id);
    CHECK(observed[0].sequence < observed[1].sequence && observed[1].sequence < observed[2].sequence);
}
}
int main() {
    const std::pair<const char *, void (*)()> tests[] = {
        {"clean restore and authoritative SSD reads", clean_restore_and_ssd_reads},
        {"DIRTY checkpoint/discard/destructor", dirty_close_and_checkpoint_discard},
        {"process crash after reuse", process_crash_after_reuse_discards_all},
        {"dual ledgers and ordered scatter", dual_ledger_and_ordered_scatter},
        {"metadata claim before I/O", metadata_fetch_claim_blocks_reuse_and_close},
        {"metadata failures and validation", metadata_write_failure_and_record_validation},
        {"marker/namespace/media failures", marker_failures_namespace_and_unknown_media},
        {"bounds/drain/descriptor retirement", bounds_drain_and_descriptor_retirement},
        {"identity events and pluggable policy", identity_bound_events_and_policy},
        {"ordered events under concurrent replacement", ordered_events_during_concurrent_replacement},
    };
    try { for (const auto &[name, test] : tests) { test(); std::cout << "PASS " << name << '\n'; } }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
