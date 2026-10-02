#include "nixlshard/storage.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace nixlshard;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
namespace {
constexpr size_t metadata = 16384, unit = 512, header = 4096;
struct File {
    std::string path;
    explicit File(size_t slots = 8) {
        auto pattern = (std::filesystem::temp_directory_path() / "nixlshard-storage-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        int fd = ::mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("mkstemp");
        ::close(fd); path = name.data();
        cfg = DiskConfig{path, metadata + slots * unit, unit, metadata, true};
    }
    ~File() { ::unlink(path.c_str()); }
    DiskConfig cfg;
    DiskConfig reopen() const { auto result = cfg; result.create = false; return result; }
};
void expect_throw(const std::function<void()> &fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception &) { caught = true; }
    CHECK(caught);
}
std::vector<uint8_t> read_range(const std::string &path, size_t offset, size_t bytes) {
    int fd = ::open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    std::vector<uint8_t> result(bytes);
    CHECK(::pread(fd, result.data(), bytes, offset) == static_cast<ssize_t>(bytes));
    ::close(fd);
    return result;
}
void write_range(const std::string &path, size_t offset, const std::vector<uint8_t> &data) {
    int fd = ::open(path.c_str(), O_WRONLY);
    CHECK(fd >= 0);
    CHECK(::pwrite(fd, data.data(), data.size(), offset) == static_cast<ssize_t>(data.size()));
    CHECK(::fdatasync(fd) == 0);
    ::close(fd);
}
uint64_t load_le(const std::vector<uint8_t> &data, size_t offset) {
    uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) out |= uint64_t(data.at(offset + i)) << (8 * i);
    return out;
}
void store_le(std::vector<uint8_t> &data, size_t offset, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) data.at(offset + i) = static_cast<uint8_t>(value >> (8 * i));
}
uint64_t digest(const std::vector<uint8_t> &data, size_t offset, size_t bytes) {
    uint64_t out = 14695981039346656037ULL;
    for (size_t i = offset; i < offset + bytes; ++i) out = (out ^ data.at(i)) * 1099511628211ULL;
    return out;
}
Allocation add(DiskIndex &disk, const std::string &key, size_t bytes) {
    Allocation a;
    CHECK(disk.reserve(key, bytes, a) == Status::success);
    CHECK(disk.publish(a.id) == Status::success);
    return a;
}
void write_payload(DiskIndex &disk, const Allocation &a, char value) {
    std::vector<char> data(disk.unit_bytes(), value);
    for (auto slot : a.slots)
        CHECK(::pwrite(disk.fd(), data.data(), data.size(), disk.slot_offset(slot)) == static_cast<ssize_t>(data.size()));
}
void roundtrip_and_duplicates() {
    File file;
    Allocation original;
    {
        DiskIndex disk(file.cfg);
        CHECK(disk.free_slots() == 8);
        Allocation pending;
        CHECK(disk.reserve("pending", 30, pending) == Status::success);
        CHECK(!disk.exists("pending"));
        CHECK(disk.snapshot_keys().empty());
        Allocation duplicate;
        CHECK(disk.reserve("pending", 30, duplicate) == Status::busy);
        CHECK(disk.abort(pending.id) == Status::success);
        CHECK(disk.abort(pending.id) == Status::not_found);
        original = add(disk, std::string("namespace\0key", 13), unit + 7);
        write_payload(disk, original, 'x');
        CHECK(disk.reserve(original.key, original.bytes, duplicate) == Status::success);
        CHECK(duplicate.already_present && duplicate.id == original.id);
        CHECK(disk.reserve(original.key, 1, duplicate) == Status::invalid_input);
        CHECK(disk.free_slots() == 6);
        CHECK(disk.snapshot_keys() == std::vector<std::string>{original.key});
        CHECK(disk.checkpoint() == Status::success);
    }
    {
        DiskIndex disk(file.reopen());
        Allocation recovered;
        CHECK(disk.pin(original.key, recovered) == Status::success);
        CHECK(recovered.id == original.id && recovered.bytes == original.bytes && recovered.slots == original.slots);
        CHECK(disk.pin("namespace", recovered) == Status::not_found);
        CHECK(disk.evict_one() == Status::busy);
        CHECK(disk.unpin(original.id) == Status::success);
        CHECK(disk.unpin(original.id) == Status::invalid_input);
        CHECK(disk.evict_one() == Status::success);
        CHECK(disk.free_slots() == 8);
        CHECK(disk.snapshot_keys().empty());
        CHECK(disk.publish(original.id) == Status::not_found);
        CHECK(disk.unpin(original.id) == Status::not_found);
    }
    DiskIndex disk(file.reopen());
    CHECK(!disk.exists(original.key));
}
void scatter_and_atomic_failure() {
    File file(4);
    DiskIndex disk(file.cfg);
    auto a = add(disk, "a", unit);
    auto b = add(disk, "b", unit);
    auto c = add(disk, "c", unit);
    auto d = add(disk, "d", unit);
    Allocation pinned;
    CHECK(disk.pin("b", pinned) == Status::success);
    CHECK(disk.evict_one() == Status::success); // a
    CHECK(disk.evict_one() == Status::success); // c, skipping pinned b
    Allocation scatter;
    CHECK(disk.reserve("scatter", unit * 3, scatter) == Status::no_space);
    CHECK(disk.free_slots() == 2 && !disk.exists("scatter"));
    CHECK(disk.reserve("scatter", unit + 1, scatter) == Status::success);
    CHECK(scatter.slots == std::vector<uint64_t>({c.slots[0], a.slots[0]}));
    CHECK(disk.abort(scatter.id) == Status::success);
    Allocation invalid;
    CHECK(disk.reserve("", 1, invalid) == Status::invalid_input);
    CHECK(disk.reserve("zero", 0, invalid) == Status::invalid_input);
    CHECK(disk.reserve("huge", std::numeric_limits<size_t>::max(), invalid) == Status::no_space);
    CHECK(disk.reserve(std::string(4097, 'k'), 1, invalid) == Status::invalid_input);
    CHECK(disk.free_slots() == 2);
    CHECK(disk.unpin(b.id) == Status::success);
    CHECK(disk.publish(a.id) == Status::not_found);
    CHECK(disk.abort(d.id) == Status::invalid_input);
    expect_throw([&] { disk.slot_offset(4); });
}
void process_crash_after_reuse() {
    File file(1);
    pid_t child = ::fork();
    CHECK(child >= 0);
    if (child == 0) {
        try {
            DiskIndex disk(file.cfg);
            auto old = add(disk, "old", unit);
            write_payload(disk, old, 'a');
            CHECK(disk.checkpoint() == Status::success);
            CHECK(disk.evict_one() == Status::success); // deletion must be durable
            auto fresh = add(disk, "fresh", unit);
            CHECK(fresh.slots == old.slots && fresh.id != old.id);
            write_payload(disk, fresh, 'b');
            CHECK(::fdatasync(disk.fd()) == 0);
            ::_exit(0); // no destructor or insertion checkpoint
        } catch (...) { ::_exit(1); }
    }
    int status;
    CHECK(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    DiskIndex disk(file.reopen());
    CHECK(!disk.exists("old") && !disk.exists("fresh") && disk.free_slots() == 1);
    char value;
    CHECK(::pread(disk.fd(), &value, 1, disk.slot_offset(0)) == 1 && value == 'b');
    Allocation out;
    CHECK(disk.pin("old", out) == Status::not_found); // never associate overwritten bytes with old key
}
void commit_crash_boundaries() {
    File file;
    {
        DiskIndex disk(file.cfg); // seq 1 body/selector 0
        auto a = add(disk, "a", unit);
        write_payload(disk, a, 'a');
        CHECK(disk.checkpoint() == Status::success); // seq 2 body/selector 1
    }
    auto old_selector = read_range(file.path, 0, header);
    {
        DiskIndex disk(file.reopen());
        CHECK(disk.evict_one() == Status::success); // seq 3 body/selector 0
    }
    auto new_selector = read_range(file.path, 0, header);
    // Crash after fully synchronized successor body but before selector publication.
    write_range(file.path, 0, old_selector);
    { DiskIndex disk(file.reopen()); CHECK(disk.exists("a")); }
    // Partial selector write: new sequence field present but checksum old.
    auto torn = old_selector;
    std::copy(new_selector.begin(), new_selector.begin() + 32, torn.begin());
    write_range(file.path, 0, torn);
    { DiskIndex disk(file.reopen()); CHECK(disk.exists("a")); }
    // Fully committed removal; recovery must never use the older insertion state.
    write_range(file.path, 0, new_selector);
    { DiskIndex disk(file.reopen()); CHECK(!disk.exists("a")); CHECK(disk.free_slots() == 8); }
    // Corrupt newest committed body: reject the disk rather than resurrect old state.
    auto byte = read_range(file.path, 2 * header + 40, 1);
    byte[0] ^= 0x80;
    write_range(file.path, 2 * header + 40, byte);
    expect_throw([&] { DiskIndex disk(file.reopen()); });
    expect_throw([&] { DiskIndex disk(file.cfg); }); // create does not erase corrupt data
}
void geometry_and_metadata_bounds() {
    File file;
    auto cfg = file.cfg; cfg.unit_bytes = 513;
    expect_throw([&] { DiskIndex disk(cfg); });
    cfg = file.cfg; cfg.metadata_bytes = 100;
    expect_throw([&] { DiskIndex disk(cfg); });
    {
        DiskIndex disk(file.cfg);
        expect_throw([&] { DiskIndex other(file.reopen()); }); // one writer per device
    }
    cfg = file.reopen(); cfg.capacity_bytes += unit;
    expect_throw([&] { DiskIndex disk(cfg); });
    cfg = file.reopen(); cfg.unit_bytes *= 2;
    expect_throw([&] { DiskIndex disk(cfg); });
    File unknown;
    int fd = ::open(unknown.path.c_str(), O_WRONLY);
    CHECK(fd >= 0 && ::ftruncate(fd, unknown.cfg.capacity_bytes) == 0);
    ::close(fd);
    write_range(unknown.path, 123, {99});
    expect_throw([&] { DiskIndex disk(unknown.cfg); });
    File bounded(32);
    DiskIndex disk(bounded.cfg);
    Allocation a;
    CHECK(disk.reserve(std::string(3950, 'a'), unit, a) == Status::success);
    CHECK(disk.publish(a.id) == Status::success);
    CHECK(disk.reserve(std::string(3950, 'b'), unit, a) == Status::no_space);
    CHECK(disk.free_slots() == 31);
}
void malformed_committed_records() {
    File file;
    {
        DiskIndex disk(file.cfg);
        add(disk, "a", unit + 1);
        CHECK(disk.checkpoint() == Status::success); // body 1, selector 1
    }
    auto original = read_range(file.path, 0, metadata);
    const size_t body = 3 * header, selector = header;
    const auto forge = [&](size_t offset, uint64_t value) {
        auto bytes = original;
        store_le(bytes, body + offset, value);
        auto length = static_cast<size_t>(load_le(bytes, selector + 24));
        store_le(bytes, selector + 32, digest(bytes, body, length));
        store_le(bytes, selector + 48, digest(bytes, selector, 48));
        write_range(file.path, 0, bytes);
        expect_throw([&] { DiskIndex disk(file.reopen()); });
    };
    forge(113, load_le(original, body + 105)); // duplicate slot ownership
    forge(105, 10000); // out-of-bounds slot
    forge(80, 0); // zero-length allocation
    forge(88, 4097); // oversized key
    forge(96, 1); // wrong slot count for length
    forge(64, std::numeric_limits<uint64_t>::max()); // unreasonable record count
    forge(56, 0); // invalid next allocation identity
    write_range(file.path, 0, original);
    DiskIndex disk(file.reopen());
    CHECK(disk.exists("a"));
}
void checkpoint_eviction_race() {
    File file(8);
    {
        DiskIndex disk(file.cfg);
        add(disk, "old", unit);
        CHECK(disk.checkpoint() == Status::success);
        std::atomic<bool> done{false};
        std::atomic<bool> valid{true};
        std::thread checkpointer([&] {
            while (!done) {
                auto status = disk.checkpoint();
                if (status != Status::success && status != Status::busy) valid = false;
                std::this_thread::yield();
            }
        });
        Status result;
        do { result = disk.evict_one(); std::this_thread::yield(); } while (result == Status::busy);
        CHECK(result == Status::success);
        auto fresh = add(disk, "fresh", unit);
        write_payload(disk, fresh, 'z');
        done = true;
        checkpointer.join();
        CHECK(valid);
        CHECK(disk.checkpoint() == Status::success);
    }
    DiskIndex disk(file.reopen());
    CHECK(!disk.exists("old") && disk.exists("fresh"));
}
} // namespace
int main() {
    try {
        roundtrip_and_duplicates();
        scatter_and_atomic_failure();
        process_crash_after_reuse();
        commit_crash_boundaries();
        geometry_and_metadata_bounds();
        malformed_committed_records();
        checkpoint_eviction_race();
        std::cout << "storage tests passed (7 suites)\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
