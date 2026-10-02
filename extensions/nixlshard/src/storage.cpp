#include "nixlshard/storage.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <linux/fs.h>
#include <mutex>
#include <stdexcept>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace nixlshard {
namespace {
constexpr uint64_t format_magic = 0x314744524148534eULL; // NSHARDG1, little endian
constexpr uint64_t selector_magic = 0x31544d435248534eULL;
constexpr uint64_t format_version = 1;
constexpr size_t selector_bytes = 4096;
constexpr size_t max_key_bytes = 4096;

uint64_t hash(const uint8_t *data, size_t bytes) {
    uint64_t result = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) result = (result ^ data[i]) * 1099511628211ULL;
    return result;
}
void put(std::vector<uint8_t> &out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
uint64_t get(const std::vector<uint8_t> &data, size_t &pos) {
    if (pos > data.size() || data.size() - pos < 8) throw std::runtime_error("truncated G3 metadata");
    uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i) result |= uint64_t(data[pos++]) << (8 * i);
    return result;
}
bool io_all(int fd, void *data, size_t length, uint64_t offset, bool write) {
    auto *p = static_cast<uint8_t *>(data);
    while (length) {
        ssize_t n = write ? ::pwrite(fd, p, length, static_cast<off_t>(offset))
                          : ::pread(fd, p, length, static_cast<off_t>(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        length -= static_cast<size_t>(n);
        offset += static_cast<uint64_t>(n);
    }
    return true;
}
bool sync_data(int fd) {
    int result;
    do { result = ::fdatasync(fd); } while (result < 0 && errno == EINTR);
    return result == 0;
}
uint64_t random_identity() {
    uint64_t result;
    if (::getrandom(&result, sizeof(result), 0) != sizeof(result))
        throw std::runtime_error("cannot obtain G3 incarnation identity");
    return (result & 0x3fffffffffffffffULL) + 1;
}
struct Selector { bool valid = false; uint64_t seq = 0, bytes = 0, digest = 0, uuid = 0; };
Selector decode_selector(const std::vector<uint8_t> &data) {
    size_t pos = 0;
    Selector out;
    if (get(data, pos) != selector_magic || get(data, pos) != format_version) return out;
    out.seq = get(data, pos);
    out.bytes = get(data, pos);
    out.digest = get(data, pos);
    out.uuid = get(data, pos);
    out.valid = get(data, pos) == hash(data.data(), 48) && out.seq != 0;
    return out;
}
} // namespace

struct DiskIndex::Impl {
    enum class State { reserved, live, retiring };
    struct Entry { Allocation allocation; State state; size_t pins = 0; };
    DiskConfig config;
    int fd = -1;
    uint64_t capacity = 0, uuid = 0, sequence = 0, next_id = 0;
    size_t snapshot_bytes = 0;
    int active_snapshot = -1;
    bool failed = false;
    mutable std::mutex metadata_mutex;
    std::mutex commit_mutex;
    std::unordered_map<uint64_t, Entry> entries;
    std::unordered_map<std::string, uint64_t> keys;
    std::vector<uint64_t> owners, free;
    std::vector<uint64_t> fifo;

    explicit Impl(const DiskConfig &cfg) : config(cfg), next_id(random_identity()) {
        try { open_disk(); } catch (...) { if (fd >= 0) ::close(fd); throw; }
    }
    ~Impl() { if (fd >= 0) ::close(fd); }
    uint64_t snapshot_offset(int index) const {
        return 2 * selector_bytes + uint64_t(index) * snapshot_bytes;
    }
    void open_disk() {
        if (config.path.empty() || config.unit_bytes < 512 ||
            (config.unit_bytes & (config.unit_bytes - 1)) ||
            config.metadata_bytes < 4 * selector_bytes ||
            config.metadata_bytes % config.unit_bytes || config.metadata_bytes % selector_bytes)
            throw std::invalid_argument("invalid G3 path or alignment geometry");
        snapshot_bytes = ((config.metadata_bytes - 2 * selector_bytes) / 2 / selector_bytes) * selector_bytes;
        fd = ::open(config.path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (config.create ? O_CREAT : 0), 0600);
        if (fd < 0) throw std::runtime_error("cannot open assigned G3 path: " + std::string(std::strerror(errno)));
        if (::flock(fd, LOCK_EX | LOCK_NB)) throw std::runtime_error("G3 path already owned");
        struct stat st{};
        if (::fstat(fd, &st)) throw std::runtime_error("cannot stat G3 path");
        bool regular = S_ISREG(st.st_mode);
        if (!regular && !S_ISBLK(st.st_mode)) throw std::invalid_argument("G3 path must be a file or block device");
        if (regular) {
            if (st.st_size == 0 && config.create) {
                if (config.capacity_bytes <= config.metadata_bytes ||
                    config.capacity_bytes > uint64_t(std::numeric_limits<off_t>::max()) ||
                    ::ftruncate(fd, static_cast<off_t>(config.capacity_bytes)))
                    throw std::invalid_argument("new debug file needs valid explicit capacity");
                capacity = config.capacity_bytes;
            } else capacity = static_cast<uint64_t>(st.st_size);
        } else if (::ioctl(fd, BLKGETSIZE64, &capacity)) throw std::runtime_error("cannot size G3 device");
        if (capacity <= config.metadata_bytes || capacity > uint64_t(std::numeric_limits<off_t>::max()) ||
            (config.capacity_bytes && capacity != config.capacity_bytes))
            throw std::invalid_argument("G3 capacity does not match configuration");
        uint64_t count = (capacity - config.metadata_bytes) / config.unit_bytes;
        if (!count || count > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            throw std::invalid_argument("invalid G3 slot count");
        owners.assign(static_cast<size_t>(count), 0);
        std::array<std::vector<uint8_t>, 2> headers{std::vector<uint8_t>(selector_bytes), std::vector<uint8_t>(selector_bytes)};
        std::array<Selector, 2> selectors;
        for (int i = 0; i != 2; ++i) {
            if (!io_all(fd, headers[i].data(), headers[i].size(), uint64_t(i) * selector_bytes, false))
                throw std::runtime_error("cannot read G3 selectors");
            selectors[i] = decode_selector(headers[i]);
        }
        if (!selectors[0].valid && !selectors[1].valid) {
            if (!config.create || !regular) throw std::runtime_error("uninitialized or corrupt G3 metadata");
            // Initialization must never erase unknown metadata, even if its selector is invalid.
            std::vector<uint8_t> region(config.metadata_bytes);
            if (!io_all(fd, region.data(), region.size(), 0, false) ||
                std::any_of(region.begin(), region.end(), [](uint8_t byte) { return byte != 0; }))
                throw std::runtime_error("refusing to initialize nonempty G3 metadata");
            uuid = random_identity();
            rebuild_free();
            if (commit() != Status::success) throw std::runtime_error("cannot initialize durable G3 metadata");
            return;
        }
        if (selectors[0].valid && selectors[1].valid && selectors[0].uuid != selectors[1].uuid)
            throw std::runtime_error("G3 selector identities disagree");
        active_snapshot = !selectors[1].valid || (selectors[0].valid && selectors[0].seq > selectors[1].seq) ? 0 : 1;
        const auto &chosen = selectors[active_snapshot];
        if (chosen.bytes > snapshot_bytes || chosen.bytes < 9 * 8)
            throw std::runtime_error("invalid G3 snapshot length");
        std::vector<uint8_t> body(static_cast<size_t>(chosen.bytes));
        if (!io_all(fd, body.data(), body.size(), snapshot_offset(active_snapshot), false) ||
            hash(body.data(), body.size()) != chosen.digest)
            throw std::runtime_error("corrupt committed G3 snapshot; no older-state fallback");
        sequence = chosen.seq;
        uuid = chosen.uuid;
        restore(body);
        rebuild_free();
    }
    void rebuild_free() {
        free.clear();
        for (size_t i = owners.size(); i > 0; --i) if (!owners[i - 1]) free.push_back(i - 1);
    }
    void restore(const std::vector<uint8_t> &body) {
        size_t pos = 0;
        if (get(body, pos) != format_magic || get(body, pos) != format_version ||
            get(body, pos) != uuid || get(body, pos) != sequence ||
            get(body, pos) != config.unit_bytes || get(body, pos) != config.metadata_bytes ||
            get(body, pos) != capacity)
            throw std::runtime_error("G3 snapshot identity/geometry mismatch");
        uint64_t saved_next = get(body, pos), count = get(body, pos);
        next_id = std::max(next_id, saved_next);
        if (!saved_next || count > (body.size() - pos) / 32)
            throw std::runtime_error("invalid G3 record count");
        for (uint64_t i = 0; i < count; ++i) {
            Allocation a;
            a.id = get(body, pos);
            uint64_t bytes = get(body, pos), key_size = get(body, pos), slot_count = get(body, pos);
            if (!a.id || a.id >= saved_next || !bytes || bytes > capacity - config.metadata_bytes ||
                bytes > std::numeric_limits<size_t>::max() || !key_size || key_size > max_key_bytes ||
                key_size > body.size() - pos || slot_count != 1 + (bytes - 1) / config.unit_bytes)
                throw std::runtime_error("invalid G3 allocation record");
            a.bytes = static_cast<size_t>(bytes);
            a.key.assign(reinterpret_cast<const char *>(body.data() + pos), static_cast<size_t>(key_size));
            pos += static_cast<size_t>(key_size);
            if (slot_count > (body.size() - pos) / 8) throw std::runtime_error("truncated G3 slot list");
            for (uint64_t j = 0; j < slot_count; ++j) {
                uint64_t slot = get(body, pos);
                if (slot >= owners.size() || owners[slot]) throw std::runtime_error("overlapping/out-of-range G3 slots");
                owners[slot] = a.id;
                a.slots.push_back(slot);
            }
            if (!keys.emplace(a.key, a.id).second || !entries.emplace(a.id, Entry{a, State::live}).second)
                throw std::runtime_error("duplicate G3 identity/key");
            fifo.push_back(a.id);
        }
        if (pos != body.size()) throw std::runtime_error("extra bytes in G3 snapshot");
    }
    size_t metadata_usage() const {
        size_t used = 9 * 8;
        for (const auto &item : entries) {
            const auto &e = item.second;
            if (e.state == State::retiring) continue;
            size_t bytes = 32 + e.allocation.key.size() + e.allocation.slots.size() * 8;
            if (bytes > snapshot_bytes - std::min(used, snapshot_bytes)) return snapshot_bytes;
            used += bytes;
        }
        return used;
    }
    std::vector<uint8_t> snapshot(uint64_t seq) const {
        std::vector<uint8_t> body;
        for (auto value : {format_magic, format_version, uuid, seq, uint64_t(config.unit_bytes),
                           uint64_t(config.metadata_bytes), capacity, next_id}) put(body, value);
        uint64_t count = 0;
        for (const auto &item : entries) if (item.second.state == State::live) ++count;
        put(body, count);
        // FIFO order makes restoration deterministic and preserves admission order.
        for (auto id : fifo) {
            const auto &e = entries.at(id);
            if (e.state != State::live) continue;
            const auto &a = e.allocation;
            put(body, a.id); put(body, a.bytes); put(body, a.key.size()); put(body, a.slots.size());
            body.insert(body.end(), a.key.begin(), a.key.end());
            for (auto slot : a.slots) put(body, slot);
        }
        return body;
    }
    void release(uint64_t id) {
        const auto &a = entries.at(id).allocation;
        for (auto slot : a.slots) { owners[slot] = 0; free.push_back(slot); }
        entries.erase(id);
        fifo.erase(std::remove(fifo.begin(), fifo.end(), id), fifo.end());
    }
    // Caller holds commit_mutex; metadata lock is released for every I/O barrier.
    Status commit() {
        std::vector<uint8_t> body;
        uint64_t next_sequence;
        {
            std::lock_guard<std::mutex> guard(metadata_mutex);
            if (failed || sequence == std::numeric_limits<uint64_t>::max()) return Status::io_error;
            next_sequence = sequence + 1;
            body = snapshot(next_sequence);
        }
        int target = active_snapshot == 0 ? 1 : 0;
        if (body.size() > snapshot_bytes) return Status::no_space;
        // Payload flush precedes any persistent insertion mapping. Snapshot is durable
        // before its independent selector is written; selector flush precedes reuse.
        std::vector<uint8_t> selector;
        for (auto value : {selector_magic, format_version, next_sequence, uint64_t(body.size()),
                           hash(body.data(), body.size()), uuid}) put(selector, value);
        put(selector, hash(selector.data(), selector.size()));
        selector.resize(selector_bytes, 0);
        if (!sync_data(fd) || !io_all(fd, body.data(), body.size(), snapshot_offset(target), true) ||
            !sync_data(fd) || !io_all(fd, selector.data(), selector.size(), uint64_t(target) * selector_bytes, true) ||
            !sync_data(fd)) {
            std::lock_guard<std::mutex> guard(metadata_mutex);
            // An uncertain durability barrier poisons the writer: further overwrites
            // could invalidate the sole recovered snapshot. Reopen for recovery.
            failed = true;
            return Status::io_error;
        }
        {
            std::lock_guard<std::mutex> guard(metadata_mutex);
            sequence = next_sequence;
            active_snapshot = target;
            std::vector<uint64_t> retired;
            for (const auto &item : entries) if (item.second.state == State::retiring) retired.push_back(item.first);
            for (auto id : retired) release(id);
        }
        return Status::success;
    }
};

DiskIndex::DiskIndex(const DiskConfig &config) : impl_(new Impl(config)) {}
DiskIndex::~DiskIndex() = default;
int DiskIndex::fd() const { return impl_->fd; }
size_t DiskIndex::unit_bytes() const { return impl_->config.unit_bytes; }
uint64_t DiskIndex::payload_base() const { return impl_->config.metadata_bytes; }
uint64_t DiskIndex::capacity_bytes() const { return impl_->capacity; }
uint64_t DiskIndex::slot_offset(uint64_t slot) const {
    if (slot >= impl_->owners.size()) throw std::out_of_range("G3 slot");
    return payload_base() + slot * unit_bytes();
}
size_t DiskIndex::free_slots() const {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    return impl_->free.size();
}
Status DiskIndex::reserve(const std::string &key, size_t bytes, Allocation &out) {
    out = Allocation{};
    if (key.empty() || key.size() > max_key_bytes || !bytes) return Status::invalid_input;
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    if (impl_->failed) return Status::io_error;
    auto found = impl_->keys.find(key);
    if (found != impl_->keys.end()) {
        const auto &e = impl_->entries.at(found->second);
        if (e.state != Impl::State::live) return Status::busy;
        if (e.allocation.bytes != bytes) return Status::invalid_input;
        out = e.allocation; out.already_present = true;
        return Status::success;
    }
    uint64_t slots = 1 + (bytes - 1) / unit_bytes();
    if (slots > impl_->free.size()) return Status::no_space;
    size_t used = impl_->metadata_usage();
    size_t available = impl_->snapshot_bytes - used;
    if (available < 32 + key.size() || slots > (available - 32 - key.size()) / 8)
        return Status::no_space;
    if (impl_->next_id == std::numeric_limits<uint64_t>::max()) return Status::no_space;
    Allocation a; a.id = impl_->next_id; a.key = key; a.bytes = bytes;
    a.slots.reserve(static_cast<size_t>(slots));
    for (uint64_t i = 0; i < slots; ++i) {
        a.slots.push_back(impl_->free[impl_->free.size() - 1 - static_cast<size_t>(i)]);
    }
    // Complete fallible allocations before changing the occupancy/free ledger.
    out = a;
    impl_->entries.emplace(a.id, Impl::Entry{a, Impl::State::reserved});
    try { impl_->keys.emplace(key, a.id); }
    catch (...) { impl_->entries.erase(a.id); out = Allocation{}; throw; }
    for (auto slot : a.slots) { impl_->free.pop_back(); impl_->owners[slot] = a.id; }
    ++impl_->next_id;
    return Status::success;
}
Status DiskIndex::publish(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    if (impl_->failed) return Status::io_error;
    auto found = impl_->entries.find(id);
    if (found == impl_->entries.end()) return Status::not_found;
    if (found->second.state != Impl::State::reserved) return Status::invalid_input;
    impl_->fifo.push_back(id);
    found->second.state = Impl::State::live;
    return Status::success;
}
Status DiskIndex::abort(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    auto found = impl_->entries.find(id);
    if (found == impl_->entries.end()) return Status::not_found;
    if (found->second.state != Impl::State::reserved) return Status::invalid_input;
    impl_->keys.erase(found->second.allocation.key);
    impl_->release(id);
    return Status::success;
}
Status DiskIndex::pin(const std::string &key, Allocation &out) {
    out = Allocation{};
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    if (impl_->failed) return Status::io_error;
    auto found = impl_->keys.find(key);
    if (found == impl_->keys.end()) return Status::not_found;
    auto &e = impl_->entries.at(found->second);
    if (e.state != Impl::State::live) return Status::not_found;
    if (e.pins == std::numeric_limits<size_t>::max()) return Status::busy;
    out = e.allocation; ++e.pins;
    return Status::success;
}
Status DiskIndex::unpin(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    auto found = impl_->entries.find(id);
    if (found == impl_->entries.end()) return Status::not_found;
    if (!found->second.pins) return Status::invalid_input;
    --found->second.pins;
    return Status::success;
}
bool DiskIndex::exists(const std::string &key) const {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    auto found = impl_->keys.find(key);
    return !impl_->failed && found != impl_->keys.end() && impl_->entries.at(found->second).state == Impl::State::live;
}
std::vector<std::string> DiskIndex::snapshot_keys() const {
    std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
    std::vector<std::string> result;
    if (impl_->failed) return result;
    result.reserve(impl_->fifo.size());
    for (auto id : impl_->fifo) {
        const auto &entry = impl_->entries.at(id);
        if (entry.state == Impl::State::live) result.push_back(entry.allocation.key);
    }
    return result;
}
Status DiskIndex::evict_one() {
    std::unique_lock<std::mutex> commit_lock(impl_->commit_mutex, std::try_to_lock);
    if (!commit_lock.owns_lock()) return Status::busy;
    {
        std::lock_guard<std::mutex> lock(impl_->metadata_mutex);
        if (impl_->failed) return Status::io_error;
        uint64_t candidate = 0;
        bool busy = false;
        for (auto id : impl_->fifo) {
            auto &e = impl_->entries.at(id);
            if (e.state != Impl::State::live) continue;
            if (e.pins) { busy = true; continue; }
            candidate = id; break;
        }
        if (!candidate) return busy ? Status::busy : Status::not_found;
        auto &entry = impl_->entries.at(candidate);
        entry.state = Impl::State::retiring;
        impl_->keys.erase(entry.allocation.key);
    }
    return impl_->commit();
}
Status DiskIndex::checkpoint() {
    std::unique_lock<std::mutex> lock(impl_->commit_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return Status::busy;
    return impl_->commit();
}
} // namespace nixlshard
