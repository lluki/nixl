#include "nixlshard/storage.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <linux/fs.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace nixlshard {
namespace {
constexpr uint64_t magic = 0x324733445248534eULL;
constexpr uint64_t record_magic = 0x324345524748534eULL;
constexpr uint64_t legacy_selector_magic = 0x31544d435248534eULL;
constexpr uint64_t version = 2;
constexpr uint64_t dirty_marker = 0x4449525459473302ULL;
constexpr uint64_t clean_marker = 0x434c45414e473302ULL;
constexpr size_t fields = 56;

uint64_t digest(const unsigned char *data, size_t bytes) {
    uint64_t value = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) value = (value ^ data[i]) * 1099511628211ULL;
    return value;
}
uint64_t read64(const unsigned char *data, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value |= uint64_t(data[offset + i]) << (8 * i);
    return value;
}
void write64(unsigned char *data, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) data[offset + i] = static_cast<unsigned char>(value >> (8 * i));
}
uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::invalid_argument("G3 geometry overflow");
    return a + b;
}
uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) throw std::invalid_argument("G3 geometry overflow");
    return a * b;
}
uint64_t align_up(uint64_t value, uint64_t alignment) {
    return checked_add(value, alignment - 1) / alignment * alignment;
}
uint64_t random_id() {
    uint64_t value = 0;
    ssize_t result;
    do { result = ::getrandom(&value, sizeof(value), 0); } while (result < 0 && errno == EINTR);
    if (result != sizeof(value)) throw std::runtime_error("cannot obtain G3 generation identity");
    return (value & 0x3fffffffffffffffULL) + 1;
}
Status durable(int fd) {
    int result;
    do { result = ::fdatasync(fd); } while (result && errno == EINTR);
    return result ? Status::io_error : Status::success;
}
struct Buffer {
    unsigned char *data = nullptr;
    size_t bytes;
    Buffer(size_t alignment, size_t length) : bytes(length) {
        void *p = nullptr;
        if (::posix_memalign(&p, std::max(alignment, sizeof(void *)), length)) throw std::bad_alloc();
        data = static_cast<unsigned char *>(p);
        std::memset(data, 0, length);
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer &) = delete;
};
class FifoPolicy final : public EvictionPolicy {
    std::deque<uint64_t> ids;
public:
    void on_block_added(const AllocationIdentity &a, size_t) noexcept override { ids.push_back(a.id); }
    void on_block_read(uint64_t) noexcept override {}
    uint64_t nominate_eviction_candidate(uint64_t) noexcept override { return ids.empty() ? 0 : ids.front(); }
    void on_eviction_result(uint64_t id, Status result) noexcept override {
        auto at = std::find(ids.begin(), ids.end(), id);
        if (at == ids.end()) return;
        ids.erase(at);
        if (result == Status::busy) ids.push_back(id);
    }
    void on_device_retired(uint64_t) noexcept override { ids.clear(); }
};
}

struct DiskIndex::Impl {
    enum class Ownership : uint8_t { free, reserved, live, retiring };
    struct Cell { uint64_t owner = 0; Ownership state = Ownership::free; };
    struct Entry {
        AllocationIdentity identity;
        Ownership state = Ownership::reserved;
        size_t claims = 0;
        bool writing_record = false;
        // Reservation bookkeeping is temporary and discarded at LIVE publication.
        std::optional<Allocation> pending;
    };
    DiskConfig config;
    MetadataIO io;
    MetadataReadBatch read_batch;
    PersistenceBarrier barrier;
    DescriptorRetirement retire_descriptor;
    IndexEventSink sink;
    std::shared_ptr<EvictionPolicy> policy;
    int descriptor = -1;
    uint64_t capacity = 0, device_generation = 0, next_id = 0, payload = 0;
    size_t header_bytes = 0, record_bytes = 0, record_count = 0;
    size_t min_bytes = 0, max_bytes = 0;
    DeviceState device_state = DeviceState::active;
    bool closing = false;
    mutable std::mutex mutex;
    std::mutex mutation_mutex, policy_mutex, event_mutex;
    uint64_t event_sequence = 0;
    std::deque<IndexEvent> pending_events;
    std::unordered_map<uint64_t, Entry> entries;
    std::unordered_map<std::string, uint64_t> directory;
    std::vector<Cell> slots, records;
    std::vector<uint64_t> free_payload, free_metadata;

    Impl(const DiskConfig &cfg, MetadataIO callback, PersistenceBarrier sync,
         IndexEventSink events, std::shared_ptr<EvictionPolicy> eviction, DescriptorRetirement retirement,
         MetadataReadBatch batch)
        : config(cfg), io(std::move(callback)), read_batch(std::move(batch)), barrier(std::move(sync)),
          retire_descriptor(std::move(retirement)), sink(std::move(events)), policy(std::move(eviction)) {
        if (!io) throw std::invalid_argument("G3 requires metadata transfer callback");
        if (!barrier) barrier = durable;
        if (!policy) policy = std::make_shared<FifoPolicy>();
        try { open(); } catch (...) {
            policy->on_device_retired(device_generation);
            close_descriptor(); throw;
        }
    }
    ~Impl() { close_descriptor(); }
    void close_descriptor() noexcept {
        if (descriptor < 0) return;
        if (retire_descriptor) {
            try { if (retire_descriptor(descriptor) != Status::success) return; }
            catch (...) { return; }
        }
        ::close(descriptor); descriptor = -1;
    }
    void queue_event(IndexEventKind kind, const AllocationIdentity &id) {
        if (sink) pending_events.push_back(IndexEvent{kind, id, ++event_sequence});
    }
    void emit() {
        std::lock_guard delivery(event_mutex);
        for (;;) {
            std::optional<IndexEvent> event;
            {
                std::lock_guard guard(mutex);
                if (pending_events.empty()) break;
                event = std::move(pending_events.front()); pending_events.pop_front();
            }
            try { sink(*event); } catch (...) {}
        }
    }
    Status transfer(bool write, uint64_t offset, Buffer &buffer) {
        try { return io(descriptor, write, offset, buffer.data, buffer.bytes); }
        catch (...) { return Status::io_error; }
    }
    Status sync() {
        try { return barrier(descriptor); } catch (...) { return Status::io_error; }
    }
    uint64_t offset(uint64_t record) const { return header_bytes + record * record_bytes; }
    void reset_ledgers() {
        entries.clear(); directory.clear();
        slots.assign(slots.size(), Cell{}); records.assign(record_count, Cell{});
        free_payload.clear(); free_metadata.clear();
        for (size_t i = slots.size(); i; --i) free_payload.push_back(i - 1);
        for (size_t i = record_count; i; --i) free_metadata.push_back(i - 1);
    }
    void geometry() {
        if (config.path.empty() || config.unit_bytes < 512 || (config.unit_bytes & (config.unit_bytes - 1)) ||
            config.metadata_alignment < 512 || (config.metadata_alignment & (config.metadata_alignment - 1)) ||
            !config.key_bytes || config.key_bytes > 32 || config.namespace_id.size() > 2048 ||
            (config.direct_io && (config.metadata_alignment < 4096 || config.unit_bytes % 4096)))
            throw std::invalid_argument("invalid G3 format geometry");
        min_bytes = config.min_object_bytes ? config.min_object_bytes : config.unit_bytes;
        max_bytes = config.max_object_bytes ? config.max_object_bytes : config.unit_bytes;
        if (!min_bytes || max_bytes < min_bytes) throw std::invalid_argument("invalid G3 object bounds");
        uint64_t max_slots = 1 + (uint64_t(max_bytes) - 1) / config.unit_bytes;
        uint64_t length = align_up(checked_add(fields + config.key_bytes + 8,
                                             checked_mul(max_slots, 8)), config.metadata_alignment);
        if (length > 64 * 1024 * 1024 || length > std::numeric_limits<size_t>::max())
            throw std::invalid_argument("G3 metadata record exceeds bounded operation size");
        record_bytes = static_cast<size_t>(length);
        header_bytes = std::max<size_t>(4096, config.metadata_alignment);
        uint64_t minimum_payload = align_up(min_bytes, config.unit_bytes);
        if (capacity <= header_bytes) throw std::invalid_argument("G3 device too small");
        uint64_t count = (capacity - header_bytes) / checked_add(minimum_payload, record_bytes);
        while (count) {
            payload = align_up(checked_add(header_bytes, checked_mul(count, record_bytes)), config.unit_bytes);
            if (payload <= capacity && count <= (capacity - payload) / minimum_payload) break;
            --count;
        }
        if (!count || count > std::numeric_limits<size_t>::max() / sizeof(Cell))
            throw std::invalid_argument("G3 device cannot fit metadata and minimum payload");
        record_count = static_cast<size_t>(count);
        uint64_t payload_count = (capacity - payload) / config.unit_bytes;
        if (payload_count > std::numeric_limits<size_t>::max() / sizeof(Cell))
            throw std::invalid_argument("G3 payload ledger too large");
        slots.resize(static_cast<size_t>(payload_count));
        reset_ledgers();
    }
    bool header_valid(const Buffer &header) const {
        return read64(header.data, 0) == magic && read64(header.data, 8) == version &&
               read64(header.data, header.bytes - 8) == digest(header.data, header.bytes - 8);
    }
    bool matches(const Buffer &header) const {
        const uint64_t values[] = {uint64_t(header_bytes), capacity, uint64_t(config.unit_bytes),
            uint64_t(min_bytes), uint64_t(max_bytes), uint64_t(config.key_bytes),
            uint64_t(config.metadata_alignment), uint64_t(record_bytes), uint64_t(record_count), payload};
        for (size_t i = 0; i < 10; ++i) if (read64(header.data, 16 + i * 8) != values[i]) return false;
        uint64_t length = read64(header.data, 128);
        return length == config.namespace_id.size() &&
               !std::memcmp(header.data + 136, config.namespace_id.data(), static_cast<size_t>(length));
    }
    Status write_header(bool clean) {
        Buffer header(config.metadata_alignment, header_bytes);
        const uint64_t values[] = {magic, version, uint64_t(header_bytes), capacity,
            uint64_t(config.unit_bytes), uint64_t(min_bytes), uint64_t(max_bytes),
            uint64_t(config.key_bytes), uint64_t(config.metadata_alignment), uint64_t(record_bytes),
            uint64_t(record_count), payload, device_generation, next_id,
            clean ? clean_marker : dirty_marker, uint64_t(config.namespace_id.size())};
        // Namespace length is at 128; marker at 112.
        for (size_t i = 0; i < 15; ++i) write64(header.data, i * 8, values[i]);
        write64(header.data, 128, config.namespace_id.size());
        std::memcpy(header.data + 136, config.namespace_id.data(), config.namespace_id.size());
        write64(header.data, header.bytes - 8, digest(header.data, header.bytes - 8));
        return transfer(true, 0, header);
    }
    Status zero_records() {
        size_t chunk = std::max<size_t>(config.metadata_alignment,
            (1024 * 1024 / config.metadata_alignment) * config.metadata_alignment);
        uint64_t total = uint64_t(record_count) * record_bytes, done = 0;
        while (done < total) {
            Buffer zeros(config.metadata_alignment, static_cast<size_t>(std::min<uint64_t>(chunk, total - done)));
            auto result = transfer(true, header_bytes + done, zeros);
            if (result != Status::success) return result;
            done += zeros.bytes;
        }
        return Status::success;
    }
    Status decode_record(uint64_t record, const unsigned char *data, size_t record_size,
                         Allocation &out, bool restoring = false) {
        if (std::all_of(data, data + record_size, [](unsigned char c) { return c == 0; })) return Status::not_found;
        if (read64(data, 0) != record_magic || read64(data, 8) != version ||
            read64(data, 16) != 1 || read64(data, record_size - 8) != digest(data, record_size - 8))
            return Status::io_error;
        uint64_t id = read64(data, 24), bytes = read64(data, 32), key_size = read64(data, 40);
        uint64_t count = read64(data, 48);
        if (!id || (restoring && id >= next_id) || bytes < min_bytes || bytes > max_bytes ||
            !key_size || key_size > config.key_bytes || count != 1 + (bytes - 1) / config.unit_bytes ||
            count > (record_size - fields - config.key_bytes - 8) / 8) return Status::io_error;
        Allocation allocation; allocation.id = id; allocation.generation = device_generation;
        allocation.record = record; allocation.bytes = static_cast<size_t>(bytes);
        allocation.key.assign(reinterpret_cast<const char *>(data + fields), static_cast<size_t>(key_size));
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t slot = read64(data, fields + config.key_bytes + static_cast<size_t>(i) * 8);
            if (slot >= slots.size() || std::find(allocation.slots.begin(), allocation.slots.end(), slot) != allocation.slots.end())
                return Status::io_error;
            allocation.slots.push_back(slot);
        }
        out = std::move(allocation); return Status::success;
    }
    Status decode(uint64_t record, const Buffer &buffer, Allocation &out, bool restoring = false) {
        return decode_record(record, buffer.data, buffer.bytes, out, restoring);
    }
    Status read_allocation(const AllocationIdentity &identity, Allocation &out) {
        Buffer data(config.metadata_alignment, record_bytes);
        auto result = transfer(false, offset(identity.record), data);
        if (result != Status::success) return result;
        result = decode(identity.record, data, out);
        if (result != Status::success) return Status::io_error;
        if (out.id != identity.id || out.key != identity.key) return Status::io_error;
        std::lock_guard lock(mutex);
        for (auto slot : out.slots) if (slots[slot].owner != identity.id ||
            (slots[slot].state != Ownership::live && slots[slot].state != Ownership::retiring))
            return Status::io_error;
        return Status::success;
    }
    Status write_allocation(const Allocation &a) {
        Buffer data(config.metadata_alignment, record_bytes);
        uint64_t values[] = {record_magic, version, 1, a.id, uint64_t(a.bytes),
                             uint64_t(a.key.size()), uint64_t(a.slots.size())};
        for (size_t i = 0; i < 7; ++i) write64(data.data, i * 8, values[i]);
        std::memcpy(data.data + fields, a.key.data(), a.key.size());
        for (size_t i = 0; i < a.slots.size(); ++i)
            write64(data.data, fields + config.key_bytes + i * 8, a.slots[i]);
        write64(data.data, data.bytes - 8, digest(data.data, data.bytes - 8));
        return transfer(true, offset(a.record), data);
    }
    bool restore() {
        for (size_t record = 0; record < record_count; ++record) {
            Buffer data(config.metadata_alignment, record_bytes);
            if (transfer(false, offset(record), data) != Status::success) return false;
            Allocation a;
            auto status = decode(record, data, a, true);
            if (status == Status::not_found) continue;
            if (status != Status::success || directory.count(a.key) || entries.count(a.id)) return false;
            for (auto slot : a.slots) if (slots[slot].owner) return false;
            AllocationIdentity identity{a.key, a.id, device_generation, record};
            entries.emplace(a.id, Entry{identity, Ownership::live, 0, false, {}});
            directory.emplace(a.key, a.id);
            records[record] = Cell{a.id, Ownership::live};
            for (auto slot : a.slots) slots[slot] = Cell{a.id, Ownership::live};
            // Recovery policy membership is rebuilt from the validated SSD set.
            policy->on_block_added(identity, a.slots.size());
        }
        free_payload.clear(); free_metadata.clear();
        for (size_t i = slots.size(); i; --i) if (!slots[i - 1].owner) free_payload.push_back(i - 1);
        for (size_t i = records.size(); i; --i) if (!records[i - 1].owner) free_metadata.push_back(i - 1);
        return true;
    }
    void open() {
        if (config.path.empty()) throw std::invalid_argument("empty G3 path");
        descriptor = ::open(config.path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW |
                            (config.create ? O_CREAT : 0) | (config.direct_io ? O_DIRECT : 0), 0600);
        if (descriptor < 0) throw std::runtime_error("cannot open assigned G3 path: " + std::string(std::strerror(errno)));
        if (::flock(descriptor, LOCK_EX | LOCK_NB)) throw std::runtime_error("G3 path already owned");
        struct stat st{};
        if (::fstat(descriptor, &st)) throw std::runtime_error("cannot stat G3 path");
        bool regular = S_ISREG(st.st_mode), initialized_file = false;
        if (!regular && !S_ISBLK(st.st_mode)) throw std::invalid_argument("G3 needs file or block device");
        if (regular) {
            if (!st.st_size && config.create) {
                if (!config.capacity_bytes || config.capacity_bytes > uint64_t(std::numeric_limits<off_t>::max()) ||
                    ::ftruncate(descriptor, static_cast<off_t>(config.capacity_bytes)))
                    throw std::invalid_argument("new G3 debug file needs explicit valid capacity");
                capacity = config.capacity_bytes; initialized_file = true;
            } else capacity = static_cast<uint64_t>(st.st_size);
        } else if (::ioctl(descriptor, BLKGETSIZE64, &capacity)) throw std::runtime_error("cannot size G3 device");
        if (!capacity || capacity > uint64_t(std::numeric_limits<off_t>::max()) ||
            (config.capacity_bytes && capacity != config.capacity_bytes)) throw std::invalid_argument("G3 capacity mismatch");
        geometry();
        Buffer header(config.metadata_alignment, header_bytes);
        if (transfer(false, 0, header) != Status::success) throw std::runtime_error("cannot read G3 header");
        bool legacy = read64(header.data, 0) == legacy_selector_magic;
        bool recognized = read64(header.data, 0) == magic || legacy;
        bool valid = header_valid(header), recovered = false;
        if (!recognized && !initialized_file) throw std::runtime_error("refusing to format unknown media");
        if (legacy && !config.reset) throw std::invalid_argument("legacy G3 media requires explicit reset");
        if (valid && !matches(header) && !config.reset)
            throw std::invalid_argument("G3 namespace/geometry mismatch needs explicit reset");
        device_generation = random_id();
        next_id = valid && matches(header) ? read64(header.data, 104) : random_id();
        if (!next_id) next_id = random_id();
        if (valid && matches(header) && !config.reset && read64(header.data, 112) == clean_marker)
            recovered = restore();
        if (!recovered) {
            policy->on_device_retired(device_generation);
            reset_ledgers();
            next_id = random_id();
        }
        // Never serve or modify allocation records until DIRTY is durably established.
        if (write_header(false) != Status::success || sync() != Status::success)
            throw std::runtime_error("cannot persist G3 DIRTY marker");
        if (!recovered && zero_records() != Status::success)
            throw std::runtime_error("cannot reset recognized dirty G3 metadata");
    }
    void fail() {
        {
            std::lock_guard policy_guard(policy_mutex);
            std::lock_guard guard(mutex);
            if (device_state == DeviceState::failed || device_state == DeviceState::closed) return;
            device_state = DeviceState::failed;
            for (const auto &[id, entry] : entries) {
                (void)id;
                if (entry.state == Ownership::live) queue_event(IndexEventKind::device_retired, entry.identity);
            }
            directory.clear();
            policy->on_device_retired(device_generation);
        }
        emit();
    }
    void release(const Allocation &a) {
        for (auto slot : a.slots) { slots[slot] = Cell{}; free_payload.push_back(slot); }
        records[a.record] = Cell{}; free_metadata.push_back(a.record);
        entries.erase(a.id);
    }
};

DiskIndex::DiskIndex(const DiskConfig &c, MetadataIO io, PersistenceBarrier barrier,
                     IndexEventSink sink, std::shared_ptr<EvictionPolicy> policy,
                     DescriptorRetirement retirement, MetadataReadBatch batch)
    : impl_(new Impl(c, std::move(io), std::move(barrier), std::move(sink), std::move(policy), std::move(retirement), std::move(batch))) {}
DiskIndex::~DiskIndex() = default;
int DiskIndex::fd() const { return impl_->descriptor; }
size_t DiskIndex::unit_bytes() const { return impl_->config.unit_bytes; }
uint64_t DiskIndex::payload_base() const { return impl_->payload; }
uint64_t DiskIndex::metadata_base() const { return impl_->header_bytes; }
size_t DiskIndex::metadata_record_bytes() const { return impl_->record_bytes; }
size_t DiskIndex::metadata_record_count() const { return impl_->record_count; }
uint64_t DiskIndex::generation() const { return impl_->device_generation; }
uint64_t DiskIndex::capacity_bytes() const { return impl_->capacity; }
uint64_t DiskIndex::slot_offset(uint64_t slot) const {
    if (slot >= impl_->slots.size()) throw std::out_of_range("G3 slot");
    return payload_base() + slot * unit_bytes();
}
uint64_t DiskIndex::record_offset(uint64_t record) const {
    if (record >= impl_->record_count) throw std::out_of_range("G3 metadata record");
    return impl_->offset(record);
}
size_t DiskIndex::free_slots() const { std::lock_guard guard(impl_->mutex); return impl_->free_payload.size(); }
size_t DiskIndex::free_records() const { std::lock_guard guard(impl_->mutex); return impl_->free_metadata.size(); }
DeviceState DiskIndex::state() const { std::lock_guard guard(impl_->mutex); return impl_->device_state; }

Status DiskIndex::reserve(const std::string &key, size_t bytes, Allocation &out) {
    out = {};
    if (key.empty() || key.size() > impl_->config.key_bytes || bytes < impl_->min_bytes || bytes > impl_->max_bytes)
        return Status::invalid_input;
    std::unique_lock guard(impl_->mutex);
    if (impl_->device_state == DeviceState::failed) return Status::io_error;
    if (impl_->device_state != DeviceState::active || impl_->closing) return Status::busy;
    auto found = impl_->directory.find(key);
    if (found != impl_->directory.end()) {
        auto &entry = impl_->entries.at(found->second);
        if (entry.state != Impl::Ownership::live) return Status::busy;
        guard.unlock();
        auto result = pin(key, out);
        if (result != Status::success) return result;
        unpin(out);
        if (out.bytes != bytes) { out = {}; return Status::invalid_input; }
        out.already_present = true; return Status::success;
    }
    uint64_t count = 1 + (bytes - 1) / unit_bytes();
    if (count > impl_->free_payload.size() || impl_->free_metadata.empty()) return Status::no_space;
    if (impl_->next_id == std::numeric_limits<uint64_t>::max()) return Status::no_space;
    Allocation a; a.id = impl_->next_id; a.key = key; a.bytes = bytes;
    a.generation = impl_->device_generation; a.record = impl_->free_metadata.back();
    for (size_t i = 0; i < count; ++i) a.slots.push_back(impl_->free_payload[impl_->free_payload.size() - 1 - i]);
    Impl::Entry entry{{key, a.id, a.generation, a.record}, Impl::Ownership::reserved, 0, false, a};
    out = a; impl_->entries.emplace(a.id, std::move(entry));
    try { impl_->directory.emplace(key, a.id); } catch (...) { impl_->entries.erase(a.id); out = {}; throw; }
    impl_->free_metadata.pop_back(); impl_->records[a.record] = {a.id, Impl::Ownership::reserved};
    for (auto slot : a.slots) { impl_->free_payload.pop_back(); impl_->slots[slot] = {a.id, Impl::Ownership::reserved}; }
    ++impl_->next_id; return Status::success;
}
Status DiskIndex::publish(const Allocation &a) {
    if (a.generation != generation()) return Status::not_found;
    return publish(a.id);
}
Status DiskIndex::publish(uint64_t id) {
    Allocation a;
    {
        std::lock_guard guard(impl_->mutex);
        if (impl_->device_state == DeviceState::failed) return Status::io_error;
        if (impl_->device_state == DeviceState::closed || impl_->device_state == DeviceState::offline) return Status::not_ready;
        auto at = impl_->entries.find(id);
        if (at == impl_->entries.end()) return Status::not_found;
        if (at->second.state != Impl::Ownership::reserved) return Status::invalid_input;
        if (at->second.writing_record) return Status::busy;
        a = *at->second.pending; at->second.writing_record = true;
    }
    auto result = impl_->write_allocation(a);
    if (result != Status::success) {
        { std::lock_guard guard(impl_->mutex); impl_->entries.at(id).writing_record = false; }
        impl_->fail(); return result;
    }
    AllocationIdentity identity;
    {
        std::lock_guard policy_guard(impl_->policy_mutex);
        std::lock_guard guard(impl_->mutex);
        auto &entry = impl_->entries.at(id); entry.writing_record = false;
        if (impl_->device_state == DeviceState::failed) return Status::io_error;
        identity = entry.identity;
        impl_->policy->on_block_added(identity, a.slots.size());
        entry.pending.reset(); entry.state = Impl::Ownership::live;
        impl_->records[a.record].state = Impl::Ownership::live;
        for (auto slot : a.slots) impl_->slots[slot].state = Impl::Ownership::live;
        impl_->queue_event(IndexEventKind::committed, identity);
    }
    impl_->emit();
    return Status::success;
}
Status DiskIndex::abort(const Allocation &a) {
    if (a.generation != generation()) return Status::not_found;
    return abort(a.id);
}
Status DiskIndex::abort(uint64_t id) {
    std::lock_guard guard(impl_->mutex);
    auto at = impl_->entries.find(id);
    if (at == impl_->entries.end()) return Status::not_found;
    if (at->second.state != Impl::Ownership::reserved) return Status::invalid_input;
    if (at->second.writing_record) return Status::busy;
    auto a = *at->second.pending; impl_->directory.erase(a.key); impl_->release(a);
    return Status::success;
}
Status DiskIndex::pin(const std::string &key, Allocation &out, uint64_t expected_id) {
    out = {}; AllocationIdentity identity;
    {
        std::lock_guard policy_guard(impl_->policy_mutex);
        std::lock_guard guard(impl_->mutex);
        if (impl_->device_state == DeviceState::failed) return Status::io_error;
        if (impl_->closing || impl_->device_state == DeviceState::offline || impl_->device_state == DeviceState::closed)
            return Status::not_ready;
        auto found = impl_->directory.find(key);
        if (found == impl_->directory.end()) return Status::not_found;
        auto &entry = impl_->entries.at(found->second);
        if (entry.state != Impl::Ownership::live || (expected_id && expected_id != entry.identity.id)) return Status::not_found;
        if (entry.claims == std::numeric_limits<size_t>::max()) return Status::busy;
        ++entry.claims; identity = entry.identity; impl_->policy->on_block_read(identity.id);
    }
    auto result = impl_->read_allocation(identity, out);
    if (result != Status::success) {
        unpin(identity.id); out = {}; impl_->fail(); return result;
    }
    return Status::success;
}
std::vector<PinResult> DiskIndex::pin_many(const std::vector<PinQuery> &queries) {
    if (queries.size() > 8) throw std::invalid_argument("G3 metadata batch exceeds eight records");
    std::vector<PinResult> results(queries.size());
    std::vector<AllocationIdentity> identities(queries.size());
    std::vector<size_t> claimed; claimed.reserve(queries.size());
    auto release_claims = [&] {
        for (auto n : claimed) unpin(identities[n].id);
        claimed.clear();
    };
    try {
        {
            std::lock_guard policy_guard(impl_->policy_mutex);
            std::lock_guard guard(impl_->mutex);
            for (size_t n = 0; n < queries.size(); ++n) {
                auto &result = results[n]; const auto &query = queries[n];
                if (impl_->device_state == DeviceState::failed) { result.status = Status::io_error; continue; }
                if (impl_->closing || impl_->device_state == DeviceState::offline || impl_->device_state == DeviceState::closed) {
                    result.status = Status::not_ready; continue;
                }
                auto found = impl_->directory.find(query.key);
                if (found == impl_->directory.end()) continue;
                auto &entry = impl_->entries.at(found->second);
                if (entry.state != Impl::Ownership::live || (query.expected_id && query.expected_id != entry.identity.id)) continue;
                if (entry.claims == std::numeric_limits<size_t>::max()) { result.status = Status::busy; continue; }
                identities[n] = entry.identity; // potentially throwing string copy precedes claim increment
                claimed.push_back(n); ++entry.claims;
                impl_->policy->on_block_read(entry.identity.id);
                result.status = Status::success;
            }
        }
        // Bound scratch to 128 MiB. Unusually large records use smaller chunks;
        // all identities remain claimed until the complete operation is quiescent.
        const size_t per_chunk = std::min<size_t>(8, std::max<size_t>(1, (128 * 1024 * 1024) / impl_->record_bytes));
        for (size_t begin = 0; begin < claimed.size(); begin += per_chunk) {
            const size_t count = std::min(per_chunk, claimed.size() - begin);
            Buffer data(impl_->config.metadata_alignment, count * impl_->record_bytes);
            std::vector<MetadataRead> ranges; ranges.reserve(count);
            for (size_t k = 0; k < count; ++k) {
                const auto n = claimed[begin + k];
                ranges.push_back({impl_->offset(identities[n].record), data.data + k * impl_->record_bytes, impl_->record_bytes});
            }
            Status status = Status::success;
            if (impl_->read_batch) {
                try { status = impl_->read_batch(fd(), ranges); } catch (...) { status = Status::io_error; }
            } else {
                for (const auto &range : ranges) {
                    try { status = impl_->io(fd(), false, range.offset, range.buffer, range.bytes); }
                    catch (...) { status = Status::io_error; }
                    if (status != Status::success) break;
                }
            }
            if (status != Status::success) {
                for (auto n : claimed) { results[n].status = status; results[n].allocation = {}; }
                release_claims(); impl_->fail(); return results;
            }
            for (size_t k = 0; k < count; ++k) {
                const auto n = claimed[begin + k]; const auto &id = identities[n];
                auto &allocation = results[n].allocation;
                status = impl_->decode_record(id.record, data.data + k * impl_->record_bytes, impl_->record_bytes, allocation);
                if (status == Status::success && (allocation.id != id.id || allocation.key != id.key)) status = Status::io_error;
                if (status == Status::success) {
                    std::lock_guard guard(impl_->mutex);
                    for (auto slot : allocation.slots)
                        if (impl_->slots[slot].owner != id.id || impl_->slots[slot].state != Impl::Ownership::live) status = Status::io_error;
                }
                if (status != Status::success) {
                    for (auto m : claimed) { results[m].status = Status::io_error; results[m].allocation = {}; }
                    release_claims(); impl_->fail(); return results;
                }
            }
        }
        claimed.clear(); // successful results now own the claims through payload completion
        return results;
    } catch (...) {
        release_claims(); throw;
    }
}
Status DiskIndex::unpin(const Allocation &a) {
    if (a.generation != generation()) return Status::not_found;
    return unpin(a.id);
}
Status DiskIndex::unpin(uint64_t id) {
    std::lock_guard guard(impl_->mutex);
    auto at = impl_->entries.find(id);
    if (at == impl_->entries.end()) return Status::not_found;
    if (!at->second.claims) return Status::invalid_input;
    --at->second.claims; return Status::success;
}
bool DiskIndex::exists(const std::string &key) const {
    std::lock_guard guard(impl_->mutex);
    auto at = impl_->directory.find(key);
    return impl_->device_state != DeviceState::failed && impl_->device_state != DeviceState::closed &&
           at != impl_->directory.end() && impl_->entries.at(at->second).state == Impl::Ownership::live;
}
std::vector<AllocationIdentity> DiskIndex::enumerate() const {
    std::lock_guard guard(impl_->mutex);
    std::vector<AllocationIdentity> out;
    if (impl_->device_state == DeviceState::failed || impl_->device_state == DeviceState::closed) return out;
    for (const auto &[id, entry] : impl_->entries) {
        (void)id;
        if (entry.state == Impl::Ownership::live) out.push_back(entry.identity);
    }
    std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
    return out;
}
std::vector<std::string> DiskIndex::snapshot_keys() const {
    std::vector<std::string> out;
    for (const auto &entry : enumerate()) out.push_back(entry.key);
    return out;
}
Status DiskIndex::evict_one() {
    std::unique_lock mutation(impl_->mutation_mutex, std::try_to_lock);
    if (!mutation.owns_lock()) return Status::busy;
    AllocationIdentity candidate; bool busy = false;
    {
        std::lock_guard policy_guard(impl_->policy_mutex);
        std::lock_guard guard(impl_->mutex);
        if (impl_->device_state == DeviceState::failed) return Status::io_error;
        if (impl_->closing || impl_->device_state == DeviceState::closed || impl_->device_state == DeviceState::offline) return Status::not_ready;
        size_t budget = impl_->entries.size();
        while (budget--) {
            uint64_t id = impl_->policy->nominate_eviction_candidate(generation());
            if (!id) break;
            auto at = impl_->entries.find(id);
            if (at == impl_->entries.end() || at->second.state != Impl::Ownership::live) {
                impl_->policy->on_eviction_result(id, Status::not_found); continue;
            }
            if (at->second.claims) { busy = true; impl_->policy->on_eviction_result(id, Status::busy); continue; }
            auto &entry = at->second; candidate = entry.identity;
            entry.state = Impl::Ownership::retiring; entry.claims = 1;
            impl_->directory.erase(candidate.key);
            impl_->records[candidate.record].state = Impl::Ownership::retiring;
            impl_->queue_event(IndexEventKind::evicted, candidate);
            break;
        }
    }
    if (!candidate.id) return busy ? Status::busy : Status::not_found;
    Allocation a;
    auto result = impl_->read_allocation(candidate, a);
    if (result == Status::success) {
        { std::lock_guard guard(impl_->mutex); for (auto slot : a.slots) impl_->slots[slot].state = Impl::Ownership::retiring; }
        Buffer invalid(impl_->config.metadata_alignment, impl_->record_bytes);
        result = impl_->transfer(true, record_offset(candidate.record), invalid);
    }
    {
        std::lock_guard policy_guard(impl_->policy_mutex);
        std::lock_guard guard(impl_->mutex);
        impl_->policy->on_eviction_result(candidate.id, result);
        if (result == Status::success) impl_->release(a);
        else impl_->entries.at(candidate.id).claims = 0;
    }
    impl_->emit();
    if (result != Status::success) impl_->fail();
    return result;
}
Status DiskIndex::checkpoint() {
    std::unique_lock mutation(impl_->mutation_mutex, std::try_to_lock);
    if (!mutation.owns_lock()) return Status::busy;
    { std::lock_guard guard(impl_->mutex);
      if (impl_->device_state == DeviceState::failed) return Status::io_error;
      if (impl_->device_state == DeviceState::closed) return Status::not_ready; }
    auto result = impl_->sync(); if (result != Status::success) impl_->fail();
    return result;
}
Status DiskIndex::set_state(DeviceState state) {
    if (state == DeviceState::failed) { impl_->fail(); return Status::success; }
    if (state == DeviceState::closed) return Status::invalid_input;
    std::lock_guard guard(impl_->mutex);
    if (impl_->device_state == DeviceState::failed || impl_->device_state == DeviceState::closed) return Status::not_ready;
    impl_->device_state = state; return Status::success;
}
Status DiskIndex::close(CloseMode mode) {
    std::unique_lock mutation(impl_->mutation_mutex, std::try_to_lock);
    if (!mutation.owns_lock()) return Status::busy;
    {
        std::lock_guard guard(impl_->mutex);
        if (impl_->device_state == DeviceState::closed) return Status::success;
        impl_->closing = true;
        if (impl_->device_state != DeviceState::failed) impl_->device_state = DeviceState::draining;
        for (const auto &[id, entry] : impl_->entries) {
            (void)id;
            if (entry.claims || entry.state == Impl::Ownership::reserved || entry.writing_record) return Status::busy;
        }
        if (mode == CloseMode::clean && impl_->device_state == DeviceState::failed) return Status::io_error;
    }
    if (mode == CloseMode::clean) {
        // With admissions stopped and no claims/reservations, finalize every FREE
        // record. Bound each contiguous FREE run to 1 MiB, or one record when
        // the configured record itself is larger. Never cross a LIVE record.
        // Payload/records are flushed before the clean marker is written.
        const size_t max_records = std::max<size_t>(1, (1024 * 1024) / impl_->record_bytes);
        for (size_t i = 0; i < impl_->record_count;) {
            if (impl_->records[i].owner) { ++i; continue; }
            size_t count = 1;
            while (count < max_records && i + count < impl_->record_count &&
                   !impl_->records[i + count].owner) ++count;
            Buffer invalid(impl_->config.metadata_alignment, count * impl_->record_bytes);
            if (impl_->transfer(true, record_offset(i), invalid) != Status::success) {
                impl_->fail(); return Status::io_error;
            }
            i += count;
        }
        if (impl_->sync() != Status::success || impl_->write_header(true) != Status::success || impl_->sync() != Status::success) {
            impl_->fail(); return Status::io_error;
        }
    }
    { std::lock_guard guard(impl_->mutex); impl_->device_state = DeviceState::closed; }
    return Status::success;
}
} // namespace nixlshard
