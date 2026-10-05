#pragma once

#include "nixlshard/types.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nixlshard {
struct DiskConfig {
    std::string path;
    uint64_t capacity_bytes = 0;
    size_t unit_bytes = 65536;
    // Legacy source compatibility only: the authoritative format derives this region.
    size_t metadata_bytes = 16 * 1024 * 1024;
    bool create = false;
    size_t min_object_bytes = 0; // zero resolves to unit_bytes
    size_t max_object_bytes = 0; // zero resolves to unit_bytes; facade supplies model bounds
    size_t key_bytes = 32;
    size_t metadata_alignment = 4096;
    std::string namespace_id; // exact canonical model/layout domain, <=2048 bytes
    bool reset = false; // explicit incompatible-format reset of recognized G3 media only
    bool direct_io = false;
};
struct Allocation {
    uint64_t id = 0;
    std::string key;
    std::vector<uint64_t> slots; // temporary operation state, never a LIVE directory cache
    size_t bytes = 0;
    bool already_present = false;
    uint64_t generation = 0;
    uint64_t record = 0;
};
struct AllocationIdentity {
    std::string key;
    uint64_t id = 0, generation = 0, record = 0;
};
enum class CloseMode { clean, discard };
enum class DeviceState { active, draining, offline, failed, closed };
enum class IndexEventKind { committed, evicted, device_retired };
struct IndexEvent { IndexEventKind kind; AllocationIdentity identity; uint64_t sequence = 0; };
// Delivery is serialized in logical transition order, outside index/policy locks.
// Sinks may query this index but must not recursively mutate it.
using IndexEventSink = std::function<void(const IndexEvent &)>;
// Every callback must return only after its underlying I/O is quiescent. The caller
// provides bounded NIXL progress and keeps registrations/backend alive through close.
// Format/header and allocation-record reads/writes use this same callback.
using MetadataIO = std::function<Status(int fd, bool write, uint64_t offset,
                                        void *aligned_buffer, size_t bytes)>;
using PersistenceBarrier = std::function<Status(int fd)>;
// Called before every descriptor close, including failed construction. A failed
// retirement keeps the descriptor open; the integration retains backend ownership.
using DescriptorRetirement = std::function<Status(int fd)>;
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual void on_block_added(const AllocationIdentity &, size_t allocated_slots) noexcept = 0;
    virtual void on_block_read(uint64_t allocation_id) noexcept = 0;
    virtual uint64_t nominate_eviction_candidate(uint64_t generation) noexcept = 0;
    virtual void on_eviction_result(uint64_t allocation_id, Status result) noexcept = 0;
    virtual void on_device_retired(uint64_t generation) noexcept = 0;
};

// Authoritative one-device G3 index. LIVE RAM entries contain identities/claims,
// not payload length or slot lists. Claims precede SSD metadata fetches.
// publish follows terminal successful payload I/O; abort/unpin follow quiescence.
// Destruction leaves DIRTY; only explicit successful clean close permits recovery.
class DiskIndex {
public:
    explicit DiskIndex(const DiskConfig &, MetadataIO,
                       PersistenceBarrier = {}, IndexEventSink = {},
                       std::shared_ptr<EvictionPolicy> = {}, DescriptorRetirement = {});
    ~DiskIndex();
    DiskIndex(const DiskIndex &) = delete;
    DiskIndex &operator=(const DiskIndex &) = delete;
    int fd() const;
    size_t unit_bytes() const;
    uint64_t payload_base() const;
    uint64_t metadata_base() const;
    size_t metadata_record_bytes() const;
    size_t metadata_record_count() const;
    uint64_t generation() const;
    uint64_t capacity_bytes() const;
    uint64_t slot_offset(uint64_t slot) const;
    uint64_t record_offset(uint64_t record) const;
    size_t free_slots() const;
    size_t free_records() const;
    Status reserve(const std::string &key, size_t bytes, Allocation &out);
    Status publish(uint64_t id);
    Status publish(const Allocation &);
    Status abort(uint64_t id);
    Status abort(const Allocation &);
    Status pin(const std::string &key, Allocation &out, uint64_t expected_id = 0);
    Status unpin(uint64_t id);
    Status unpin(const Allocation &);
    bool exists(const std::string &key) const;
    std::vector<std::string> snapshot_keys() const;
    std::vector<AllocationIdentity> enumerate() const;
    Status evict_one();
    Status checkpoint(); // flush DIRTY state; never makes a running cache recoverable
    Status close(CloseMode);
    Status set_state(DeviceState);
    DeviceState state() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nixlshard
