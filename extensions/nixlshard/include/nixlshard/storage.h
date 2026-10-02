#pragma once

#include "nixlshard/types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nixlshard {
struct DiskConfig {
    std::string path;
    uint64_t capacity_bytes = 0;
    size_t unit_bytes = 65536;
    size_t metadata_bytes = 16 * 1024 * 1024;
    bool create = false;
};

struct Allocation {
    uint64_t id = 0;
    std::string key;
    std::vector<uint64_t> slots;
    size_t bytes = 0;
    bool already_present = false;
};

// One immutable-object ledger for one caller-assigned disk/debug file.
// publish follows completed payload writes; abort/unpin follow I/O quiescence.
// create explicitly initializes regular files only; existing devices are never formatted.
class DiskIndex {
public:
    explicit DiskIndex(const DiskConfig &config);
    ~DiskIndex();
    DiskIndex(const DiskIndex &) = delete;
    DiskIndex &operator=(const DiskIndex &) = delete;
    int fd() const;
    size_t unit_bytes() const;
    uint64_t payload_base() const;
    uint64_t capacity_bytes() const;
    uint64_t slot_offset(uint64_t slot) const;
    size_t free_slots() const;
    Status reserve(const std::string &key, size_t bytes, Allocation &out);
    Status publish(uint64_t id);
    Status abort(uint64_t id);
    Status pin(const std::string &key, Allocation &out);
    Status unpin(uint64_t id);
    bool exists(const std::string &key) const;
    std::vector<std::string> snapshot_keys() const;
    Status evict_one();
    Status checkpoint();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nixlshard
