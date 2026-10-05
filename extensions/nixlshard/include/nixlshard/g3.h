/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "nixlshard/storage.h"
#include "nixl.h"
#include <chrono>
#include <memory>
#include <future>

namespace nixlshard {
enum class MemoryMode { automatic, explicit_registration };
using G3MemoryHandle = uint64_t;
using G3Deadline = std::chrono::steady_clock::time_point;
struct G3Buffer {
    uintptr_t address = 0;
    size_t bytes = 0;
    nixl_mem_t type = DRAM_SEG;
    uint64_t device = 0;
    G3MemoryHandle registration = 0; // zero: resolve by registered range, or AUTOMATIC
};
struct G3DeviceConfig {
    DiskConfig disk;
    int numa_node = -1; // discover assigned block-device topology; debug files require override
};
struct G3Config {
    std::string instance_id;
    std::vector<G3DeviceConfig> devices;
    MemoryMode memory_mode = MemoryMode::explicit_registration;
    size_t max_active = 64;
    size_t staging_bytes = 128 * 1024 * 1024;
    uint64_t timeout_ms = 5000;
    IndexEventSink events;
};
// Borrowed SDK context. It and its selected backends outlive this layer, its
// registrations, and close(). Backend terminal/release must establish quiescence.
struct G3Context {
    nixlAgent *agent = nullptr;
    std::string agent_name;
    nixl_opt_args_t file_options;
    nixl_opt_args_t memory_options;
};
struct G3Metrics {
    // Byte counters count successful transfers/copies; time includes failed attempts.
    uint64_t metadata_ns = 0, metadata_bytes = 0;
    uint64_t metadata_read_ns = 0, metadata_read_bytes = 0;
    uint64_t metadata_write_ns = 0, metadata_write_bytes = 0;
    uint64_t payload_ns = 0, payload_bytes = 0, copy_ns = 0, copy_bytes = 0;
};
struct G3Result {
    Status status = Status::invalid_input;
    AllocationIdentity identity;
    size_t bytes = 0;
    size_t device_index = 0;
    bool direct = false;
    G3Metrics metrics;
};
struct G3Read {
    std::string key;
    std::vector<G3Buffer> buffers;
    uint64_t expected_allocation_id = 0;
};
struct G3BatchResult {
    std::vector<G3Result> objects;
    G3Metrics metrics; // aggregate payload time counted once for the shared NIXL request
};
struct G3DeviceInfo {
    size_t index = 0;
    std::string path;
    int numa_node = -1;
    DeviceState state = DeviceState::failed;
    uint64_t generation = 0;
    std::string diagnostic;
};
struct G3Entry { size_t device_index; AllocationIdentity identity; };
struct G3CloseDevice { size_t device_index; Status status; };
struct G3CloseResult { std::vector<G3CloseDevice> devices; bool clean = false; };

// Synchronous, bounded operation core: the distributed facade supplies its worker
// queue. No thread per operation. Claims and registration leases remain alive
// until all submitted NIXL I/O is quiescent, including logical deadline expiry.
class G3TransferLayer {
public:
    G3TransferLayer(G3Config, G3Context);
    ~G3TransferLayer(); // DISCARD; cannot promise clean persistence
    G3TransferLayer(const G3TransferLayer &) = delete;
    G3TransferLayer &operator=(const G3TransferLayer &) = delete;
    G3MemoryHandle register_memory(const std::vector<G3Buffer> &);
    G3MemoryHandle borrow_registered_memory(const std::vector<G3Buffer> &);
    Status deregister_memory(G3MemoryHandle); // retire new use, wait active references
    G3Result write(const std::string &, const std::vector<G3Buffer> &, int numa,
                   G3Deadline = G3Deadline::max());
    G3Result write(const std::string &, const std::vector<G3Buffer> &,
                   G3Deadline = G3Deadline::max());
    G3Result read(const std::string &, const std::vector<G3Buffer> &, int numa,
                  G3Deadline = G3Deadline::max(), uint64_t expected_id = 0);
    G3Result read(const std::string &, const std::vector<G3Buffer> &,
                  G3Deadline = G3Deadline::max(), uint64_t expected_id = 0);
    G3BatchResult read_batch(const std::vector<G3Read> &, int numa,
                             G3Deadline = G3Deadline::max()); // at most eight objects
    bool exists(const std::string &) const;
    std::vector<G3Entry> enumerate() const;
    std::vector<G3DeviceInfo> devices() const;
    Status set_device_state(size_t, DeviceState);
    Status checkpoint(); // flush DIRTY state only
    G3CloseResult close(CloseMode);
    static int infer_numa(const std::vector<G3Buffer> &); // kernel buffer topology only
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class G3Session;
    std::shared_ptr<void> acquire_async_use(const std::vector<G3Buffer> &);
    void run_with_async_use(const std::shared_ptr<void> &, const std::function<void()> &);
    Status start_async_deregister(G3MemoryHandle);
    void cancel_async_deregister(G3MemoryHandle);
    Status finish_deregister(G3MemoryHandle, bool already_retiring);
};
// Standalone asynchronous API. Open uses a bounded shared two-worker executor;
// each opened session has a fixed worker count and bounded pending queue. Buffers
// and the SDK context must outlive their futures and explicit registration use.
class G3Session {
public:
    static std::future<std::shared_ptr<G3Session>> open(G3Config, G3Context,
                                                       size_t workers = 2, size_t queue_capacity = 64);
    ~G3Session(); // drains and DISCARD-closes; explicit CLEAN completion is required
    G3Session(const G3Session &) = delete;
    G3Session &operator=(const G3Session &) = delete;
    std::future<G3MemoryHandle> register_memory(std::vector<G3Buffer>);
    std::future<G3MemoryHandle> borrow_registered_memory(std::vector<G3Buffer>);
    std::future<Status> deregister_memory(G3MemoryHandle);
    std::future<G3Result> write(std::string, std::vector<G3Buffer>, int numa,
                                G3Deadline = G3Deadline::max());
    std::future<G3Result> write(std::string, std::vector<G3Buffer>,
                                G3Deadline = G3Deadline::max());
    std::future<G3Result> read(std::string, std::vector<G3Buffer>, int numa,
                               G3Deadline = G3Deadline::max(), uint64_t expected_id = 0);
    std::future<G3Result> read(std::string, std::vector<G3Buffer>,
                               G3Deadline = G3Deadline::max(), uint64_t expected_id = 0);
    std::future<G3BatchResult> read_batch(std::vector<G3Read>, int numa,
                                         G3Deadline = G3Deadline::max());
    bool exists(const std::string &) const;
    std::shared_future<G3CloseResult> close(CloseMode);
private:
    struct Impl;
    explicit G3Session(std::unique_ptr<G3TransferLayer>, size_t workers,
                        size_t capacity, uint64_t timeout_ms);
    std::unique_ptr<Impl> impl_;
};
} // namespace nixlshard
