/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "nixlshard/g3.h"
#include <map>
#include <optional>
#include <memory>

namespace nixlshard {
struct G3InstanceConfig {
    std::string name = "local", namespace_id;
    int numa_node = 0;
    std::vector<G3DeviceConfig> devices;
};
struct AgentConfig {
    std::string name;
    std::vector<DiskConfig> disks;
    std::map<std::string, int> disk_numa_nodes;
    std::string namespace_id, g3_instance = "local";
    int numa_node = 0; // legacy generic API; framework deployments supply this explicitly
    MemoryMode registration_mode = MemoryMode::explicit_registration;
    std::vector<G3InstanceConfig> g3_instances;
    Endpoint listen;
    std::optional<Endpoint> metadata_endpoint;
    std::map<std::string, Endpoint> peers;
    std::size_t max_inflight = 64;
    std::size_t workers = 2;
    std::size_t staging_slots = 8;
    std::size_t staging_slot_bytes = 32 * 1024 * 1024;
    // Consecutive same-owner loads may share one owned staging slot/RPC.
    // Both endpoints must support load_batch; one retains the original path.
    std::size_t remote_batch_limit = 1;
    unsigned timeout_ms = 5000;
    bool direct_io = false;
    // Bounded request-specific diagnostics; off on the normal execution path.
    bool enable_trace = false;
    // Register caller DRAM and read directly into it. Caller destinations must
    // remain unavailable for reuse until is_quiescent() and release() succeed.
    bool direct_receive = false;
};
struct TraceEvent {
    std::string stage, request_id;
    std::uint64_t start_ns = 0, end_ns = 0, bytes = 0;
    std::size_t first_object = 0, object_count = 0;
    // Owner durations are sequential children of remote_rpc, never extra time.
    std::uint8_t owner_timing_flags = 0;
    std::uint64_t owner_posix_ns = 0, owner_ucx_ns = 0, owner_read_bytes = 0;
    std::uint64_t owner_metadata_ns = 0, owner_metadata_bytes = 0;
    std::uint64_t owner_staging_copy_ns = 0, owner_staging_copy_bytes = 0;
    bool direct_receive = false;
    std::size_t destination_segments = 0;
};
class Agent {
public:
    explicit Agent(const AgentConfig &);
    ~Agent();
    Agent(const Agent &) = delete;
    std::uint64_t register_memory(std::uintptr_t address, std::size_t length);
    void deregister_memory(std::uint64_t token);
    std::uint64_t batch_store(const std::vector<Object> &);
    std::uint64_t batch_load(const std::vector<Object> &);
    std::optional<std::vector<Status>> poll(std::uint64_t handle) const;
    std::vector<TraceEvent> trace(std::uint64_t handle) const;
    bool is_quiescent(std::uint64_t handle) const;
    void release(std::uint64_t handle);
    std::vector<bool> batch_exists(const std::vector<std::string> &keys,
                                  const std::vector<std::string> &hints = {},
                                  const std::string &g3_instance = "");
    Status checkpoint();
    Endpoint endpoint() const;
    std::map<std::string, std::uint64_t> stats() const;
    void close(CloseMode mode = CloseMode::clean);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class MetadataServer {
public:
    explicit MetadataServer(Endpoint, std::size_t max_entries = 100000,
                            unsigned ttl_ms = 60000);
    ~MetadataServer();
    Endpoint endpoint() const;
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nixlshard
