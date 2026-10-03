/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "nixlshard/storage.h"
#include <map>
#include <optional>
#include <memory>

namespace nixlshard {
struct AgentConfig {
    std::string name;
    std::vector<DiskConfig> disks;
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
    void release(std::uint64_t handle);
    std::vector<bool> batch_exists(const std::vector<std::string> &keys,
                                  const std::vector<std::string> &hints = {});
    Status checkpoint();
    Endpoint endpoint() const;
    std::map<std::string, std::uint64_t> stats() const;
    void close();
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
