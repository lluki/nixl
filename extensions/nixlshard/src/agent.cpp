/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"
#include <nixl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace nixlshard {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t batch_limit = 128, key_limit = 65536, peer_limit = 128;
struct Finally {
    std::function<void()> fn;
    ~Finally() { if (fn) fn(); }
};
std::string incarnation(const std::string &name) {
    std::random_device random;
    return name + "-" + std::to_string(random()) + "-" + std::to_string(random());
}
void require_nixl(nixl_status_t rc, const char *operation) {
    if (rc != NIXL_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}
Status read_status(wire::Reader &r) {
    auto value = r.u8();
    if (value > static_cast<unsigned>(Status::canceled)) throw std::invalid_argument("invalid peer status");
    return static_cast<Status>(value);
}
std::string status_reply(Status status) {
    wire::Writer w; w.u8(static_cast<std::uint8_t>(status)); return w.data;
}
nixlAgentConfig native_config() {
    nixlAgentConfig c;
    c.useProgThread = true;
    c.useListenThread = false;
    c.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    c.pthrDelay = 100;
    return c;
}
}

struct Agent::Impl {
    AgentConfig config;
    std::string identity;
    std::unique_ptr<nixlAgent> native;
    nixlBackendH *posix = nullptr, *ucx = nullptr;
    nixl_opt_args_t posix_options, ucx_options, dram_options;
    nixl_reg_dlist_t dram_registration{DRAM_SEG};
    std::vector<nixl_reg_dlist_t> disk_registrations;
    std::vector<std::unique_ptr<DiskIndex>> disks;
    std::vector<int> payload_fds;
    void *scratch = nullptr;
    std::string scratch_metadata;
    std::mutex native_md_mutex;
    std::set<std::string> imported_names;

    struct Registration { std::uintptr_t address; std::size_t bytes; std::size_t refs = 0; };
    struct Batch {
        std::vector<Object> objects;
        std::vector<Status> results;
        std::vector<bool> reported;
        bool store = false, finished = false, canceled = false;
        Clock::time_point deadline;
        std::mutex mutex;
        std::condition_variable done;
    };
    mutable std::mutex mutex;
    std::condition_variable work;
    std::unordered_map<std::uint64_t, Registration> registrations;
    std::unordered_map<std::uint64_t, std::shared_ptr<Batch>> batches;
    std::deque<std::shared_ptr<Batch>> queue;
    std::uint64_t next_token = 1, next_handle = 1;
    std::atomic<std::uint64_t> next_request{1};
    std::vector<std::thread> workers;
    std::atomic<bool> stopping{false}, closed{false};
    std::mutex close_mutex;
    std::unique_ptr<wire::Server> server;

    struct Peer {
        Endpoint endpoint;
        std::string identity;
        std::unique_ptr<wire::Connection> channel;
        // Used only by maintenance, so cleanup RPCs never occupy foreground channel.
        std::unique_ptr<wire::Connection> cleanup_channel;
        std::mutex mutex;
    };
    std::mutex peers_mutex;
    std::map<std::string, std::shared_ptr<Peer>> peers;
    std::unordered_map<std::string, std::string> hints;
    std::set<std::string> lookup_queue;
    std::thread maintenance;
    std::mutex maintenance_mutex;
    std::condition_variable maintenance_wake;

    std::mutex pool_mutex;
    std::vector<std::size_t> free_slots;
    struct Quarantine { std::size_t slot; std::string owner, identity, request; };
    std::vector<Quarantine> quarantines;
    struct Cleanup { std::string owner, identity, request; };
    std::deque<Cleanup> cleanup_queue;
    struct Incoming { bool active = false, canceled = false; };
    std::mutex incoming_mutex;
    std::unordered_map<std::string, Incoming> incoming;
    mutable std::mutex stats_mutex;
    std::map<std::string, std::uint64_t> counters;

    explicit Impl(const AgentConfig &cfg) : config(cfg), identity(incarnation(cfg.name)) {
        if (config.name.empty() || config.name.size() > 512 || !config.max_inflight ||
            config.max_inflight > 4096 || !config.workers || config.workers > 64 ||
            !config.staging_slots || config.staging_slots > 4096 ||
            !config.staging_slot_bytes || config.staging_slot_bytes % 4096 ||
            config.staging_slot_bytes > std::numeric_limits<std::size_t>::max() / config.staging_slots ||
            !config.timeout_ms || config.peers.size() > peer_limit || config.disks.size() > 128)
            throw std::invalid_argument("invalid agent configuration or resource limits");
        native = std::make_unique<nixlAgent>(identity, native_config());
        // Linux AIO has tested error/cancellation draining in the selected NIXL base.
        // Other POSIX queue implementations are not enabled by this prototype.
        require_nixl(native->createBackend("POSIX", {{"use_aio", "true"}}, posix), "POSIX Linux AIO backend");
        require_nixl(native->createBackend("UCX", {}, ucx), "UCX backend");
        posix_options.backends = {posix};
        ucx_options.backends = {ucx};
        dram_options.backends = {posix, ucx};
        const auto pool_bytes = config.staging_slots * config.staging_slot_bytes;
        if (::posix_memalign(&scratch, 4096, pool_bytes)) throw std::bad_alloc();
        // Registration/destruction ordering is protected by the constructor guard.
        Finally failed{[this] { if (server) server->close();
                               native.reset(); std::free(scratch); scratch = nullptr;
                               for (auto fd : payload_fds) ::close(fd); }};
        dram_registration.addDesc(nixlBlobDesc(reinterpret_cast<std::uintptr_t>(scratch), pool_bytes, 0, ""));
        require_nixl(native->registerMem(dram_registration, &dram_options), "staging registration");
        auto md_options = ucx_options; md_options.includeConnInfo = true;
        require_nixl(native->getLocalPartialMD(dram_registration, scratch_metadata, &md_options),
                     "staging metadata");
        for (std::size_t i = 0; i < config.staging_slots; ++i) free_slots.push_back(i);
        for (const auto &dc : config.disks) {
            if (!dc.unit_bytes || dc.unit_bytes > config.staging_slot_bytes || config.staging_slot_bytes % dc.unit_bytes ||
                (config.direct_io && dc.unit_bytes % 4096))
                throw std::invalid_argument("disk allocation unit incompatible with aligned staging");
            auto disk = std::make_unique<DiskIndex>(dc);
            int fd = ::open(dc.path.c_str(), O_RDWR | O_CLOEXEC | (config.direct_io ? O_DIRECT : 0));
            if (fd < 0) throw std::runtime_error("cannot open disk payload descriptor");
            payload_fds.push_back(fd);
            nixl_reg_dlist_t desc(FILE_SEG);
            desc.addDesc(nixlBlobDesc(0, disk->capacity_bytes(), fd, ""));
            require_nixl(native->registerMem(desc, &posix_options), "disk registration");
            disk_registrations.push_back(desc);
            disks.push_back(std::move(disk));
        }
        for (const auto &[owner, ep] : config.peers) {
            if (owner.empty() || owner.size() > 1024 || owner == config.name || ep.host.empty() || !ep.port)
                throw std::invalid_argument("invalid configured peer");
            auto peer = std::make_shared<Peer>(); peer->endpoint = ep; peers.emplace(owner, peer);
        }
        server = std::make_unique<wire::Server>(config.listen,
            [this](std::string_view request) { return serve(request); }, config.timeout_ms, 2 * peer_limit + 16);
        try {
            for (std::size_t i = 0; i < config.workers; ++i) workers.emplace_back([this] { worker(); });
            maintenance = std::thread([this] { maintain(); });
        } catch (...) {
            stopping.store(true); work.notify_all();
            for (auto &w : workers) if (w.joinable()) w.join();
            server->close(); throw;
        }
        failed.fn = {};
    }
    ~Impl() {
        // close() has established quiescence before freeing registered storage.
        if (native) {
            for (auto &r : disk_registrations) native->deregisterMem(r, &posix_options);
            native->deregisterMem(dram_registration, &dram_options);
            native.reset();
        }
        for (auto fd : payload_fds) ::close(fd);
        std::free(scratch);
    }
    void count(const std::string &name, std::uint64_t value = 1) {
        std::lock_guard lock(stats_mutex); counters[name] += value;
    }
    void elapsed(const std::string &name, Clock::time_point start) {
        count(name, std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }
    void *address(std::size_t slot) const {
        return static_cast<char *>(scratch) + slot * config.staging_slot_bytes;
    }
    std::optional<std::size_t> acquire_slot() {
        std::lock_guard lock(pool_mutex);
        if (free_slots.empty()) { count("staging_busy"); return {}; }
        auto slot = free_slots.back(); free_slots.pop_back(); return slot;
    }
    void free_slot(std::size_t slot) {
        std::lock_guard lock(pool_mutex); free_slots.push_back(slot);
    }
    std::shared_ptr<Peer> get_peer(const std::string &owner) {
        std::lock_guard lock(peers_mutex);
        auto it = peers.find(owner); return it == peers.end() ? nullptr : it->second;
    }
    bool import_metadata(const std::string &expected, const std::string &metadata) {
        std::lock_guard lock(native_md_mutex);
        if (imported_names.count(expected)) return true;
        if (imported_names.size() >= peer_limit) return false;
        std::string extracted;
        if (native->loadRemoteMD(metadata, extracted) != NIXL_SUCCESS) return false;
        if (extracted != expected) { native->invalidateRemoteMD(extracted); return false; }
        imported_names.insert(expected);
        return true;
    }
    std::uint64_t submit(const std::vector<Object> &objects, bool store) {
        if (objects.empty() || objects.size() > batch_limit) throw std::invalid_argument("invalid batch size");
        auto batch = std::make_shared<Batch>(); batch->objects = objects; batch->store = store;
        batch->results.assign(objects.size(), Status::not_ready);
        batch->reported.assign(objects.size(), false);
        batch->deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
        std::lock_guard lock(mutex);
        if (stopping.load()) throw std::runtime_error("agent closing");
        if (batches.size() >= config.max_inflight) throw std::runtime_error("maximum in-flight handles reached");
        for (const auto &object : objects) {
            if (object.key.empty() || object.key.size() > key_limit || object.segments.empty() ||
                object.segments.size() > 65536 || object.hint.size() > 1024)
                throw std::invalid_argument("invalid object");
            std::size_t total = 0;
            for (const auto &segment : object.segments) {
                auto r = registrations.find(segment.registration);
                if (r == registrations.end() || !segment.length || segment.offset > r->second.bytes ||
                    segment.length > r->second.bytes - segment.offset ||
                    segment.length > config.staging_slot_bytes - total)
                    throw std::invalid_argument("invalid registered segment");
                total += segment.length;
            }
        }
        auto handle = next_handle++;
        if (!handle) throw std::overflow_error("handle sequence exhausted");
        batches.emplace(handle, batch);
        try { queue.push_back(batch); }
        catch (...) { batches.erase(handle); throw; }
        for (const auto &object : objects)
            for (const auto &segment : object.segments) ++registrations.at(segment.registration).refs;
        work.notify_one();
        return handle;
    }
    std::size_t object_bytes(const Object &object) const {
        std::size_t bytes = 0; for (auto &s : object.segments) bytes += s.length; return bytes;
    }
    // Caller registrations remain pinned by the live batch until release().
    void copy(const Object &object, void *buffer, bool into_staging) {
        auto start = Clock::now();
        std::vector<std::uintptr_t> addresses;
        {
            std::lock_guard lock(mutex);
            for (const auto &s : object.segments) addresses.push_back(registrations.at(s.registration).address + s.offset);
        }
        auto *cursor = static_cast<char *>(buffer);
        for (std::size_t i = 0; i < object.segments.size(); ++i) {
            const auto bytes = object.segments[i].length;
            if (into_staging) std::memcpy(cursor, reinterpret_cast<void *>(addresses[i]), bytes);
            else std::memcpy(reinterpret_cast<void *>(addresses[i]), cursor, bytes);
            cursor += bytes;
        }
        count("staging_copy_bytes", object_bytes(object)); elapsed("staging_copy_ns", start);
    }
    Status transfer(nixl_xfer_op_t op, const nixl_xfer_dlist_t &local,
                    const nixl_xfer_dlist_t &remote, const std::string &owner,
                    nixl_opt_args_t &options, Clock::time_point deadline) {
        if (Clock::now() >= deadline) return Status::timeout;
        nixlXferReqH *request = nullptr;
        auto rc = native->createXferReq(op, local, remote, owner, request, &options);
        if (rc != NIXL_SUCCESS) return Status::io_error;
        rc = native->postXferReq(request, &options);
        bool timed_out = false;
        // Do not cancel POSIX handles: backend callbacks retain descriptor/request pointers.
        // Logical deadlines are observable through poll(), while this worker drains I/O.
        while (rc == NIXL_IN_PROG) {
            timed_out |= Clock::now() >= deadline;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            rc = native->getXferStatus(request);
        }
        require_nixl(native->releaseXferReq(request), "transfer release");
        if (timed_out || Clock::now() >= deadline) return Status::timeout;
        return rc == NIXL_SUCCESS ? Status::success : Status::io_error;
    }
    Status disk_io(std::size_t disk_id, const Allocation &allocation, std::size_t slot,
                   bool write, Clock::time_point deadline) {
        auto start = Clock::now();
        auto &disk = *disks[disk_id];
        const auto unit = disk.unit_bytes();
        Status result = Status::success;
        std::size_t cursor = 0;
        while (cursor < allocation.slots.size()) {
            nixl_xfer_dlist_t memory(DRAM_SEG), file(FILE_SEG);
            std::size_t descriptors = 0;
            while (cursor < allocation.slots.size() && descriptors++ < 128) {
                auto end = cursor + 1;
                while (end < allocation.slots.size() &&
                       allocation.slots[end] == allocation.slots[end - 1] + 1) ++end;
                auto length = (end - cursor) * unit;
                memory.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(address(slot)) + cursor * unit, length, 0));
                file.addDesc(nixlBasicDesc(disk.slot_offset(allocation.slots[cursor]), length, payload_fds[disk_id]));
                cursor = end;
            }
            result = transfer(write ? NIXL_WRITE : NIXL_READ, memory, file, identity, posix_options, deadline);
            if (result != Status::success) break;
        }
        elapsed(write ? "posix_write_ns" : "posix_read_ns", start);
        if (result == Status::success) count(write ? "posix_write_bytes" : "posix_read_bytes",
                                          allocation.slots.size() * unit);
        return result;
    }
    std::pair<std::size_t, Status> pin(const std::string &key, Allocation &allocation) {
        Status result = Status::not_found;
        for (std::size_t i = 0; i < disks.size(); ++i) {
            auto status = disks[i]->pin(key, allocation);
            if (status == Status::success) return {i, status};
            if (status != Status::not_found) result = status;
        }
        return {0, result};
    }
    Status store_object(const Object &object, std::size_t slot, Clock::time_point deadline, Batch &batch) {
        const auto bytes = object_bytes(object);
        if (disks.empty()) return Status::no_space;
        auto first = std::hash<std::string>{}(object.key) % disks.size();
        Status last = Status::no_space;
        for (std::size_t probe = 0; probe < disks.size(); ++probe) {
            auto id = (first + probe) % disks.size();
            auto &disk = *disks[id];
            Allocation allocation;
            auto status = disk.reserve(object.key, bytes, allocation);
            // Eagerly reclaim one victim at a time; every successful removal is durable.
            while (status == Status::no_space && Clock::now() < deadline) {
                auto evicted = disk.evict_one();
                if (evicted != Status::success) break;
                count("evictions");
                status = disk.reserve(object.key, bytes, allocation);
            }
            if (status != Status::success) { last = status; continue; }
            if (allocation.already_present) return Status::success;
            Finally rollback{[&disk, &allocation] { disk.abort(allocation.id); }};
            std::memset(address(slot), 0, allocation.slots.size() * disk.unit_bytes());
            {
                std::lock_guard lock(batch.mutex);
                if (batch.canceled || Clock::now() >= deadline) return Status::timeout;
                copy(object, address(slot), true);
            }
            status = disk_io(id, allocation, slot, true, deadline);
            if (status != Status::success) return status;
            status = disk.publish(allocation.id);
            if (status == Status::success) { rollback.fn = {}; count("stores"); }
            return status;
        }
        return Clock::now() >= deadline ? Status::timeout : last;
    }
    std::string owner_hint(const Object &object) {
        if (!object.hint.empty()) return object.hint;
        std::lock_guard lock(peers_mutex);
        auto it = hints.find(object.key);
        if (it != hints.end()) return it->second;
        if (config.metadata_endpoint && lookup_queue.size() < 100000) lookup_queue.insert(object.key);
        maintenance_wake.notify_one();
        return {};
    }
    void refresh_hint(const Object &object) {
        if (!config.metadata_endpoint || !object.hint.empty()) return;
        std::lock_guard lock(peers_mutex);
        hints.erase(object.key);
        if (lookup_queue.size() < 100000) lookup_queue.insert(object.key);
        maintenance_wake.notify_one();
    }
    void enqueue_cleanup(const std::string &owner, const std::string &peer_identity,
                         const std::string &request) {
        std::lock_guard lock(pool_mutex);
        if (cleanup_queue.size() < config.max_inflight * batch_limit)
            cleanup_queue.push_back({owner, peer_identity, request});
        maintenance_wake.notify_one();
    }
    Status remote_load(const Object &object, std::size_t slot, Clock::time_point deadline,
                       bool &quarantined) {
        auto owner = owner_hint(object);
        if (owner.empty()) return config.metadata_endpoint ? Status::not_ready : Status::not_found;
        if (owner == config.name) return Status::not_found;
        auto peer = get_peer(owner);
        if (!peer) { count("remote_unknown_peer"); refresh_hint(object); return Status::not_ready; }
        std::unique_lock peer_lock(peer->mutex, std::try_to_lock);
        if (!peer_lock.owns_lock()) { count("remote_channel_busy"); return Status::not_ready; }
        if (!peer->channel || !peer->channel->usable()) { count("remote_channel_not_ready"); refresh_hint(object); return Status::not_ready; }
        if (Clock::now() >= deadline) return Status::timeout;
        auto request = identity + ":" + std::to_string(next_request.fetch_add(1));
        auto target_identity = peer->identity;
        wire::Writer w;
        w.u8(wire::load); w.str(request); w.str(target_identity); w.str(object.key);
        w.u64(object_bytes(object)); w.u64(reinterpret_cast<std::uintptr_t>(address(slot)));
        w.str(identity); w.str(scratch_metadata);
        auto start = Clock::now();
        bool control_timeout = false;
        try {
            const auto timeout = std::max<std::int64_t>(1,
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
            std::string response;
            try { response = peer->channel->call(w.data, static_cast<unsigned>(timeout)); }
            catch (const wire::Timeout &) { control_timeout = true; throw; }
            wire::Reader r(response); auto result = read_status(r); r.finish();
            enqueue_cleanup(owner, target_identity, request);
            if (result != Status::success) refresh_hint(object);
            elapsed("remote_control_ns", start);
            if (result == Status::success) count("remote_read_bytes", object_bytes(object));
            return result;
        } catch (...) {
            peer->channel.reset();
            {
                std::lock_guard lock(pool_mutex);
                quarantines.push_back({slot, owner, target_identity, request});
            }
            quarantined = true; count("quarantined_slots"); maintenance_wake.notify_one();
            refresh_hint(object);
            elapsed("remote_control_ns", start);
            return control_timeout || Clock::now() >= deadline ? Status::timeout : Status::io_error;
        }
    }
    Status load_object(const Object &object, std::size_t slot, Clock::time_point deadline,
                       bool &quarantined) {
        Allocation allocation;
        auto [id, status] = pin(object.key, allocation);
        if (status == Status::success) {
            Finally unpin{[this, id, &allocation] { disks[id]->unpin(allocation.id); }};
            if (allocation.bytes != object_bytes(object)) return Status::invalid_input;
            return disk_io(id, allocation, slot, false, deadline);
        }
        if (status != Status::not_found) return status;
        return remote_load(object, slot, deadline, quarantined);
    }
    void worker() {
        for (;;) {
            std::shared_ptr<Batch> batch;
            {
                std::unique_lock lock(mutex);
                work.wait(lock, [this] { return stopping.load() || !queue.empty(); });
                if (queue.empty()) return;
                batch = queue.front(); queue.pop_front();
            }
            for (std::size_t i = 0; i < batch->objects.size(); ++i) {
                Status result = Status::io_error;
                bool canceled;
                { std::lock_guard lock(batch->mutex); canceled = batch->canceled; }
                if (canceled || Clock::now() >= batch->deadline) result = Status::timeout;
                else if (auto slot = acquire_slot()) {
                    bool quarantined = false;
                    Finally release_slot{[this, slot, &quarantined] { if (!quarantined) free_slot(*slot); }};
                    try {
                        result = batch->store ? store_object(batch->objects[i], *slot, batch->deadline, *batch) :
                                 load_object(batch->objects[i], *slot, batch->deadline, quarantined);
                        std::lock_guard lock(batch->mutex);
                        if (batch->canceled || Clock::now() >= batch->deadline) result = Status::timeout;
                        if (!batch->store && result == Status::success)
                            copy(batch->objects[i], address(*slot), false);
                    } catch (...) { result = Status::io_error; }
                } else result = Status::busy;
                { std::lock_guard lock(batch->mutex);
                  if (!batch->reported[i]) { batch->results[i] = result; batch->reported[i] = true; } }
                count(status_name(result));
            }
            { std::lock_guard lock(batch->mutex); batch->finished = true; }
            batch->done.notify_all();
        }
    }
    std::string serve(std::string_view request) {
        try {
            wire::Reader r(request); auto operation = r.u8();
            if (operation == wire::hello) {
                r.finish(); wire::Writer w; w.u8(static_cast<unsigned>(Status::success));
                w.str(identity); w.str(scratch_metadata); return w.data;
            }
            if (operation == wire::exists) {
                auto expected = r.str(2048);
                auto n = r.u32(); if (!n || n > batch_limit) throw std::invalid_argument("invalid exists batch");
                std::vector<std::string> keys; for (unsigned i = 0; i < n; ++i) keys.push_back(r.str(key_limit));
                r.finish();
                if (expected != identity) return status_reply(Status::not_ready);
                wire::Writer w; w.u8(static_cast<unsigned>(Status::success)); w.u32(n);
                for (auto &key : keys) {
                    bool found = false; for (auto &disk : disks) found |= disk->exists(key);
                    w.u8(found);
                }
                return w.data;
            }
            if (operation == wire::cleanup) {
                auto expected = r.str(2048), id = r.str(4096); r.finish();
                if (expected != identity) return status_reply(Status::not_ready);
                count("remote_cleanup_requests");
                std::lock_guard lock(incoming_mutex);
                auto it = incoming.find(id);
                wire::Writer w; w.u8(static_cast<unsigned>(Status::success));
                if (it != incoming.end()) {
                    if (it->second.active) w.u8(false);
                    else if (it->second.canceled) w.u8(true);
                    else { incoming.erase(it); w.u8(true); }
                } else if (incoming.size() < config.max_inflight * batch_limit) {
                    // Fences a load delayed behind the cleanup on another connection.
                    incoming.emplace(id, Incoming{false, true}); w.u8(true);
                } else w.u8(false);
                return w.data;
            }
            if (operation != wire::load) return status_reply(Status::invalid_input);
            auto id = r.str(4096), expected = r.str(2048), key = r.str(key_limit);
            auto bytes = r.u64(), target = r.u64();
            auto requester = r.str(2048), metadata = r.str(4 * 1024 * 1024); r.finish();
            if (expected != identity || stopping.load()) return status_reply(Status::not_ready);
            if (key.empty() || !bytes || bytes > config.staging_slot_bytes ||
                target > std::numeric_limits<std::uintptr_t>::max() - bytes ||
                id.empty() || requester.empty()) return status_reply(Status::invalid_input);
            {
                std::lock_guard lock(incoming_mutex);
                auto existing = incoming.find(id);
                if (existing != incoming.end())
                    return status_reply(existing->second.canceled ? Status::canceled : Status::busy);
                if (incoming.size() >= config.max_inflight * batch_limit) return status_reply(Status::busy);
                incoming.emplace(id, Incoming{true, false});
            }
            Finally complete{[this, &id] {
                std::lock_guard lock(incoming_mutex); incoming.at(id).active = false;
            }};
            auto slot = acquire_slot();
            if (!slot) return status_reply(Status::busy);
            Finally release_slot{[this, &slot] { free_slot(*slot); }};
            Allocation allocation;
            auto [disk_id, status] = pin(key, allocation);
            if (status != Status::success) return status_reply(status);
            Finally unpin{[this, disk_id, &allocation] { disks[disk_id]->unpin(allocation.id); }};
            if (allocation.bytes != bytes) return status_reply(Status::invalid_input);
            if (!import_metadata(requester, metadata)) return status_reply(Status::not_ready);
            auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
            status = disk_io(disk_id, allocation, *slot, false, deadline);
            if (status != Status::success) return status_reply(status);
            nixl_xfer_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
            local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(address(*slot)), bytes, 0));
            remote.addDesc(nixlBasicDesc(target, bytes, 0));
            auto start = Clock::now();
            status = transfer(NIXL_WRITE, local, remote, requester, ucx_options, deadline);
            elapsed("ucx_write_ns", start);
            if (status == Status::success) count("ucx_write_bytes", bytes);
            return status_reply(status);
        } catch (const std::invalid_argument &) { return status_reply(Status::invalid_input); }
        catch (...) { return status_reply(Status::io_error); }
    }
    void connect_peer(const std::shared_ptr<Peer> &peer) {
        std::unique_lock lock(peer->mutex, std::try_to_lock);
        if (!lock.owns_lock() || (peer->channel && peer->channel->usable())) return;
        try {
            auto channel = std::make_unique<wire::Connection>(peer->endpoint, std::min(config.timeout_ms, 250u));
            wire::Writer w; w.u8(wire::hello);
            auto response = channel->call(w.data, std::min(config.timeout_ms, 500u));
            wire::Reader r(response);
            if (read_status(r) != Status::success) return;
            auto name = r.str(2048), metadata = r.str(4 * 1024 * 1024); r.finish();
            if (!import_metadata(name, metadata)) return;
            peer->identity = std::move(name); peer->channel = std::move(channel);
            count("peer_connections");
        } catch (...) { peer->channel.reset(); count("peer_connect_failures"); }
    }
    bool cleanup(const Cleanup &item) {
        auto peer = get_peer(item.owner);
        if (!peer) return false;
        std::unique_lock lock(peer->mutex, std::try_to_lock);
        if (!lock.owns_lock() || !peer->channel || !peer->channel->usable()) return false;
        // A new control incarnation cannot prove that an older UCX write drained.
        // Retain the target slot until that specific incarnation confirms quiescence.
        if (peer->identity != item.identity) return false;
        auto endpoint = peer->endpoint;
        lock.unlock();
        try {
            if (!peer->cleanup_channel || !peer->cleanup_channel->usable())
                peer->cleanup_channel = std::make_unique<wire::Connection>(endpoint, std::min(config.timeout_ms, 250u));
            wire::Writer w; w.u8(wire::cleanup); w.str(item.identity); w.str(item.request);
            auto response = peer->cleanup_channel->call(w.data, std::min(config.timeout_ms, 250u));
            wire::Reader r(response);
            if (read_status(r) != Status::success) return false;
            bool quiescent = r.u8(); r.finish(); return quiescent;
        } catch (...) { peer->cleanup_channel.reset(); return false; }
    }
    void reclaim_quarantines() {
        std::vector<Quarantine> pending;
        std::deque<Cleanup> cleanups;
        {
            std::lock_guard lock(pool_mutex);
            pending.swap(quarantines); cleanups.swap(cleanup_queue);
        }
        std::vector<Quarantine> retained;
        for (auto &q : pending) {
            if (cleanup({q.owner, q.identity, q.request})) { free_slot(q.slot); count("quarantines_released"); }
            else retained.push_back(q);
        }
        std::deque<Cleanup> retry;
        for (auto &q : cleanups) { if (!cleanup(q)) retry.push_back(q); }
        {
            std::lock_guard lock(pool_mutex);
            quarantines.insert(quarantines.end(), retained.begin(), retained.end());
            while (!retry.empty() && cleanup_queue.size() < config.max_inflight * batch_limit) {
                cleanup_queue.push_back(std::move(retry.front())); retry.pop_front();
            }
        }
    }
    Status md_call(wire::Connection &channel, const wire::Writer &w) {
        auto response = channel.call(w.data, std::min(config.timeout_ms, 500u));
        wire::Reader r(response); auto status = read_status(r); r.finish(); return status;
    }
    void maintain() {
        std::unique_ptr<wire::Connection> md;
        std::uint64_t announcement = 0;
        auto last_checkpoint = Clock::now() - std::chrono::seconds(2);
        auto last_announce = last_checkpoint;
        while (!stopping.load()) {
            std::vector<std::shared_ptr<Peer>> known;
            { std::lock_guard lock(peers_mutex); for (auto &[_, p] : peers) known.push_back(p); }
            for (auto &peer : known) { if (stopping.load()) break; connect_peer(peer); }
            reclaim_quarantines();
            auto now = Clock::now();
            if (now - last_checkpoint >= std::chrono::seconds(1)) {
                auto start = Clock::now();
                for (auto &disk : disks) if (disk->checkpoint() != Status::success) count("checkpoint_errors");
                elapsed("metadata_checkpoint_ns", start); last_checkpoint = now;
            }
            if (config.metadata_endpoint) {
                try {
                    if (!md || !md->usable()) {
                        md = std::make_unique<wire::Connection>(*config.metadata_endpoint, std::min(config.timeout_ms, 250u));
                        last_announce = now - std::chrono::seconds(2);
                    }
                    if (now - last_announce >= std::chrono::seconds(1)) {
                        wire::Writer w; w.u8(wire::register_owner); w.str(config.name); w.str(identity);
                        auto ep = server->endpoint(); w.str(ep.host); w.u32(ep.port);
                        if (md_call(*md, w) != Status::success) throw std::runtime_error("metadata registration rejected");
                        std::vector<std::string> keys;
                        for (auto &disk : disks) {
                            auto snapshot = disk->snapshot_keys(); keys.insert(keys.end(), snapshot.begin(), snapshot.end());
                        }
                        for (std::size_t i = 0; i < keys.size(); i += batch_limit) {
                            wire::Writer a; a.u8(wire::announce); a.str(config.name); a.str(identity);
                            a.u64(++announcement); auto n = std::min(batch_limit, keys.size() - i); a.u32(n);
                            for (std::size_t j = 0; j < n; ++j) a.str(keys[i + j]);
                            if (md_call(*md, a) != Status::success) count("metadata_announce_errors");
                        }
                        last_announce = now;
                    }
                    std::vector<std::string> keys;
                    {
                        std::lock_guard lock(peers_mutex);
                        for (auto it = lookup_queue.begin(); it != lookup_queue.end() && keys.size() < batch_limit;) {
                            keys.push_back(*it); it = lookup_queue.erase(it);
                        }
                    }
                    if (!keys.empty()) {
                        wire::Writer w; w.u8(wire::lookup); w.u32(keys.size()); for (auto &key : keys) w.str(key);
                        auto response = md->call(w.data, std::min(config.timeout_ms, 500u));
                        wire::Reader r(response);
                        if (read_status(r) != Status::success || r.u32() != keys.size()) throw std::runtime_error("invalid metadata lookup");
                        std::vector<std::string> owners;
                        for (auto &key : keys) {
                            auto owner = r.str(1024); auto owner_incarnation = r.str(2048); (void) owner_incarnation;
                            if (owner.empty() || owner == config.name) continue;
                            { std::lock_guard lock(peers_mutex); if (hints.size() >= 100000) hints.clear(); hints[key] = owner; }
                            owners.push_back(owner);
                        }
                        r.finish();
                        for (auto &owner : owners) {
                            wire::Writer d; d.u8(wire::directory); d.str(owner);
                            auto result = md->call(d.data, std::min(config.timeout_ms, 500u));
                            wire::Reader dr(result); auto status = read_status(dr);
                            if (status != Status::success) { dr.finish(); continue; }
                            auto inc = dr.str(2048); (void) inc; auto host = dr.str(1024); auto port = dr.u32(); dr.finish();
                            if (!port || port > 65535) continue;
                            std::shared_ptr<Peer> p;
                            {
                                std::lock_guard lock(peers_mutex);
                                auto it = peers.find(owner);
                                if (it != peers.end()) p = it->second;
                                else if (peers.size() < peer_limit) {
                                    p = std::make_shared<Peer>(); peers.emplace(owner, p);
                                }
                            }
                            if (p) {
                                std::lock_guard lock(p->mutex);
                                if (p->endpoint.host != host || p->endpoint.port != port) {
                                    p->channel.reset();
                                    p->cleanup_channel.reset();
                                    p->endpoint = {host, static_cast<std::uint16_t>(port)};
                                }
                            }
                        }
                    }
                } catch (...) { md.reset(); count("metadata_connection_failures"); }
            }
            std::unique_lock lock(maintenance_mutex);
            maintenance_wake.wait_for(lock, std::chrono::milliseconds(100), [this] { return stopping.load(); });
        }
    }
    std::vector<bool> exists(const std::vector<std::string> &keys, const std::vector<std::string> &owners) {
        if (keys.size() > batch_limit || (!owners.empty() && keys.size() != owners.size()))
            throw std::invalid_argument("invalid exists batch");
        if (stopping.load()) throw std::runtime_error("agent closing");
        const auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
        std::vector<bool> results(keys.size(), false);
        std::map<std::string, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].empty() || keys[i].size() > key_limit) throw std::invalid_argument("invalid key");
            for (auto &disk : disks) results[i] = results[i] || disk->exists(keys[i]);
            if (results[i]) continue;
            Object object; object.key = keys[i]; if (!owners.empty()) object.hint = owners[i];
            auto owner = owner_hint(object);
            if (!owner.empty() && owner != config.name) groups[owner].push_back(i);
        }
        for (auto &[owner, indices] : groups) {
            if (Clock::now() >= deadline) break;
            auto peer = get_peer(owner); if (!peer) continue;
            std::unique_lock lock(peer->mutex, std::try_to_lock);
            if (!lock.owns_lock() || !peer->channel || !peer->channel->usable()) continue;
            try {
                wire::Writer w; w.u8(wire::exists); w.str(peer->identity); w.u32(indices.size());
                for (auto i : indices) w.str(keys[i]);
                const auto start = Clock::now();
                Finally timed{[this, start] { elapsed("exists_control_ns", start); }};
                const auto remaining = std::max<std::int64_t>(1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - start).count());
                auto response = peer->channel->call(w.data, static_cast<unsigned>(remaining));
                wire::Reader r(response); if (read_status(r) != Status::success) { r.finish(); continue; }
                if (r.u32() != indices.size()) throw std::invalid_argument("invalid exists response");
                for (auto i : indices) results[i] = r.u8() != 0;
                r.finish();
            } catch (...) { peer->channel.reset(); }
        }
        return results;
    }
    void close() {
        std::lock_guard closing(close_mutex);
        if (closed.load()) return;
        stopping.store(true); work.notify_all(); maintenance_wake.notify_all();
        if (maintenance.joinable()) maintenance.join();
        for (auto &w : workers) if (w.joinable()) w.join();
        // Complete incoming owner writes before unregistering source buffers/disks.
        server->close();
        reclaim_quarantines();
        {
            std::lock_guard lock(pool_mutex);
            if (!quarantines.empty()) throw std::runtime_error("remote I/O not quiescent; native staging retained");
        }
        for (auto &disk : disks) if (disk->checkpoint() != Status::success) count("checkpoint_errors");
        closed.store(true);
    }
};

Agent::Agent(const AgentConfig &config) : impl_(std::make_unique<Impl>(config)) {}
Agent::~Agent() {
    if (!impl_) return;
    try { impl_->close(); }
    catch (...) {
        // Unconfirmed remote writes may still hold this destination address.
        // A bounded retained pool is safer than freeing/reusing registered memory.
        (void) impl_.release();
    }
}
std::uint64_t Agent::register_memory(std::uintptr_t address, std::size_t bytes) {
    if (!address || !bytes || address > std::numeric_limits<std::uintptr_t>::max() - bytes)
        throw std::invalid_argument("invalid caller memory");
    std::lock_guard lock(impl_->mutex);
    if (impl_->stopping.load()) throw std::runtime_error("agent closing");
    if (impl_->registrations.size() >= 65536) throw std::runtime_error("registration limit reached");
    auto token = impl_->next_token++;
    if (!token) throw std::overflow_error("registration sequence exhausted");
    impl_->registrations.emplace(token, Impl::Registration{address, bytes, 0}); return token;
}
void Agent::deregister_memory(std::uint64_t token) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->registrations.find(token);
    if (it == impl_->registrations.end()) throw std::invalid_argument("unknown registration");
    if (it->second.refs) throw std::runtime_error("registration belongs to unreleased transfers");
    impl_->registrations.erase(it);
}
std::uint64_t Agent::batch_store(const std::vector<Object> &objects) { return impl_->submit(objects, true); }
std::uint64_t Agent::batch_load(const std::vector<Object> &objects) { return impl_->submit(objects, false); }
std::optional<std::vector<Status>> Agent::poll(std::uint64_t handle) const {
    std::shared_ptr<Impl::Batch> batch;
    { std::lock_guard lock(impl_->mutex); auto it = impl_->batches.find(handle);
      if (it == impl_->batches.end()) throw std::invalid_argument("unknown transfer handle");
      batch = it->second; }
    std::lock_guard lock(batch->mutex);
    if (batch->finished) return batch->results;
    if (Clock::now() >= batch->deadline) {
        batch->canceled = true;
        for (std::size_t i = 0; i < batch->results.size(); ++i) {
            if (!batch->reported[i]) { batch->results[i] = Status::timeout; batch->reported[i] = true; }
        }
        return batch->results;
    }
    return {};
}
void Agent::release(std::uint64_t handle) {
    std::shared_ptr<Impl::Batch> batch;
    { std::lock_guard lock(impl_->mutex); auto it = impl_->batches.find(handle);
      if (it == impl_->batches.end()) throw std::invalid_argument("unknown transfer handle");
      batch = it->second; }
    { std::unique_lock lock(batch->mutex); batch->done.wait(lock, [&batch] { return batch->finished; }); }
    std::lock_guard lock(impl_->mutex);
    if (!impl_->batches.erase(handle)) throw std::invalid_argument("transfer already released");
    for (const auto &o : batch->objects) for (const auto &s : o.segments)
        --impl_->registrations.at(s.registration).refs;
}
std::vector<bool> Agent::batch_exists(const std::vector<std::string> &keys, const std::vector<std::string> &hints) {
    return impl_->exists(keys, hints);
}
Status Agent::checkpoint() {
    if (impl_->stopping.load()) return Status::not_ready;
    Status result = Status::success;
    auto start = Clock::now();
    for (auto &disk : impl_->disks) { auto s = disk->checkpoint(); if (s != Status::success) result = s; }
    impl_->elapsed("metadata_checkpoint_ns", start); return result;
}
Endpoint Agent::endpoint() const { return impl_->server->endpoint(); }
std::map<std::string, std::uint64_t> Agent::stats() const {
    auto *p = impl_.get();
    std::map<std::string, std::uint64_t> result;
    { std::lock_guard lock(p->stats_mutex); result = p->counters; }
    { std::lock_guard lock(p->pool_mutex); result["staging_free_slots"] = p->free_slots.size();
      result["staging_quarantined_slots"] = p->quarantines.size(); }
    return result;
}
void Agent::close() { impl_->close(); }
} // namespace nixlshard
