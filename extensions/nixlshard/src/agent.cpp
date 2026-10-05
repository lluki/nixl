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
std::uint64_t clock_ns(Clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}
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
    struct Domain {
        std::string name, namespace_id;
        int numa = 0;
        std::unique_ptr<G3TransferLayer> layer;
        G3MemoryHandle scratch_registration = 0;
    };
    std::vector<Domain> domains;
    // Derived identities serve MD announcements only. G3 remains authoritative.
    std::mutex directory_mutex;
    std::map<std::string, AllocationIdentity> directory;
    void *scratch = nullptr;
    std::string scratch_metadata;
    std::mutex native_md_mutex;
    std::set<std::string> imported_names;
    std::map<std::string, std::string> imported_metadata;
    std::map<std::string, std::shared_ptr<std::mutex>> memory_gates;
    std::string caller_metadata;

    struct Registration {
        std::uintptr_t address;
        std::size_t bytes;
        std::size_t refs = 0;
        std::unique_ptr<nixl_reg_dlist_t> native_desc;
        std::vector<std::pair<G3TransferLayer *, G3MemoryHandle>> g3_registrations;
    };
    struct Batch : std::enable_shared_from_this<Batch> {
        std::vector<Object> objects;
        std::vector<Status> results;
        std::vector<bool> reported;
        bool store = false, finished = false, canceled = false;
        std::size_t pending_direct = 0;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::vector<TraceEvent> trace;
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
    struct DirectQuarantine {
        std::shared_ptr<Batch> batch;
        std::string owner, identity, request;
    };
    std::vector<DirectQuarantine> direct_quarantines;
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
        if (!config.remote_batch_limit || config.remote_batch_limit > 8 || config.numa_node < 0)
            throw std::invalid_argument("invalid remote batch limit");
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
                               domains.clear(); native.reset(); std::free(scratch); scratch = nullptr; }};
        dram_registration.addDesc(nixlBlobDesc(reinterpret_cast<std::uintptr_t>(scratch), pool_bytes, 0, ""));
        require_nixl(native->registerMem(dram_registration, &dram_options), "staging registration");
        auto md_options = ucx_options; md_options.includeConnInfo = true;
        require_nixl(native->getLocalPartialMD(dram_registration, scratch_metadata, &md_options),
                     "staging metadata");
        caller_metadata = scratch_metadata;
        for (std::size_t i = 0; i < config.staging_slots; ++i) free_slots.push_back(i);
        auto instances = config.g3_instances;
        if (instances.empty()) {
            G3InstanceConfig instance;
            instance.name = config.g3_instance; instance.namespace_id = config.namespace_id;
            instance.numa_node = config.numa_node;
            for (const auto &disk : config.disks) {
                auto numa = config.disk_numa_nodes.find(disk.path);
                instance.devices.push_back({disk, numa == config.disk_numa_nodes.end() ? config.numa_node : numa->second});
            }
            instances.push_back(std::move(instance));
        }
        if (instances.size() > 128) throw std::invalid_argument("too many G3 instances");
        std::set<std::string> assigned_paths, names, namespaces;
        for (auto &instance : instances) {
            if (instance.namespace_id.empty()) instance.namespace_id = "nixlshard.generic.v2";
            if (instance.name.empty() || instance.name.size() > 512 || instance.numa_node < 0 ||
                instance.namespace_id.size() > 2048 || !names.insert(instance.name).second ||
                !namespaces.insert(instance.namespace_id).second)
                throw std::invalid_argument("invalid or ambiguous G3 instance namespace");
            Domain domain; domain.name = instance.name; domain.namespace_id = instance.namespace_id;
            domain.numa = instance.numa_node;
            G3Config gc; gc.instance_id = instance.name; gc.memory_mode = config.registration_mode;
            gc.timeout_ms = config.timeout_ms; gc.max_active = config.max_inflight;
            gc.staging_bytes = config.staging_slot_bytes;
            gc.events = [this, ns = instance.namespace_id](const IndexEvent &event) { index_event(ns, event); };
            for (auto device : instance.devices) {
                auto &dc = device.disk;
                if (!assigned_paths.insert(dc.path).second) throw std::invalid_argument("SSD assigned to multiple G3 instances");
                if (!dc.unit_bytes || dc.unit_bytes > config.staging_slot_bytes ||
                    config.staging_slot_bytes % dc.unit_bytes || (config.direct_io && dc.unit_bytes % 4096))
                    throw std::invalid_argument("disk allocation unit incompatible with aligned staging");
                if (!dc.namespace_id.empty() && dc.namespace_id != domain.namespace_id)
                    throw std::invalid_argument("disk namespace differs from its G3 instance");
                dc.namespace_id = domain.namespace_id; dc.direct_io = config.direct_io;
                if (!dc.min_object_bytes) dc.min_object_bytes = 1;
                if (!dc.max_object_bytes) dc.max_object_bytes = config.staging_slot_bytes;
                gc.devices.push_back(std::move(device));
            }
            if (!gc.devices.empty()) {
                domain.layer = std::make_unique<G3TransferLayer>(gc, G3Context{native.get(), identity, posix_options, dram_options});
                domain.scratch_registration = domain.layer->borrow_registered_memory(
                    {{reinterpret_cast<std::uintptr_t>(scratch), pool_bytes}});
                for (const auto &entry : domain.layer->enumerate())
                    index_event(domain.namespace_id, {IndexEventKind::committed, entry.identity});
            }
            domains.push_back(std::move(domain));
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
            domains.clear(); // close/deregister file and borrowed-memory views before the SDK
            for (auto &[token, registration] : registrations)
                if (registration.native_desc)
                    native->deregisterMem(*registration.native_desc, &dram_options);
            native->deregisterMem(dram_registration, &dram_options);
            native.reset();
        }
        std::free(scratch);
    }
    Domain &domain(const std::string &name = "") {
        const auto &selected = name.empty() ? config.g3_instance : name;
        for (auto &d : domains) if (d.name == selected) return d;
        throw std::invalid_argument("unknown G3 instance");
    }
    Domain *namespace_domain(const std::string &ns) {
        for (auto &d : domains) if (d.namespace_id == ns) return &d;
        return nullptr;
    }
    static std::string qualified_key(const std::string &ns, const std::string &key) {
        wire::Writer w; w.str(ns); w.str(key); return w.data;
    }
    std::string qualified_key(const Object &object) {
        return qualified_key(domain(object.g3_instance).namespace_id, object.key);
    }
    std::pair<Domain *, std::string> decode_key(const std::string &value) {
        wire::Reader r(value); auto ns = r.str(2048), key = r.str(32); r.finish();
        if (key.empty()) throw std::invalid_argument("empty local key");
        return {namespace_domain(ns), std::move(key)};
    }
    void index_event(const std::string &ns, const IndexEvent &event) {
        std::lock_guard lock(directory_mutex);
        if (event.kind == IndexEventKind::device_retired) {
            for (auto it = directory.begin(); it != directory.end();) {
                wire::Reader r(it->first); const auto entry_ns = r.str(2048);
                if (entry_ns == ns && it->second.generation == event.identity.generation) it = directory.erase(it);
                else ++it;
            }
            return;
        }
        auto key = qualified_key(ns, event.identity.key);
        if (event.kind == IndexEventKind::committed) directory[key] = event.identity;
        else {
            auto it = directory.find(key);
            if (it != directory.end() && it->second.id == event.identity.id &&
                it->second.generation == event.identity.generation && it->second.record == event.identity.record)
                directory.erase(it);
        }
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
    std::shared_ptr<std::mutex> memory_gate(const std::string &expected) {
        std::lock_guard lock(native_md_mutex);
        auto it = memory_gates.find(expected);
        if (it != memory_gates.end()) return it->second;
        if (memory_gates.size() >= peer_limit) return {};
        return memory_gates.emplace(expected, std::make_shared<std::mutex>()).first->second;
    }
    // The caller holds this peer's memory gate through every owner UCX write.
    // Region/RKEY updates therefore cannot invalidate a live transfer. Metadata
    // includes the complete registered view, so deregistration/re-registration
    // at the same address is safe once the old handles have been released.
    bool import_metadata_locked(const std::string &expected, const std::string &metadata) {
        std::lock_guard lock(native_md_mutex);
        auto cached = imported_metadata.find(expected);
        if (cached != imported_metadata.end() && cached->second == metadata) return true;
        if (cached != imported_metadata.end()) {
            if (native->invalidateRemoteMD(expected) != NIXL_SUCCESS) return false;
            imported_metadata.erase(cached); imported_names.erase(expected);
        }
        if (imported_names.size() >= peer_limit) return false;
        std::string extracted;
        if (native->loadRemoteMD(metadata, extracted) != NIXL_SUCCESS) return false;
        if (extracted != expected) { native->invalidateRemoteMD(extracted); return false; }
        imported_names.insert(expected);
        imported_metadata.emplace(expected, metadata);
        return true;
    }
    bool import_metadata(const std::string &expected, const std::string &metadata) {
        auto gate = memory_gate(expected);
        if (!gate) return false;
        std::lock_guard lock(*gate);
        return import_metadata_locked(expected, metadata);
    }
    // Called with the registration/admission mutex held. The snapshot is small:
    // whole framework pools, never one registration or RKEY per cache page.
    std::string export_caller_metadata(std::uint64_t omit = 0) {
        nixl_reg_dlist_t regions(DRAM_SEG);
        for (const auto &desc : dram_registration) regions.addDesc(desc);
        for (const auto &[token, registration] : registrations)
            if (token != omit && registration.native_desc)
                for (const auto &desc : *registration.native_desc) regions.addDesc(desc);
        auto options = ucx_options; options.includeConnInfo = true;
        std::string result;
        require_nixl(native->getLocalPartialMD(regions, result, &options), "caller DRAM metadata");
        if (result.size() > 4 * 1024 * 1024) throw std::runtime_error("caller metadata limit reached");
        return result;
    }
    std::uint64_t submit(const std::vector<Object> &objects, bool store) {
        if (objects.empty() || objects.size() > batch_limit) throw std::invalid_argument("invalid batch size");
        auto batch = std::make_shared<Batch>(); batch->objects = objects; batch->store = store;
        batch->results.assign(objects.size(), Status::not_ready);
        batch->reported.assign(objects.size(), false);
        batch->submitted = Clock::now();
        if (config.enable_trace && !store) batch->trace.reserve(1 + 3 * objects.size());
        batch->deadline = batch->submitted + std::chrono::milliseconds(config.timeout_ms);
        std::lock_guard lock(mutex);
        if (stopping.load()) throw std::runtime_error("agent closing");
        if (batches.size() >= config.max_inflight) throw std::runtime_error("maximum in-flight handles reached");
        for (const auto &object : objects) {
            if (object.key.empty() || object.key.size() > 32 || object.segments.empty() ||
                object.segments.size() > 65536 || object.hint.size() > 1024)
                throw std::invalid_argument("invalid object");
            (void) domain(object.g3_instance);
            if (object.numa < -1) throw std::invalid_argument("invalid intended NUMA node");
            std::size_t total = 0;
            for (const auto &segment : object.segments) {
                auto r = registrations.find(segment.registration);
                if (r == registrations.end() || !segment.length || segment.offset > r->second.bytes ||
                    segment.length > r->second.bytes - segment.offset ||
                    segment.length > config.staging_slot_bytes - total)
                    throw std::invalid_argument("invalid registered segment");
                total += segment.length;
            }
            if (config.direct_receive && !store && object.segments.size() > 256)
                throw std::invalid_argument("direct receive segment limit reached");
        }
        if (config.direct_receive) {
            using Range = std::pair<std::uintptr_t, std::uintptr_t>;
            auto ranges = [&](const std::vector<Object> &items) {
                std::vector<Range> result;
                for (const auto &object : items) for (const auto &segment : object.segments) {
                    const auto address = registrations.at(segment.registration).address + segment.offset;
                    result.emplace_back(address, address + segment.length);
                }
                std::sort(result.begin(), result.end()); return result;
            };
            const auto target = ranges(objects);
            if (!store) for (std::size_t i = 1; i < target.size(); ++i)
                if (target[i].first < target[i - 1].second)
                    throw std::invalid_argument("direct receive destinations overlap");
            for (const auto &[handle, live] : batches) {
                if (store && live->store) continue;
                const auto other = ranges(live->objects);
                std::size_t a = 0, b = 0;
                while (a < target.size() && b < other.size()) {
                    if (target[a].second <= other[b].first) ++a;
                    else if (other[b].second <= target[a].first) ++b;
                    else throw std::runtime_error("caller region belongs to an unreleased direct transfer");
                }
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
    std::vector<std::pair<std::uintptr_t, std::size_t>> destinations(const Object &object) {
        std::lock_guard lock(mutex);
        std::vector<std::pair<std::uintptr_t, std::size_t>> result;
        for (const auto &segment : object.segments)
            result.emplace_back(registrations.at(segment.registration).address + segment.offset, segment.length);
        return result;
    }
    // The worker is the sole writer; readers use the existing batch gate.
    // Capacity is reserved before admission so diagnostics never grow unbounded.
    void add_trace_locked(Batch &batch, TraceEvent event) {
        if (config.enable_trace && batch.trace.size() < 1 + 3 * batch.objects.size())
            batch.trace.push_back(std::move(event));
    }
    void add_trace(Batch &batch, TraceEvent event) {
        if (!config.enable_trace) return;
        std::lock_guard lock(batch.mutex); add_trace_locked(batch, std::move(event));
    }
    // Caller registrations remain pinned by the live batch until release().
    void copy(const Object &object, void *buffer, bool into_staging,
              Batch *batch = nullptr, std::size_t index = 0) {
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
        const auto end = Clock::now();
        count("staging_copy_bytes", object_bytes(object)); elapsed("staging_copy_ns", start);
        // copy() is called while the cancellation/publication gate is held.
        if (batch && config.enable_trace)
            add_trace_locked(*batch, {"staging_copy", "", clock_ns(start), clock_ns(end),
                                     object_bytes(object), index, 1});
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
    std::vector<G3Buffer> g3_targets(const Object &object) {
        std::vector<G3Buffer> result;
        for (const auto &[target, length] : destinations(object)) result.push_back({target, length});
        return result;
    }
    int intended_numa(const Object &object) {
        return object.numa < 0 ? domain(object.g3_instance).numa : object.numa;
    }
    void g3_metrics(const G3Metrics &metrics, bool write) {
        count(write ? "posix_write_ns" : "posix_read_ns", metrics.payload_ns);
        count(write ? "posix_write_bytes" : "posix_read_bytes", metrics.payload_bytes);
        count("metadata_read_ns", metrics.metadata_read_ns);
        count("metadata_read_bytes", metrics.metadata_read_bytes);
        count("metadata_write_ns", metrics.metadata_write_ns);
        count("metadata_write_bytes", metrics.metadata_write_bytes);
        count("staging_copy_bytes", metrics.copy_bytes); count("staging_copy_ns", metrics.copy_ns);
    }
    Status store_object(const Object &object, std::size_t slot, Clock::time_point deadline, Batch &batch) {
        auto &d = domain(object.g3_instance);
        if (!d.layer) return Status::no_space;
        {
            std::lock_guard lock(batch.mutex);
            if (batch.canceled || Clock::now() >= deadline) return Status::timeout;
            copy(object, address(slot), true);
        }
        auto result = d.layer->write(object.key,
            {{reinterpret_cast<std::uintptr_t>(address(slot)), object_bytes(object)}},
            intended_numa(object), deadline);
        g3_metrics(result.metrics, true);
        if (result.status == Status::success) count("stores");
        return result.status;
    }
    std::string owner_hint(const Object &object) {
        if (!object.hint.empty()) return object.hint;
        std::lock_guard lock(peers_mutex);
        auto it = hints.find(qualified_key(object));
        if (it != hints.end()) return it->second;
        if (config.metadata_endpoint && lookup_queue.size() < 100000) lookup_queue.insert(qualified_key(object));
        maintenance_wake.notify_one();
        return {};
    }
    void refresh_hint(const Object &object) {
        if (!config.metadata_endpoint || !object.hint.empty()) return;
        std::lock_guard lock(peers_mutex);
        hints.erase(qualified_key(object));
        if (lookup_queue.size() < 100000) lookup_queue.insert(qualified_key(object));
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
        w.u8(wire::load); w.str(request); w.str(target_identity); w.str(qualified_key(object));
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
    std::vector<Status> remote_load_group(const std::vector<Object> &objects,
                                        std::size_t first, std::size_t n, std::size_t slot,
                                        const std::string &owner, Clock::time_point deadline,
                                        bool &quarantined, Batch &batch) {
        std::vector<Status> results(n, Status::not_ready);
        auto peer = get_peer(owner);
        auto refresh_group = [&] {
            for (std::size_t i = 0; i < n; ++i) refresh_hint(objects[first + i]);
        };
        if (!peer) { refresh_group(); return results; }
        std::unique_lock peer_lock(peer->mutex, std::try_to_lock);
        if (!peer_lock.owns_lock()) return results;
        if (!peer->channel || !peer->channel->usable()) { refresh_group(); return results; }
        if (Clock::now() >= deadline) return std::vector<Status>(n, Status::timeout);
        auto request = identity + ":" + std::to_string(next_request.fetch_add(1));
        auto target_identity = peer->identity;
        wire::Writer w;
        w.u8(config.direct_receive ?
             (config.enable_trace ? wire::load_scatter_trace : wire::load_scatter) :
             (config.enable_trace ? wire::load_batch_trace : wire::load_batch));
        w.str(request); w.str(target_identity);
        w.str(identity);
        if (config.direct_receive) {
            std::lock_guard lock(mutex); w.str(caller_metadata);
        } else w.str(scratch_metadata);
        w.u64(config.direct_receive ? 0 : reinterpret_cast<std::uintptr_t>(address(slot))); w.u32(n);
        const auto remaining = std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
        w.u32(static_cast<unsigned>(remaining));
        for (std::size_t i = 0; i < n; ++i) {
            w.str(qualified_key(objects[first + i])); w.u64(object_bytes(objects[first + i]));
            if (config.direct_receive) {
                const auto ranges = destinations(objects[first + i]);
                w.u32(ranges.size());
                for (const auto &[target, length] : ranges) { w.u64(target); w.u64(length); }
            }
        }
        auto start = Clock::now();
        bool control_timeout = false;
        TraceEvent event;
        if (config.enable_trace)
            event = {"remote_rpc", request, clock_ns(start), 0, 0, first, n};
        event.direct_receive = config.direct_receive;
        try {
            std::string response;
            try { response = peer->channel->call(w.data, static_cast<unsigned>(remaining)); }
            catch (const wire::Timeout &) { control_timeout = true; throw; }
            if (config.enable_trace) event.end_ns = clock_ns(Clock::now());
            wire::Reader r(response); auto status = read_status(r);
            if (status != Status::success) std::fill(results.begin(), results.end(), status);
            else {
                if (r.u32() != n) throw std::invalid_argument("invalid remote batch response");
                for (auto &result : results) result = read_status(r);
                if (config.enable_trace) {
                    event.owner_timing_flags = r.u8();
                    event.owner_posix_ns = r.u64(); event.owner_ucx_ns = r.u64();
                    event.owner_read_bytes = r.u64();
                    event.owner_metadata_ns = r.u64(); event.owner_metadata_bytes = r.u64();
                    event.owner_staging_copy_ns = r.u64(); event.owner_staging_copy_bytes = r.u64();
                    if ((event.owner_timing_flags & ~15) ||
                        event.owner_posix_ns > event.end_ns - event.start_ns ||
                        event.owner_metadata_ns > event.end_ns - event.start_ns - event.owner_posix_ns ||
                        event.owner_staging_copy_ns > event.end_ns - event.start_ns - event.owner_posix_ns - event.owner_metadata_ns ||
                        event.owner_ucx_ns > event.end_ns - event.start_ns - event.owner_posix_ns - event.owner_metadata_ns - event.owner_staging_copy_ns)
                        throw std::invalid_argument("invalid owner timing response");
                }
            }
            r.finish();
            enqueue_cleanup(owner, target_identity, request);
            elapsed("remote_control_ns", start); count("remote_load_batch_requests");
            for (std::size_t i = 0; i < n; ++i) {
                if (results[i] == Status::success) {
                    count("remote_read_bytes", object_bytes(objects[first + i]));
                    if (config.direct_receive) {
                        count("direct_receive_bytes", object_bytes(objects[first + i]));
                        count("direct_receive_segments", objects[first + i].segments.size());
                        event.destination_segments += objects[first + i].segments.size();
                    }
                }
                else refresh_hint(objects[first + i]);
                if (config.enable_trace && results[i] == Status::success)
                    event.bytes += object_bytes(objects[first + i]);
            }
            if (config.enable_trace) add_trace(batch, std::move(event));
            return results;
        } catch (...) {
            peer->channel.reset();
            if (config.direct_receive) {
                { std::lock_guard lock(batch.mutex); ++batch.pending_direct; batch.canceled = true; }
                std::lock_guard lock(pool_mutex);
                direct_quarantines.push_back({batch.shared_from_this(), owner, target_identity, request});
                count("direct_quarantined_batches");
            } else {
                std::lock_guard lock(pool_mutex);
                quarantines.push_back({slot, owner, target_identity, request});
            }
            quarantined = true;
            if (!config.direct_receive) count("quarantined_slots");
            maintenance_wake.notify_one();
            for (std::size_t i = 0; i < n; ++i) refresh_hint(objects[first + i]);
            elapsed("remote_control_ns", start);
            if (config.enable_trace) {
                event.end_ns = clock_ns(Clock::now()); event.owner_timing_flags = 0;
                add_trace(batch, std::move(event));
            }
            return std::vector<Status>(n, control_timeout || Clock::now() >= deadline ?
                                      Status::timeout : Status::io_error);
        }
    }
    // Group only consecutive same-owner remote objects. Copies remain ordered,
    // including aliased destinations, and use the original batch cancellation gate.
    std::size_t process_remote_group(Batch &batch, std::size_t first, std::size_t &serial_until) {
        if (batch.store || (config.remote_batch_limit == 1 && !config.enable_trace && !config.direct_receive)) return 0;
        std::string owner;
        std::size_t bytes = 0, n = 0;
        for (std::size_t i = first; i < batch.objects.size() && n < config.remote_batch_limit; ++i) {
            const auto &object = batch.objects[i];
            auto &d = domain(object.g3_instance);
            if (d.layer && d.layer->exists(object.key)) break;
            if (n && (object.g3_instance != batch.objects[first].g3_instance ||
                      intended_numa(object) != intended_numa(batch.objects[first]))) break;
            auto hint = owner_hint(object);
            if (hint.empty() || hint == config.name || (n && hint != owner)) break;
            const auto length = object_bytes(object);
            if (length > config.staging_slot_bytes - bytes) break;
            owner = std::move(hint); bytes += length; ++n;
        }
        if (!n || (n < 2 && !config.enable_trace && !config.direct_receive)) return 0;
        std::vector<Status> results(n, Status::busy);
        bool canceled;
        { std::lock_guard lock(batch.mutex); canceled = batch.canceled; }
        if (canceled || Clock::now() >= batch.deadline) std::fill(results.begin(), results.end(), Status::timeout);
        else if (auto slot = config.direct_receive ? std::optional<std::size_t>(0) : acquire_slot()) {
            bool quarantined = false;
            Finally release_slot{[this, slot, &quarantined] { if (!config.direct_receive && !quarantined) free_slot(*slot); }};
            try {
                results = remote_load_group(batch.objects, first, n, *slot, owner, batch.deadline, quarantined, batch);
                // The peer may have a smaller slot or larger allocation padding.
                // A known quiescent no-space reply permits the original path to
                // serve the objects individually without exposing grouped bytes.
                if (std::find(results.begin(), results.end(), Status::no_space) != results.end()) {
                    serial_until = first + n; count("remote_group_fallbacks"); return 0;
                }
            } catch (...) { std::fill(results.begin(), results.end(), Status::io_error); }
            std::size_t offset = 0;
            for (std::size_t i = 0; i < n; ++i) {
                std::lock_guard lock(batch.mutex);
                if (batch.canceled || Clock::now() >= batch.deadline) results[i] = Status::timeout;
                if (results[i] == Status::success && !config.direct_receive) {
                    try { copy(batch.objects[first + i], static_cast<char *>(address(*slot)) + offset, false, &batch, first + i); }
                    catch (...) { results[i] = Status::io_error; }
                }
                // Publish under the copy gate, so a later page's timeout cannot
                // relabel an already copied page as failed.
                if (!batch.reported[first + i]) {
                    batch.results[first + i] = results[i]; batch.reported[first + i] = true;
                }
                offset += object_bytes(batch.objects[first + i]);
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            { std::lock_guard lock(batch.mutex);
              if (!batch.reported[first + i]) {
                  batch.results[first + i] = results[i]; batch.reported[first + i] = true;
              } }
            count(status_name(results[i]));
        }
        return n;
    }
    Status load_object(const Object &object, std::size_t slot, Clock::time_point deadline,
                       bool &quarantined, Batch &batch, std::size_t index, bool &direct_done) {
        auto &d = domain(object.g3_instance);
        Status status = Status::not_found;
        if (d.layer) {
            const auto start = Clock::now();
            auto buffers = config.direct_receive ? g3_targets(object) :
                std::vector<G3Buffer>{{reinterpret_cast<std::uintptr_t>(address(slot)), object_bytes(object)}};
            auto result = d.layer->read(object.key, buffers, intended_numa(object), deadline);
            status = result.status;
            g3_metrics(result.metrics, false);
            if (status == Status::success) {
                direct_done = config.direct_receive; // G3 also scatters a safe padded fallback
                if (config.direct_receive && result.direct) {
                    count("direct_local_read_bytes", result.bytes);
                    count("direct_receive_bytes", result.bytes);
                    count("direct_receive_segments", buffers.size());
                } else if (config.direct_receive) count("local_direct_fallbacks");
                if (config.enable_trace) {
                    TraceEvent event{"local_posix", "", clock_ns(start), clock_ns(Clock::now()),
                                     result.bytes, index, 1};
                    event.owner_timing_flags = 1 | 4 | 8;
                    event.owner_posix_ns = result.metrics.payload_ns;
                    event.owner_read_bytes = result.metrics.payload_bytes;
                    event.owner_metadata_ns = result.metrics.metadata_ns;
                    event.owner_metadata_bytes = result.metrics.metadata_bytes;
                    event.owner_staging_copy_ns = result.metrics.copy_ns;
                    event.owner_staging_copy_bytes = result.metrics.copy_bytes;
                    event.direct_receive = config.direct_receive && result.direct;
                    event.destination_segments = event.direct_receive ? buffers.size() : 0;
                    add_trace(batch, std::move(event));
                }
                return status;
            }
        }
        if (status != Status::not_found) return status;
        if (config.direct_receive) {
            direct_done = true;
            const auto owner = owner_hint(object);
            if (owner.empty()) return config.metadata_endpoint ? Status::not_ready : Status::not_found;
            if (owner == config.name) return Status::not_found;
            return remote_load_group(batch.objects, index, 1, slot, owner, deadline, quarantined, batch).at(0);
        }
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
            if (config.enable_trace && !batch->store)
                add_trace(*batch, {"queue", "", clock_ns(batch->submitted), clock_ns(Clock::now()),
                                   0, 0, batch->objects.size()});
            std::size_t serial_until = 0;
            for (std::size_t i = 0; i < batch->objects.size(); ++i) {
                if (i >= serial_until)
                    if (auto n = process_remote_group(*batch, i, serial_until)) { i += n - 1; continue; }
                Status result = Status::io_error;
                bool canceled;
                { std::lock_guard lock(batch->mutex); canceled = batch->canceled; }
                if (canceled || Clock::now() >= batch->deadline) result = Status::timeout;
                else if (auto slot = acquire_slot()) {
                    bool quarantined = false;
                    Finally release_slot{[this, slot, &quarantined] { if (!quarantined || config.direct_receive) free_slot(*slot); }};
                    try {
                        bool direct_done = false;
                        result = batch->store ? store_object(batch->objects[i], *slot, batch->deadline, *batch) :
                                 load_object(batch->objects[i], *slot, batch->deadline, quarantined, *batch, i, direct_done);
                        std::lock_guard lock(batch->mutex);
                        if (batch->canceled || Clock::now() >= batch->deadline) result = Status::timeout;
                        if (!batch->store && !direct_done && result == Status::success)
                            copy(batch->objects[i], address(*slot), false, batch.get(), i);
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
    std::string serve_load_group(wire::Reader &r, bool trace = false, bool scatter = false) {
        auto id = r.str(4096), expected = r.str(2048);
        auto requester = r.str(2048), metadata = r.str(4 * 1024 * 1024);
        auto target = r.u64(); auto n = r.u32(); auto budget_ms = r.u32();
        if (!n || n > 8 || !budget_ms || id.empty() || requester.empty())
            return status_reply(Status::invalid_input);
        const auto deadline = Clock::now() + std::chrono::milliseconds(std::min(config.timeout_ms, budget_ms));
        struct Page {
            std::string key;
            std::size_t bytes = 0, target_offset = 0, offset = 0;
            Domain *domain = nullptr;
            std::vector<std::pair<std::uintptr_t, std::size_t>> targets;
            Status status = Status::not_found;

        };
        std::vector<Page> pages(n);
        std::size_t logical_bytes = 0;
        for (auto &page : pages) {
            auto qualified = r.str(key_limit);
            auto decoded = decode_key(qualified); page.domain = decoded.first; page.key = std::move(decoded.second);
            auto bytes = r.u64();
            if (page.key.empty() || !bytes)
                return status_reply(Status::invalid_input);
            if (bytes > config.staging_slot_bytes - logical_bytes) return status_reply(Status::no_space);
            page.bytes = static_cast<std::size_t>(bytes); page.target_offset = logical_bytes;
            logical_bytes += page.bytes;
            if (scatter) {
                const auto count = r.u32();
                if (!count || count > 256) return status_reply(Status::invalid_input);
                std::size_t sum = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    const auto address = r.u64(), length = r.u64();
                    if (!address || !length || length > page.bytes - sum ||
                        address > std::numeric_limits<std::uintptr_t>::max() - length)
                        return status_reply(Status::invalid_input);
                    page.targets.emplace_back(address, length); sum += length;
                }
                if (sum != page.bytes) return status_reply(Status::invalid_input);
            }
        }
        r.finish();
        if (scatter) {
            std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
            for (const auto &page : pages) for (const auto &[address, length] : page.targets)
                ranges.emplace_back(address, address + length);
            std::sort(ranges.begin(), ranges.end());
            for (std::size_t i = 1; i < ranges.size(); ++i)
                if (ranges[i].first < ranges[i - 1].second) return status_reply(Status::invalid_input);
        }
        if (target > std::numeric_limits<std::uintptr_t>::max() - logical_bytes)
            return status_reply(Status::invalid_input);
        if (expected != identity || stopping.load()) return status_reply(Status::not_ready);
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

        std::uint8_t timing_flags = 0;
        std::uint64_t read_ns = 0, ucx_ns = 0, physical_read_bytes = 0, metadata_ns = 0, metadata_bytes = 0;
        std::uint64_t copy_ns = 0, copy_bytes = 0;
        auto reply = [&] {
            wire::Writer w; w.u8(static_cast<unsigned>(Status::success)); w.u32(n);
            for (auto &page : pages) w.u8(static_cast<unsigned>(page.status));
            if (trace) {
                w.u8(timing_flags); w.u64(read_ns); w.u64(ucx_ns); w.u64(physical_read_bytes);
                w.u64(metadata_ns); w.u64(metadata_bytes);
                w.u64(copy_ns); w.u64(copy_bytes);
            }
            return w.data;
        };
        Domain *selected = nullptr;
        std::size_t staging_end = 0;
        std::vector<G3Read> reads;
        std::vector<std::size_t> read_indices;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            auto &page = pages[i];
            if (!page.domain || !page.domain->layer) continue;
            if (selected && selected != page.domain) return status_reply(Status::invalid_input);
            selected = page.domain;
            const auto offset = (staging_end + 4095) / 4096 * 4096;
            if (offset > config.staging_slot_bytes || page.bytes > config.staging_slot_bytes - offset)
                return status_reply(Status::no_space);
            page.offset = offset; staging_end = offset + page.bytes;
            reads.push_back({page.key, {{reinterpret_cast<std::uintptr_t>(address(*slot)) + offset, page.bytes}}});
            read_indices.push_back(i); page.status = Status::success;
        }
        if (reads.empty()) return reply();
        const auto read_result = selected->layer->read_batch(reads, selected->numa, deadline);
        g3_metrics(read_result.metrics, false);
        Status io = Status::success;
        for (std::size_t i = 0; i < read_indices.size(); ++i)
            pages[read_indices[i]].status = read_result.objects.at(i).status;
        if (trace) {
            timing_flags |= 1 | 4 | 8; read_ns = read_result.metrics.payload_ns;
            physical_read_bytes = read_result.metrics.payload_bytes;
            metadata_ns = read_result.metrics.metadata_ns; metadata_bytes = read_result.metrics.metadata_bytes;
            copy_ns = read_result.metrics.copy_ns; copy_bytes = read_result.metrics.copy_bytes;
        }
        auto gate = memory_gate(requester);
        if (!gate) {
            for (auto &page : pages) if (page.status == Status::success) page.status = Status::not_ready;
            return reply();
        }
        std::unique_lock memory_lock(*gate);
        if (!import_metadata_locked(requester, metadata)) {
            for (auto &page : pages) if (page.status == Status::success) page.status = Status::not_ready;
            return reply();
        }
        memory_lock.unlock();
        nixl_xfer_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
        std::size_t write_bytes = 0;
        for (auto &page : pages) if (page.status == Status::success) {
            write_bytes += page.bytes;
            if (scatter) {
                std::size_t offset = 0;
                for (const auto &[destination, length] : page.targets) {
                    local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(address(*slot)) + page.offset + offset, length, 0));
                    remote.addDesc(nixlBasicDesc(destination, length, 0)); offset += length;
                }
            } else {
                local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(address(*slot)) + page.offset, page.bytes, 0));
                remote.addDesc(nixlBasicDesc(target + page.target_offset, page.bytes, 0));
            }
        }
        count("remote_served_batch_requests");
        if (local.descCount()) {
            memory_lock.lock();
            if (!import_metadata_locked(requester, metadata)) {
                for (auto &page : pages) if (page.status == Status::success) page.status = Status::not_ready;
                return reply();
            }
            auto write_start = Clock::now();
            io = transfer(NIXL_WRITE, local, remote, requester, ucx_options, deadline);
            if (trace && io == Status::success) {
                timing_flags |= 2; ucx_ns = clock_ns(Clock::now()) - clock_ns(write_start);
            }
            elapsed("ucx_write_ns", write_start);
            if (io == Status::success) count("ucx_write_bytes", write_bytes);
            else for (auto &page : pages) if (page.status == Status::success) page.status = io;
        }
        return reply();
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
                    auto [d, local_key] = decode_key(key);
                    w.u8(d && d->layer && d->layer->exists(local_key));
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
            if (operation == wire::load_batch) return serve_load_group(r);
            if (operation == wire::load_batch_trace) return serve_load_group(r, true);
            if (operation == wire::load_scatter) return serve_load_group(r, false, true);
            if (operation == wire::load_scatter_trace) return serve_load_group(r, true, true);
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
            auto [d, local_key] = decode_key(key);
            if (!d || !d->layer) return status_reply(Status::not_found);
            auto gate = memory_gate(requester);
            if (!gate) return status_reply(Status::not_ready);
            std::unique_lock memory_lock(*gate);
            if (!import_metadata_locked(requester, metadata)) return status_reply(Status::not_ready);
            memory_lock.unlock();
            auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
            auto read = d->layer->read(local_key,
                {{reinterpret_cast<std::uintptr_t>(address(*slot)), static_cast<std::size_t>(bytes)}}, d->numa, deadline);
            g3_metrics(read.metrics, false);
            auto status = read.status;
            if (status != Status::success) return status_reply(status);
            nixl_xfer_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
            local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(address(*slot)), bytes, 0));
            remote.addDesc(nixlBasicDesc(target, bytes, 0));
            memory_lock.lock();
            if (!import_metadata_locked(requester, metadata)) return status_reply(Status::not_ready);
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
        std::vector<DirectQuarantine> direct;
        std::deque<Cleanup> cleanups;
        {
            std::lock_guard lock(pool_mutex);
            pending.swap(quarantines); direct.swap(direct_quarantines); cleanups.swap(cleanup_queue);
        }
        std::vector<Quarantine> retained;
        for (auto &q : pending) {
            if (cleanup({q.owner, q.identity, q.request})) { free_slot(q.slot); count("quarantines_released"); }
            else retained.push_back(q);
        }
        std::vector<DirectQuarantine> retained_direct;
        for (auto &q : direct) {
            if (cleanup({q.owner, q.identity, q.request})) {
                { std::lock_guard lock(q.batch->mutex); --q.batch->pending_direct; }
                q.batch->done.notify_all(); count("direct_quarantines_released");
            } else retained_direct.push_back(std::move(q));
        }
        std::deque<Cleanup> retry;
        for (auto &q : cleanups) { if (!cleanup(q)) retry.push_back(q); }
        {
            std::lock_guard lock(pool_mutex);
            quarantines.insert(quarantines.end(), retained.begin(), retained.end());
            for (auto &q : retained_direct) direct_quarantines.push_back(std::move(q));
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
        auto last_announce = Clock::now() - std::chrono::seconds(2);
        while (!stopping.load()) {
            std::vector<std::shared_ptr<Peer>> known;
            { std::lock_guard lock(peers_mutex); for (auto &[_, p] : peers) known.push_back(p); }
            for (auto &peer : known) { if (stopping.load()) break; connect_peer(peer); }
            reclaim_quarantines();
            auto now = Clock::now();
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
                        {
                            std::lock_guard lock(directory_mutex);
                            for (const auto &[key, identity] : directory) { (void) identity; keys.push_back(key); }
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
    std::vector<bool> exists(const std::vector<std::string> &keys, const std::vector<std::string> &owners,
                             const std::string &instance) {
        if (keys.size() > batch_limit || (!owners.empty() && keys.size() != owners.size()))
            throw std::invalid_argument("invalid exists batch");
        if (stopping.load()) throw std::runtime_error("agent closing");
        const auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
        auto &d = domain(instance);
        std::vector<bool> results(keys.size(), false);
        std::map<std::string, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].empty() || keys[i].size() > 32) throw std::invalid_argument("invalid key");
            results[i] = d.layer && d.layer->exists(keys[i]);
            if (results[i]) continue;
            Object object; object.key = keys[i]; object.g3_instance = instance;
            if (!owners.empty()) object.hint = owners[i];
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
                for (auto i : indices) w.str(qualified_key(d.namespace_id, keys[i]));
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
    void close(CloseMode mode = CloseMode::clean) {
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
            if (!quarantines.empty() || !direct_quarantines.empty())
                throw std::runtime_error("remote I/O not quiescent; registered destinations retained");
        }
        bool failed = false;
        for (auto &d : domains) if (d.layer) {
            auto result = d.layer->close(mode);
            if (mode == CloseMode::clean && !result.clean) failed = true;
            for (const auto &device : result.devices) if (device.status != Status::success) {
                count("close_errors"); failed = true;
            }
        }
        closed.store(true);
        if (failed) throw std::runtime_error("G3 close failed; failed devices remain DIRTY");
    }
};

Agent::Agent(const AgentConfig &config) : impl_(std::make_unique<Impl>(config)) {}
Agent::~Agent() {
    if (!impl_) return;
    try { impl_->close(CloseMode::discard); }
    catch (...) {
        // Unconfirmed remote writes may still hold this destination address.
        // A bounded retained pool is safer than freeing/reusing registered memory.
        if (!impl_->closed.load()) (void) impl_.release();
    }
}
std::uint64_t Agent::register_memory(std::uintptr_t address, std::size_t bytes) {
    if (!address || !bytes || address > std::numeric_limits<std::uintptr_t>::max() - bytes)
        throw std::invalid_argument("invalid caller memory");
    std::lock_guard lock(impl_->mutex);
    if (impl_->stopping.load()) throw std::runtime_error("agent closing");
    if (impl_->registrations.size() >= 65536) throw std::runtime_error("registration limit reached");
    for (const auto &[existing, registration] : impl_->registrations)
        if (address < registration.address + registration.bytes && registration.address < address + bytes)
            throw std::invalid_argument("native DRAM registrations overlap");
    auto token = impl_->next_token++;
    if (!token) throw std::overflow_error("registration sequence exhausted");
    std::unique_ptr<nixl_reg_dlist_t> descriptor;
    {
        descriptor = std::make_unique<nixl_reg_dlist_t>(DRAM_SEG);
        descriptor->addDesc(nixlBlobDesc(address, bytes, 0, ""));
        require_nixl(impl_->native->registerMem(*descriptor, &impl_->dram_options), "caller DRAM registration");
    }
    Impl::Registration registration{address, bytes, 0, std::move(descriptor), {}};
    try {
        for (auto &d : impl_->domains) if (d.layer)
            registration.g3_registrations.emplace_back(d.layer.get(),
                d.layer->borrow_registered_memory({{address, bytes}}));
        impl_->registrations.emplace(token, std::move(registration));
    } catch (...) {
        for (auto &[layer, handle] : registration.g3_registrations) layer->deregister_memory(handle);
        if (registration.native_desc) impl_->native->deregisterMem(*registration.native_desc, &impl_->dram_options);
        throw;
    }
    if (impl_->config.direct_receive) {
        try { impl_->caller_metadata = impl_->export_caller_metadata(); }
        catch (...) {
            auto &registration = impl_->registrations.at(token);
            for (auto &[layer, handle] : registration.g3_registrations) layer->deregister_memory(handle);
            impl_->native->deregisterMem(*registration.native_desc, &impl_->dram_options);
            impl_->registrations.erase(token); throw;
        }
    }
    return token;
}
void Agent::deregister_memory(std::uint64_t token) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->registrations.find(token);
    if (it == impl_->registrations.end()) throw std::invalid_argument("unknown registration");
    if (it->second.refs) throw std::runtime_error("registration belongs to unreleased transfers");
    if (!impl_->closed.load()) for (auto &[layer, handle] : it->second.g3_registrations)
        if (layer->deregister_memory(handle) != Status::success) throw std::runtime_error("G3 registration did not drain");
    if (it->second.native_desc) {
        auto metadata = impl_->export_caller_metadata(token);
        require_nixl(impl_->native->deregisterMem(*it->second.native_desc, &impl_->dram_options), "caller DRAM deregistration");
        impl_->caller_metadata = std::move(metadata);
    }
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
std::vector<TraceEvent> Agent::trace(std::uint64_t handle) const {
    std::shared_ptr<Impl::Batch> batch;
    { std::lock_guard lock(impl_->mutex); auto it = impl_->batches.find(handle);
      if (it == impl_->batches.end()) throw std::invalid_argument("unknown transfer handle");
      batch = it->second; }
    std::lock_guard lock(batch->mutex);
    return batch->trace;
}
bool Agent::is_quiescent(std::uint64_t handle) const {
    std::shared_ptr<Impl::Batch> batch;
    { std::lock_guard lock(impl_->mutex); auto it = impl_->batches.find(handle);
      if (it == impl_->batches.end()) throw std::invalid_argument("unknown transfer handle");
      batch = it->second; }
    std::lock_guard lock(batch->mutex);
    return batch->finished && !batch->pending_direct;
}
void Agent::release(std::uint64_t handle) {
    std::shared_ptr<Impl::Batch> batch;
    { std::lock_guard lock(impl_->mutex); auto it = impl_->batches.find(handle);
      if (it == impl_->batches.end()) throw std::invalid_argument("unknown transfer handle");
      batch = it->second; }
    { std::unique_lock lock(batch->mutex);
      if (impl_->config.direct_receive && (!batch->finished || batch->pending_direct))
          throw std::runtime_error("direct receive not quiescent; handle and caller regions retained");
      batch->done.wait(lock, [&batch] { return batch->finished; });
      if (batch->pending_direct) throw std::runtime_error("direct receive not quiescent; handle and caller regions retained"); }
    std::lock_guard lock(impl_->mutex);
    if (!impl_->batches.erase(handle)) throw std::invalid_argument("transfer already released");
    for (const auto &o : batch->objects) for (const auto &s : o.segments)
        --impl_->registrations.at(s.registration).refs;
}
std::vector<bool> Agent::batch_exists(const std::vector<std::string> &keys, const std::vector<std::string> &hints,
                                     const std::string &g3_instance) {
    return impl_->exists(keys, hints, g3_instance);
}
Status Agent::checkpoint() {
    if (impl_->stopping.load()) return Status::not_ready;
    Status result = Status::success;
    auto start = Clock::now();
    for (auto &d : impl_->domains) if (d.layer) {
        auto s = d.layer->checkpoint(); if (s != Status::success) result = s;
    }
    impl_->elapsed("metadata_checkpoint_ns", start); return result;
}
Endpoint Agent::endpoint() const { return impl_->server->endpoint(); }
std::map<std::string, std::uint64_t> Agent::stats() const {
    auto *p = impl_.get();
    std::map<std::string, std::uint64_t> result;
    { std::lock_guard lock(p->stats_mutex); result = p->counters; }
    { std::lock_guard lock(p->pool_mutex); result["staging_free_slots"] = p->free_slots.size();
      result["staging_quarantined_slots"] = p->quarantines.size();
      result["direct_quarantined_handles"] = p->direct_quarantines.size(); }
    return result;
}
void Agent::close(CloseMode mode) { impl_->close(mode); }
} // namespace nixlshard
