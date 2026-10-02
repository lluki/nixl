/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "nixlshard/wire.h"

#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nixlshard {
namespace {
constexpr size_t max_batch = 128;
constexpr size_t max_key = 65536;
constexpr size_t max_identity = 1024;
constexpr size_t max_host = 4096;
using Clock = std::chrono::steady_clock;
bool valid_identity(const std::string &value) {
    return !value.empty() && value.find('\0') == std::string::npos;
}
std::string status_response(Status status) {
    wire::Writer out;
    out.u8(static_cast<uint8_t>(status));
    return out.data;
}
std::vector<std::string> read_keys(wire::Reader &reader) {
    auto count = reader.u32();
    if (count > max_batch) throw std::invalid_argument("metadata batch too large");
    std::vector<std::string> keys;
    keys.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto key = reader.str(max_key);
        if (key.empty()) throw std::invalid_argument("empty metadata key");
        keys.push_back(std::move(key));
    }
    return keys;
}
} // namespace

struct MetadataServer::Impl {
    struct Owner {
        std::string incarnation;
        Endpoint endpoint;
        Clock::time_point expires;
        uint64_t sequence = 0;
        bool has_sequence = false;
        std::unordered_set<std::string> keys;
        std::unordered_set<std::string> retired_incarnations;
    };
    struct Hint { std::string owner, incarnation; };
    size_t limit, retired_count = 0;
    std::chrono::milliseconds ttl;
    std::mutex mutex;
    std::unordered_map<std::string, Owner> owners;
    std::unordered_map<std::string, Hint> hints;
    std::unique_ptr<wire::Server> server;

    Impl(Endpoint endpoint, size_t max_entries, unsigned ttl_ms)
        : limit(max_entries), ttl(ttl_ms) {
        if (!limit || !ttl_ms) throw std::invalid_argument("metadata limits must be positive");
        server = std::make_unique<wire::Server>(std::move(endpoint),
            [this](std::string_view request) { return handle(request); }, 5000, 16);
    }
    void remove_hints(const std::string &name, Owner &owner) {
        for (const auto &key : owner.keys) {
            auto it = hints.find(key);
            if (it != hints.end() && it->second.owner == name) hints.erase(it);
        }
        owner.keys.clear();
    }
    void expire(std::unordered_map<std::string, Owner>::iterator owner) {
        remove_hints(owner->first, owner->second);
        retired_count -= owner->second.retired_incarnations.size();
        owners.erase(owner);
    }
    void prune(Clock::time_point now) {
        for (auto it = owners.begin(); it != owners.end();) {
            if (it->second.expires <= now) { auto old = it++; expire(old); }
            else ++it;
        }
    }
    Status register_owner(const std::string &name, const std::string &incarnation,
                          const Endpoint &endpoint) {
        std::lock_guard<std::mutex> lock(mutex);
        auto now = Clock::now();
        auto found = owners.find(name);
        if (found != owners.end() && found->second.expires <= now) {
            expire(found); found = owners.end();
        }
        if (found == owners.end()) {
            if (owners.size() >= limit) prune(now);
            if (owners.size() >= limit) return Status::no_space;
            Owner owner;
            owner.incarnation = incarnation; owner.endpoint = endpoint; owner.expires = now + ttl;
            owners.emplace(name, std::move(owner));
            return Status::success;
        }
        auto &owner = found->second;
        if (owner.incarnation != incarnation) {
            if (owner.retired_incarnations.count(incarnation)) return Status::not_found;
            // Tombstones fence delayed heartbeats from known previous processes.
            // Both their total count and owner/hint counts have explicit limits.
            if (retired_count >= limit) {
                prune(now);
                if (retired_count >= limit) return Status::busy;
            }
            owner.retired_incarnations.insert(owner.incarnation);
            ++retired_count;
            remove_hints(name, owner);
            owner.incarnation = incarnation;
            owner.has_sequence = false;
            owner.sequence = 0;
        }
        owner.endpoint = endpoint;
        owner.expires = now + ttl;
        return Status::success;
    }
    Status announce(const std::string &name, const std::string &incarnation,
                    uint64_t sequence, const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex);
        auto now = Clock::now();
        auto found = owners.find(name);
        if (found == owners.end()) return Status::not_found;
        if (found->second.expires <= now) { expire(found); return Status::not_found; }
        auto &owner = found->second;
        if (owner.incarnation != incarnation) return Status::not_found;
        if (owner.has_sequence && sequence <= owner.sequence) return Status::success;
        std::unordered_set<std::string> unique(keys.begin(), keys.end());
        size_t new_entries = 0;
        for (const auto &key : unique) if (!hints.count(key)) ++new_entries;
        if (new_entries > limit - hints.size()) {
            prune(now);
            new_entries = 0;
            for (const auto &key : unique) if (!hints.count(key)) ++new_entries;
            if (new_entries > limit - hints.size()) return Status::no_space;
        }
        for (const auto &key : unique) {
            auto hint = hints.find(key);
            if (hint != hints.end() && hint->second.owner != name) {
                auto previous = owners.find(hint->second.owner);
                if (previous != owners.end()) previous->second.keys.erase(key);
            }
            hints[key] = Hint{name, incarnation};
            owner.keys.insert(key);
        }
        owner.sequence = sequence; owner.has_sequence = true;
        return Status::success;
    }
    std::string lookup(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex);
        auto now = Clock::now();
        wire::Writer out;
        out.u8(static_cast<uint8_t>(Status::success));
        out.u32(static_cast<uint32_t>(keys.size()));
        for (const auto &key : keys) {
            auto found = hints.find(key);
            bool present = false;
            if (found != hints.end()) {
                auto owner = owners.find(found->second.owner);
                if (owner != owners.end() && owner->second.expires > now &&
                    owner->second.incarnation == found->second.incarnation) present = true;
            }
            if (present) { out.str(found->second.owner); out.str(found->second.incarnation); }
            else { out.str(""); out.str(""); }
        }
        return out.data;
    }
    std::string directory(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = owners.find(name);
        if (found == owners.end()) return status_response(Status::not_found);
        if (found->second.expires <= Clock::now()) { expire(found); return status_response(Status::not_found); }
        wire::Writer out;
        out.u8(static_cast<uint8_t>(Status::success));
        out.str(found->second.incarnation);
        out.str(found->second.endpoint.host);
        out.u32(found->second.endpoint.port);
        return out.data;
    }
    std::string handle(std::string_view message) {
        try {
            wire::Reader reader(message);
            auto operation = reader.u8();
            if (operation == wire::register_owner) {
                auto name = reader.str(max_identity), incarnation = reader.str(max_identity), host = reader.str(max_host);
                auto port = reader.u32();
                reader.finish();
                if (!valid_identity(name) || !valid_identity(incarnation) || !valid_identity(host) ||
                    !port || port > std::numeric_limits<uint16_t>::max()) return status_response(Status::invalid_input);
                return status_response(register_owner(name, incarnation, Endpoint{host, static_cast<uint16_t>(port)}));
            }
            if (operation == wire::announce) {
                auto name = reader.str(max_identity), incarnation = reader.str(max_identity);
                auto sequence = reader.u64();
                auto keys = read_keys(reader);
                reader.finish();
                if (!valid_identity(name) || !valid_identity(incarnation)) return status_response(Status::invalid_input);
                return status_response(announce(name, incarnation, sequence, keys));
            }
            if (operation == wire::lookup) {
                auto keys = read_keys(reader);
                reader.finish();
                return lookup(keys);
            }
            if (operation == wire::directory) {
                auto name = reader.str(max_identity);
                reader.finish();
                if (!valid_identity(name)) return status_response(Status::invalid_input);
                return directory(name);
            }
            return status_response(Status::invalid_input);
        } catch (const std::invalid_argument &) {
            return status_response(Status::invalid_input);
        } catch (const std::bad_alloc &) {
            return status_response(Status::no_space);
        }
    }
};

MetadataServer::MetadataServer(Endpoint endpoint, size_t max_entries, unsigned ttl_ms)
    : impl_(std::make_unique<Impl>(std::move(endpoint), max_entries, ttl_ms)) {}
MetadataServer::~MetadataServer() = default;
Endpoint MetadataServer::endpoint() const { return impl_->server->endpoint(); }
void MetadataServer::close() { impl_->server->close(); }
} // namespace nixlshard
