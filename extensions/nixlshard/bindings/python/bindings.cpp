/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nixlshard;

namespace {
template <typename T>
void option(const py::dict &source, const char *key, T &value) {
    if (source.contains(key)) value = py::cast<T>(source[key]);
}
Endpoint endpoint(const py::dict &source) {
    Endpoint result;
    option(source, "host", result.host);
    option(source, "port", result.port);
    return result;
}
py::dict endpoint_dict(const Endpoint &value) {
    py::dict result;
    result["host"] = value.host;
    result["port"] = value.port;
    return result;
}
G3DeviceConfig device_config(const py::dict &disk, int numa) {
    G3DeviceConfig result; result.numa_node = numa;
    auto &value = result.disk;
    value.path = py::cast<std::string>(disk["path"]);
    option(disk, "capacity_bytes", value.capacity_bytes);
    option(disk, "unit_bytes", value.unit_bytes);
    option(disk, "metadata_bytes", value.metadata_bytes);
    option(disk, "create", value.create);
    option(disk, "min_object_bytes", value.min_object_bytes);
    option(disk, "max_object_bytes", value.max_object_bytes);
    option(disk, "key_bytes", value.key_bytes);
    option(disk, "metadata_alignment", value.metadata_alignment);
    option(disk, "namespace_id", value.namespace_id);
    option(disk, "reset", value.reset);
    option(disk, "numa_node", result.numa_node);
    return result;
}
std::string binary_key(const py::handle &key) {
    if (!py::isinstance<py::str>(key) && !py::isinstance<py::bytes>(key))
        throw py::type_error("key requires bytes or a UTF-8 string");
    return py::cast<std::string>(key); // preserves NUL and non-UTF8 octets in bytes
}
AgentConfig config(const py::dict &source) {
    AgentConfig result;
    result.name = py::cast<std::string>(source["name"]);
    option(source, "listen_host", result.listen.host);
    option(source, "listen_port", result.listen.port);
    option(source, "max_inflight", result.max_inflight);
    option(source, "workers", result.workers);
    option(source, "staging_slots", result.staging_slots);
    option(source, "staging_slot_bytes", result.staging_slot_bytes);
    option(source, "remote_batch_limit", result.remote_batch_limit);
    option(source, "enable_trace", result.enable_trace);
    option(source, "direct_receive", result.direct_receive);
    option(source, "timeout_ms", result.timeout_ms);
    option(source, "direct_io", result.direct_io);
    option(source, "namespace_id", result.namespace_id);
    option(source, "g3_instance", result.g3_instance);
    option(source, "numa_node", result.numa_node);
    if (source.contains("registration_mode")) {
        auto mode = py::cast<std::string>(source["registration_mode"]);
        if (mode == "EXPLICIT") result.registration_mode = MemoryMode::explicit_registration;
        else if (mode == "AUTOMATIC") result.registration_mode = MemoryMode::automatic;
        else throw py::value_error("registration_mode requires EXPLICIT or AUTOMATIC");
    }
    if (source.contains("metadata_endpoint") && !source["metadata_endpoint"].is_none())
        result.metadata_endpoint = endpoint(py::cast<py::dict>(source["metadata_endpoint"]));
    if (source.contains("peers")) {
        for (auto item : py::cast<py::dict>(source["peers"]))
            result.peers.emplace(py::cast<std::string>(item.first),
                                 endpoint(py::cast<py::dict>(item.second)));
    }
    if (source.contains("disks")) {
        for (auto item : py::cast<py::iterable>(source["disks"])) {
            auto disk = py::cast<py::dict>(item);
            auto device = device_config(disk, result.numa_node);
            result.disk_numa_nodes[device.disk.path] = device.numa_node;
            result.disks.push_back(std::move(device.disk));
        }
    }
    if (source.contains("g3_instances")) {
        if (!result.disks.empty()) throw py::value_error("use disks or g3_instances, not both");
        for (auto item : py::cast<py::iterable>(source["g3_instances"])) {
            auto input = py::cast<py::dict>(item); G3InstanceConfig instance;
            option(input, "name", instance.name); option(input, "namespace_id", instance.namespace_id);
            option(input, "numa_node", instance.numa_node);
            if (input.contains("disks")) for (auto disk : py::cast<py::iterable>(input["disks"]))
                instance.devices.push_back(device_config(py::cast<py::dict>(disk), instance.numa_node));
            result.g3_instances.push_back(std::move(instance));
        }
        if (!source.contains("g3_instance") && !result.g3_instances.empty())
            result.g3_instance = result.g3_instances.front().name;
    }
    return result;
}
std::vector<Object> objects(const py::iterable &source) {
    std::vector<Object> result;
    for (auto item : source) {
        auto input = py::cast<py::dict>(item);
        Object value;
        value.key = binary_key(input["key"]);
        option(input, "hint", value.hint);
        option(input, "g3_instance", value.g3_instance);
        option(input, "numa", value.numa);
        for (auto segment : py::cast<py::iterable>(input["segments"])) {
            auto fields = py::cast<py::sequence>(segment);
            if (py::len(fields) != 3)
                throw py::value_error("segment requires (registration, offset, length)");
            value.segments.push_back({py::cast<std::uint64_t>(fields[0]),
                                      py::cast<std::size_t>(fields[1]),
                                      py::cast<std::size_t>(fields[2])});
        }
        result.push_back(std::move(value));
    }
    return result;
}
} // namespace

PYBIND11_MODULE(_bindings, module) {
    module.doc() = "Native asynchronous NIXLShard cache interface";
    module.attr("build_git") = NIXLSHARD_BUILD_GIT;
    module.attr("direct_receive_supported") = true;
    module.attr("authoritative_g3_supported") = true;
    py::class_<Agent>(module, "Agent")
        .def_property_readonly_static("direct_receive_supported", [](py::object) { return true; })
        .def_property_readonly_static("authoritative_g3_supported", [](py::object) { return true; })
        .def(py::init([](const py::dict &value) {
            auto parsed = config(value);
            py::gil_scoped_release unlocked;
            return std::make_unique<Agent>(parsed);
        }), py::arg("config"))
        .def("register_memory", &Agent::register_memory,
             py::arg("address"), py::arg("length"), py::call_guard<py::gil_scoped_release>())
        .def("deregister_memory", &Agent::deregister_memory,
             py::arg("token"), py::call_guard<py::gil_scoped_release>())
        .def("batch_store", [](Agent &agent, const py::iterable &items) {
            auto parsed = objects(items);
            py::gil_scoped_release unlocked;
            return agent.batch_store(parsed);
        }, py::arg("objects"))
        .def("batch_load", [](Agent &agent, const py::iterable &items) {
            auto parsed = objects(items);
            py::gil_scoped_release unlocked;
            return agent.batch_load(parsed);
        }, py::arg("objects"))
        .def("poll", [](const Agent &agent, std::uint64_t handle) -> py::object {
            std::optional<std::vector<Status>> statuses;
            { py::gil_scoped_release unlocked; statuses = agent.poll(handle); }
            if (!statuses) return py::none();
            py::list result;
            for (auto status : *statuses) result.append(status_name(status));
            return std::move(result);
        }, py::arg("handle"))
        .def("trace", [](const Agent &agent, std::uint64_t handle) {
            std::vector<TraceEvent> events;
            { py::gil_scoped_release unlocked; events = agent.trace(handle); }
            py::list result;
            for (const auto &event : events) {
                py::dict row;
                row["stage"] = event.stage; row["request_id"] = event.request_id;
                row["start_ns"] = event.start_ns; row["end_ns"] = event.end_ns;
                row["bytes"] = event.bytes; row["first_object"] = event.first_object;
                row["object_count"] = event.object_count;
                if (event.owner_timing_flags & 1) {
                    row["owner_posix_ns"] = event.owner_posix_ns;
                    row["owner_read_bytes"] = event.owner_read_bytes;
                }
                if (event.owner_timing_flags & 2) row["owner_ucx_ns"] = event.owner_ucx_ns;
                if (event.owner_timing_flags & 4) {
                    row["owner_metadata_ns"] = event.owner_metadata_ns;
                    row["owner_metadata_bytes"] = event.owner_metadata_bytes;
                }
                if (event.owner_timing_flags & 8) {
                    row["owner_staging_copy_ns"] = event.owner_staging_copy_ns;
                    row["owner_staging_copy_bytes"] = event.owner_staging_copy_bytes;
                }
                if (event.direct_receive) {
                    row["direct_receive"] = true;
                    row["destination_segments"] = event.destination_segments;
                }
                result.append(row);
            }
            return result;
        }, py::arg("handle"))
        .def("is_quiescent", &Agent::is_quiescent, py::arg("handle"),
             py::call_guard<py::gil_scoped_release>())
        .def("release", &Agent::release, py::arg("handle"),
             py::call_guard<py::gil_scoped_release>())
        .def("batch_exists", [](Agent &agent, const py::iterable &keys,
                                const py::object &hints, const std::string &instance) {
            std::vector<std::string> parsed_keys;
            for (auto key : keys) parsed_keys.push_back(binary_key(key));
            auto parsed = hints.is_none() ? std::vector<std::string>{}
                                         : hints.cast<std::vector<std::string>>();
            py::gil_scoped_release unlocked;
            return agent.batch_exists(parsed_keys, parsed, instance);
        }, py::arg("keys"), py::arg("hints") = py::none(), py::arg("g3_instance") = "")
        .def("checkpoint", [](Agent &agent) {
            Status status;
            { py::gil_scoped_release unlocked; status = agent.checkpoint(); }
            return status_name(status);
        })
        .def("endpoint", [](const Agent &agent) { return endpoint_dict(agent.endpoint()); })
        .def("stats", &Agent::stats, py::call_guard<py::gil_scoped_release>())
        .def("close", [](Agent &agent, const std::string &mode) {
            if (mode != "CLEAN" && mode != "DISCARD") throw py::value_error("close mode requires CLEAN or DISCARD");
            py::gil_scoped_release unlocked;
            agent.close(mode == "CLEAN" ? CloseMode::clean : CloseMode::discard);
        }, py::arg("mode") = "CLEAN")
        .def("__enter__", [](Agent &agent) -> Agent & { return agent; },
             py::return_value_policy::reference_internal)
        .def("__exit__", [](Agent &agent, py::object, py::object, py::object) {
            py::gil_scoped_release unlocked;
            agent.close();
        });
    py::class_<MetadataServer>(module, "MetadataServer")
        .def(py::init([](const py::dict &value, std::size_t max_entries, unsigned ttl_ms) {
            auto parsed = endpoint(value);
            py::gil_scoped_release unlocked;
            return std::make_unique<MetadataServer>(parsed, max_entries, ttl_ms);
        }), py::arg("endpoint"), py::arg("max_entries") = 100000, py::arg("ttl_ms") = 60000)
        .def("endpoint", [](const MetadataServer &server) { return endpoint_dict(server.endpoint()); })
        .def("close", &MetadataServer::close, py::call_guard<py::gil_scoped_release>())
        .def("__enter__", [](MetadataServer &server) -> MetadataServer & { return server; },
             py::return_value_policy::reference_internal)
        .def("__exit__", [](MetadataServer &server, py::object, py::object, py::object) {
            py::gil_scoped_release unlocked;
            server.close();
        });
}
