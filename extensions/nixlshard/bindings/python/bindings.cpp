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
AgentConfig config(const py::dict &source) {
    AgentConfig result;
    result.name = py::cast<std::string>(source["name"]);
    option(source, "listen_host", result.listen.host);
    option(source, "listen_port", result.listen.port);
    option(source, "max_inflight", result.max_inflight);
    option(source, "workers", result.workers);
    option(source, "staging_slots", result.staging_slots);
    option(source, "staging_slot_bytes", result.staging_slot_bytes);
    option(source, "timeout_ms", result.timeout_ms);
    option(source, "direct_io", result.direct_io);
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
            DiskConfig value;
            value.path = py::cast<std::string>(disk["path"]);
            option(disk, "capacity_bytes", value.capacity_bytes);
            option(disk, "unit_bytes", value.unit_bytes);
            option(disk, "metadata_bytes", value.metadata_bytes);
            option(disk, "create", value.create);
            result.disks.push_back(std::move(value));
        }
    }
    return result;
}
std::vector<Object> objects(const py::iterable &source) {
    std::vector<Object> result;
    for (auto item : source) {
        auto input = py::cast<py::dict>(item);
        Object value;
        value.key = py::cast<std::string>(input["key"]);
        option(input, "hint", value.hint);
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
    py::class_<Agent>(module, "Agent")
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
        .def("release", &Agent::release, py::arg("handle"),
             py::call_guard<py::gil_scoped_release>())
        .def("batch_exists", [](Agent &agent, const std::vector<std::string> &keys,
                                const py::object &hints) {
            auto parsed = hints.is_none() ? std::vector<std::string>{}
                                         : hints.cast<std::vector<std::string>>();
            py::gil_scoped_release unlocked;
            return agent.batch_exists(keys, parsed);
        }, py::arg("keys"), py::arg("hints") = py::none())
        .def("checkpoint", [](Agent &agent) {
            Status status;
            { py::gil_scoped_release unlocked; status = agent.checkpoint(); }
            return status_name(status);
        })
        .def("endpoint", [](const Agent &agent) { return endpoint_dict(agent.endpoint()); })
        .def("stats", &Agent::stats, py::call_guard<py::gil_scoped_release>())
        .def("close", &Agent::close, py::call_guard<py::gil_scoped_release>())
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
