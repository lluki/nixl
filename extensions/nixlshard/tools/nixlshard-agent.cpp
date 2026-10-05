/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "cli.h"
#include <iostream>

using namespace nixlshard;
int main(int argc, char **argv) {
    try {
        AgentConfig cfg;
        DiskConfig disk;
        std::vector<std::string> paths;
        Endpoint metadata;
        bool have_metadata = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "nixlshard-agent --name NAME [--listen-host HOST] [--listen-port PORT]\n"
                             "  [--disk PATH ...] [--capacity-bytes N] [--unit-bytes N]\n"
                             "  [--metadata-bytes N] [--create] [--direct-io]\n"
                             "  [--namespace SCHEMA] [--numa-node N] [--min-object-bytes N --max-object-bytes N]\n"
                             "  [--key-bytes N] [--metadata-alignment N] [--reset] [--registration-mode EXPLICIT|AUTOMATIC]\n"
                             "  [--metadata-host HOST --metadata-port PORT] [--peer NAME HOST PORT]\n"
                             "  [--staging-slots N] [--staging-slot-bytes N] [--workers N]\n"
                             "  [--max-inflight N] [--timeout-ms N]\n";
                return 0;
            }
            if (arg == "--name") cfg.name = cli::value(i, argc, argv);
            else if (arg == "--listen-host") cfg.listen.host = cli::value(i, argc, argv);
            else if (arg == "--listen-port") cfg.listen.port = cli::integer<std::uint16_t>(cli::value(i, argc, argv));
            else if (arg == "--disk") paths.push_back(cli::value(i, argc, argv));
            else if (arg == "--capacity-bytes") disk.capacity_bytes = cli::integer<std::uint64_t>(cli::value(i, argc, argv));
            else if (arg == "--unit-bytes") disk.unit_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--metadata-bytes") disk.metadata_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--namespace") cfg.namespace_id = cli::value(i, argc, argv);
            else if (arg == "--numa-node") cfg.numa_node = cli::integer<unsigned>(cli::value(i, argc, argv));
            else if (arg == "--min-object-bytes") disk.min_object_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--max-object-bytes") disk.max_object_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--key-bytes") disk.key_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--metadata-alignment") disk.metadata_alignment = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--reset") disk.reset = true;
            else if (arg == "--registration-mode") {
                auto mode = cli::value(i, argc, argv);
                if (mode == "EXPLICIT") cfg.registration_mode = MemoryMode::explicit_registration;
                else if (mode == "AUTOMATIC") cfg.registration_mode = MemoryMode::automatic;
                else throw std::invalid_argument("invalid registration mode");
            }
            else if (arg == "--create") disk.create = true;
            else if (arg == "--direct-io") cfg.direct_io = true;
            else if (arg == "--metadata-host") { metadata.host = cli::value(i, argc, argv); have_metadata = true; }
            else if (arg == "--metadata-port") { metadata.port = cli::integer<std::uint16_t>(cli::value(i, argc, argv)); have_metadata = true; }
            else if (arg == "--peer") {
                auto name = cli::value(i, argc, argv);
                Endpoint peer;
                peer.host = cli::value(i, argc, argv);
                peer.port = cli::integer<std::uint16_t>(cli::value(i, argc, argv));
                if (!cfg.peers.emplace(name, peer).second) throw std::invalid_argument("duplicate peer name");
            }
            else if (arg == "--staging-slots") cfg.staging_slots = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--staging-slot-bytes") cfg.staging_slot_bytes = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--workers") cfg.workers = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--max-inflight") cfg.max_inflight = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--timeout-ms") cfg.timeout_ms = cli::integer<unsigned>(cli::value(i, argc, argv));
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (cfg.name.empty()) throw std::invalid_argument("--name is required");
        if (have_metadata) {
            if (!metadata.port) throw std::invalid_argument("--metadata-port must be nonzero");
            cfg.metadata_endpoint = metadata;
        }
        for (auto &path : paths) { disk.path = path; cfg.disks.push_back(disk); }
        Agent agent(cfg);
        auto bound = agent.endpoint();
        std::cout << "nixlshard-agent " << cfg.name << " listening "
                  << bound.host << ':' << bound.port << std::endl;
        cli::wait();
        agent.close();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "nixlshard-agent: " << error.what() << '\n';
        return 1;
    }
}
