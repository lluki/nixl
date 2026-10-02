/* SPDX-License-Identifier: Apache-2.0 */
#include "nixlshard/agent.h"
#include "cli.h"
#include <iostream>

using namespace nixlshard;
int main(int argc, char **argv) {
    try {
        Endpoint listen;
        std::size_t max_entries = 100000;
        unsigned ttl_ms = 60000;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "nixlshard-metadata [--listen-host HOST] [--listen-port PORT]\n"
                             "  [--max-entries N] [--ttl-ms N]\n";
                return 0;
            }
            if (arg == "--listen-host") listen.host = cli::value(i, argc, argv);
            else if (arg == "--listen-port") listen.port = cli::integer<std::uint16_t>(cli::value(i, argc, argv));
            else if (arg == "--max-entries") max_entries = cli::integer<std::size_t>(cli::value(i, argc, argv));
            else if (arg == "--ttl-ms") ttl_ms = cli::integer<unsigned>(cli::value(i, argc, argv));
            else throw std::invalid_argument("unknown option: " + arg);
        }
        MetadataServer server(listen, max_entries, ttl_ms);
        auto bound = server.endpoint();
        std::cout << "nixlshard-metadata listening " << bound.host << ':' << bound.port << std::endl;
        cli::wait();
        server.close();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "nixlshard-metadata: " << error.what() << '\n';
        return 1;
    }
}
