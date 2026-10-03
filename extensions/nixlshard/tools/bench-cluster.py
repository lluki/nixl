#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verified cross-host NIXLShard reads; HTTP carries benchmark metadata only.

Start owner and requester in separate pods. Select UCX transports through the
environment and retain UCX logs; successful reads alone do not establish RDMA.
"""
import argparse
import ctypes
import hashlib
import http.server
import json
import os
import pathlib
import platform
import tempfile
import time
import urllib.request
import urllib.error

from importlib.machinery import SourceFileLoader

native = SourceFileLoader("bench_native", str(pathlib.Path(__file__).with_name("bench-native.py"))).load_module()


def provenance():
    import nixlshard
    return {
        "build_marker": nixlshard.__build_marker__,
        "package": nixlshard.__file__,
        "platform": platform.platform(),
        "boot_id": pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "ucx_environment": {key: value for key, value in os.environ.items() if key.startswith("UCX_")},
        "native_libraries": sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                                    if "/" in line and any(name in line for name in ("libnixl", "libucp", "libuct", "libabsl"))}),
    }


def configuration(args, name, disk, slot_bytes):
    cfg = native.configuration(name, disk, slot_bytes, args.direct_io)
    cfg.update(listen_host=args.host, listen_port=0)
    return cfg


def owner(args, sizes, slot_bytes):
    from nixlshard import Agent
    with tempfile.TemporaryDirectory(prefix="cluster-owner-", dir=args.directory) as directory:
        agent = Agent(configuration(args, "cluster-owner", str(pathlib.Path(directory) / "cache.ssd"), slot_bytes))
        try:
            pattern = bytes((i * 17 + 11) & 255 for i in range(256))
            source = ctypes.create_string_buffer((pattern * ((slot_bytes + 255) // 256))[:slot_bytes], slot_bytes)
            token = agent.register_memory(ctypes.addressof(source), slot_bytes)
            hashes = {}
            for size in sizes:
                key = f"cluster:{size}"
                native.wait(agent, agent.batch_store([{"key": key, "segments": [(token, 0, size)]}]))
                hashes[key] = hashlib.sha256(source.raw[:size]).hexdigest()
            if agent.checkpoint() != "success":
                raise RuntimeError("owner checkpoint failed")
            agent.deregister_memory(token)
            manifest = {"endpoint": agent.endpoint(), "hashes": hashes, "sizes": sizes,
                        "direct_io": args.direct_io, "provenance": provenance()}
            stopping = False

            class Handler(http.server.BaseHTTPRequestHandler):
                def setup(self):
                    super().setup()
                    self.connection.settimeout(5)

                def do_GET(self):
                    nonlocal stopping
                    if self.path == "/ready":
                        value = manifest
                    elif self.path == "/stats":
                        value = agent.stats()
                    elif self.path == "/stop":
                        stopping = True
                        value = {"stopping": True}
                    else:
                        self.send_error(404)
                        return
                    body = json.dumps(value).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_):
                    pass

            with http.server.HTTPServer((args.host, args.control_port), Handler) as server:
                server.timeout = 0.5
                print(json.dumps({"ready": manifest, "control_port": server.server_port}), flush=True)
                until = time.monotonic() + args.owner_seconds
                while not stopping and time.monotonic() < until:
                    server.handle_request()
                pathlib.Path(args.output).write_text(json.dumps({"manifest": manifest, "final_stats": agent.stats()}, indent=2) + "\n")
        finally:
            agent.close()


def requester(args, sizes, slot_bytes):
    from nixlshard import Agent

    def get(path):
        with urllib.request.urlopen(args.owner.rstrip("/") + path, timeout=15) as response:
            return json.load(response)

    until = time.monotonic() + 30
    while True:
        try:
            ready = get("/ready")
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= until:
                raise
            time.sleep(0.1)
    if ready["sizes"] != sizes or ready["direct_io"] != args.direct_io:
        raise ValueError("owner/requester sizes and direct-I/O configuration must match")
    if ready["provenance"]["boot_id"] == provenance()["boot_id"]:
        raise ValueError("cross-host benchmark requires different host boot IDs")
    results = []
    with tempfile.TemporaryDirectory(prefix="cluster-requester-", dir=args.directory) as directory:
        cfg = configuration(args, "cluster-requester", str(pathlib.Path(directory) / "cache.ssd"), slot_bytes)
        cfg["peers"] = {"cluster-owner": ready["endpoint"]}
        agent = Agent(cfg)
        try:
            destination = ctypes.create_string_buffer(slot_bytes)
            token = agent.register_memory(ctypes.addressof(destination), slot_bytes)
            for size in sizes:
                key = f"cluster:{size}"
                until = time.monotonic() + 30
                while not agent.batch_exists([key], ["cluster-owner"])[0]:
                    if time.monotonic() >= until:
                        raise TimeoutError("remote owner did not become ready")
                    time.sleep(0.02)
                item = {"key": key, "hint": "cluster-owner", "segments": [(token, 0, size)]}
                for _ in range(args.warmup):
                    native.wait(agent, agent.batch_load([item]))
                before = agent.stats()
                owner_before = get("/stats")
                samples = []
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    native.wait(agent, agent.batch_load([item]))
                    samples.append(time.perf_counter_ns() - start)
                    if hashlib.sha256(destination.raw[:size]).hexdigest() != ready["hashes"][key]:
                        raise AssertionError("successful remote load returned incorrect bytes")
                results.append({"bytes": size, "verified_loads": len(samples), "samples_ns": samples,
                                **native.summarize(samples), "requester_counters": native.delta(agent.stats(), before),
                                "owner_counters": native.delta(get("/stats"), owner_before)})
            agent.deregister_memory(token)
            result = {"schema": "nixlshard-cross-host-file-v1", "requester": provenance(),
                      "owner": ready, "one_request_outstanding": True, "results": results,
                      "scope": "cross-host verified file reads; inspect UCX logs for transport; not model TTFT"}
            pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            agent.close()
    if args.stop_owner:
        get("/stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owner", "requester"))
    parser.add_argument("--host", required=True, help="Concrete routable numeric IPv4 address")
    parser.add_argument("--owner", help="Owner benchmark metadata URL")
    parser.add_argument("--control-port", type=int, default=32080)
    parser.add_argument("--directory", default="/scratch/nixlshard-v2")
    parser.add_argument("--sizes", default="4096,1048576,16777216")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--owner-seconds", type=int, default=600)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--stop-owner", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(size <= 0 or size > 32 * 1024 * 1024 for size in sizes):
        parser.error("sizes must be positive and at most 32MiB")
    if args.iterations < 1 or args.warmup < 0 or args.owner_seconds < 1:
        parser.error("invalid iteration, warmup or owner lifetime")
    if args.mode == "requester" and not args.owner:
        parser.error("requester requires --owner")
    pathlib.Path(args.directory).mkdir(parents=True, exist_ok=True)
    slot_bytes = (max(sizes) + 4095) // 4096 * 4096
    (owner if args.mode == "owner" else requester)(args, sizes, slot_bytes)
