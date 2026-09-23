#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-agent NIXLShard client/service GET/SET correctness scenario.

Run the separate nixlshard_bench.py server in a peer Pod first. This client
runs the naming service and its RPC server locally, then routes data to the
peer. Every SET and GET is checked before any success record is emitted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from nixl_shard import (
    AgentConfig,
    ClientConfig,
    DeviceRegistration,
    GetItem,
    KVStatus,
    RemoteDevice,
    ResultCode,
    ServiceConfig,
    SetItem,
    ShardAgent,
    ShardClient,
    ShardNamingService,
    ShardNamingTCPClient,
    ShardNamingTCPServer,
)

from nixlshard_bench import (
    emit,
    metadata,
    parse_bytes,
    pattern,
    percentiles,
    require_ucx_interface,
    set_json_output,
)


def check_results(results, expected_count, operation, device_id, size):
    if len(results) != expected_count:
        raise AssertionError(
            f"{operation}: expected {expected_count} results, got {len(results)}"
        )
    for index, result in enumerate(results):
        if result.status is not KVStatus.OK or result.device_id != device_id:
            raise AssertionError(
                f"{operation} item {index}: {result.status} "
                f"device={result.device_id} detail={result.detail}"
            )
        if result.logical_length != size:
            raise AssertionError(
                f"{operation} item {index}: expected length {size}, "
                f"got {result.logical_length}"
            )


def run(args):
    if args.transport == "ucx":
        require_ucx_interface()
    file_path = Path(args.file)
    if file_path.exists() and not args.reuse_file:
        raise FileExistsError(f"{file_path} exists; choose a new file or --reuse-file")
    service = ShardNamingService(
        ServiceConfig(auto_evict=False, device_timeout_s=300.0)
    )
    naming_server = ShardNamingTCPServer(service).start()
    naming = ShardNamingTCPClient(*naming_server.address, timeout=args.timeout)
    route_kwargs = {"transport": "ucx"} if args.transport == "ucx" else {}
    agent_config = AgentConfig(
        file_path=file_path,
        size=args.file_size,
        logical_device_id=args.local_device_id,
        create=not args.reuse_file,
        direct_io=False,
        max_inflight_ops=args.max_inflight,
        remote_devices=(
            RemoteDevice(
                args.device_id, args.remote_host, args.remote_port, 1, **route_kwargs
            ),
        ),
    )
    try:
        with ShardAgent.open(agent_config) as agent:
            with ShardClient(
                ClientConfig(
                    service=naming, agent=agent, default_timeout_s=args.timeout
                )
            ) as client:
                endpoint = f"{args.transport}://{args.remote_host}:{args.remote_port}"
                registered = naming.register_device(
                    DeviceRegistration(
                        args.device_id, endpoint, args.remote_file_size, 4096, 1
                    )
                )
                if registered.code is not ResultCode.OK:
                    raise AssertionError(f"registration failed: {registered}")
                emit(
                    "e2e_run_config",
                    transport=args.transport,
                    remote_host=args.remote_host,
                    remote_port=args.remote_port,
                    local_file=str(file_path),
                    remote_file_size=args.remote_file_size,
                    sizes=args.sizes,
                    batch_sizes=args.batch_sizes,
                    rounds=args.rounds,
                    **metadata(),
                )
                total_items = 0
                total_logical_bytes = 0
                set_samples = []
                get_samples = []
                for size in args.sizes:
                    for batch_size in args.batch_sizes:
                        for round_index in range(args.rounds):
                            condition = (
                                f"{args.transport}:{size}:{batch_size}:{round_index}"
                            )
                            keys = [
                                f"nixlshard-e2e/{os.getpid()}/{condition}/{i}".encode()
                                for i in range(batch_size)
                            ]
                            sources = [
                                bytearray(pattern(size, condition, i))
                                for i in range(batch_size)
                            ]
                            before_ucx = (
                                agent.ucx_transfer_bytes
                                if args.transport == "ucx"
                                else None
                            )
                            set_items = [
                                SetItem(key, source, preferred_device_id=args.device_id)
                                for key, source in zip(keys, sources, strict=True)
                            ]
                            start = time.perf_counter_ns()
                            stored = client.batch_set(set_items)
                            set_ns = time.perf_counter_ns() - start
                            check_results(
                                stored, batch_size, "SET", args.device_id, size
                            )
                            if client.batch_exists(keys) != [True] * batch_size:
                                raise AssertionError(
                                    f"{condition}: committed keys not visible"
                                )
                            destinations = [bytearray(size) for _ in range(batch_size)]
                            get_items = [
                                GetItem(
                                    key, destination, preferred_device_id=args.device_id
                                )
                                for key, destination in zip(
                                    keys, destinations, strict=True
                                )
                            ]
                            start = time.perf_counter_ns()
                            loaded = client.batch_get(get_items)
                            get_ns = time.perf_counter_ns() - start
                            check_results(
                                loaded, batch_size, "GET", args.device_id, size
                            )
                            for i, (source, destination) in enumerate(
                                zip(sources, destinations, strict=True)
                            ):
                                if destination != source:
                                    raise AssertionError(
                                        f"{condition}: key {i} byte mismatch"
                                    )
                            if client.batch_exists([b"nixlshard-e2e/missing"]) != [
                                False
                            ]:
                                raise AssertionError(
                                    f"{condition}: nonexistent key appeared"
                                )
                            ucx_delta = None
                            if args.transport == "ucx":
                                ucx_delta = agent.ucx_transfer_bytes - before_ucx
                                if ucx_delta < 2 * batch_size * size:
                                    raise AssertionError(
                                        f"{condition}: UCX bytes {ucx_delta} smaller than "
                                        f"SET+GET {2 * batch_size * size}"
                                    )
                            total_items += batch_size
                            total_logical_bytes += batch_size * size
                            set_samples.append(set_ns)
                            get_samples.append(get_ns)
                            metrics = naming.get_metrics()
                            if (
                                metrics["objects_ready"] != total_items
                                or metrics["leases_active"] != 0
                            ):
                                raise AssertionError(
                                    f"{condition}: service accounting {metrics}"
                                )
                            naming.validate_invariants()
                            emit(
                                "e2e_correctness_pass",
                                transport=args.transport,
                                measurement_scope="client_metadata_and_data_path",
                                size=size,
                                batch_size=batch_size,
                                round=round_index,
                                logical_bytes=batch_size * size,
                                set_ns=set_ns,
                                get_ns=get_ns,
                                ucx_transfer_bytes_delta=ucx_delta,
                                service_metrics=metrics,
                            )
                metrics = naming.get_metrics()
                if (
                    metrics["reservations"] < total_items
                    or metrics["commits"] < total_items
                ):
                    raise AssertionError(
                        f"metadata reserve/commit counts too low: {metrics}"
                    )
                if metrics["lookups"] < total_items:
                    raise AssertionError(f"metadata lookup count too low: {metrics}")
                client_metrics = client.get_metrics()
                if client_metrics["hits"] < total_items:
                    raise AssertionError(f"client hit count too low: {client_metrics}")
                emit(
                    "e2e_complete",
                    transport=args.transport,
                    measurement_scope="client_metadata_and_data_path",
                    total_items=total_items,
                    total_logical_bytes=total_logical_bytes,
                    set_latency=percentiles(set_samples),
                    get_latency=percentiles(get_samples),
                    set_samples_ns=set_samples,
                    get_samples_ns=get_samples,
                    service_metrics=metrics,
                    client_metrics=client_metrics,
                    agent_metrics=vars(agent.get_metrics()),
                    ucx_transfer_bytes=(
                        agent.ucx_transfer_bytes if args.transport == "ucx" else None
                    ),
                )
    finally:
        naming_server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("tcp", "ucx"), required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument(
        "--output", help="write JSONL records here, separate from native logs"
    )
    parser.add_argument("--file-size", type=parse_bytes, default=16 * 1024**2)
    parser.add_argument("--remote-file-size", type=parse_bytes, default=512 * 1024**2)
    parser.add_argument("--reuse-file", action="store_true")
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--local-device-id", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument(
        "--sizes", type=parse_bytes, nargs="+", default=[4096, 16384, 65536]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.rounds < 1 or any(value < 1 for value in args.batch_sizes):
        parser.error("rounds and batch sizes must be positive")
    needed = (
        args.rounds
        * sum(args.batch_sizes)
        * sum(((size + 4095) // 4096) * 4096 for size in args.sizes)
    )
    if needed > args.remote_file_size:
        parser.error(
            f"remote file size {args.remote_file_size} is below required {needed}"
        )
    if args.output:
        set_json_output(args.output)
    run(args)


if __name__ == "__main__":
    main()
