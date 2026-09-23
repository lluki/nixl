#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-node NIXLShard data-path check and agent microbenchmark.

Run from a NIXL source checkout with PYTHONPATH pointing at nixl-meta.
Each client condition performs a byte-for-byte SET/GET check before timing.
JSON lines on stdout are the raw artifact; stderr is for diagnostics.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

from nixl_shard import (
    AgentConfig,
    CompletionStatus,
    IoItem,
    RemoteDevice,
    ShardAgent,
    buffer_region_from_writable,
)

_json_stream = sys.stdout


def set_json_output(path):
    """Write machine-readable records separately from native library logs."""
    global _json_stream
    _json_stream = open(path, "w", encoding="utf-8", buffering=1)


def emit(kind, **values):
    print(
        json.dumps({"kind": kind, "time_ns": time.time_ns(), **values}, sort_keys=True),
        file=_json_stream,
        flush=True,
    )


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def metadata():
    import importlib.metadata
    import nixl
    import nixl_shard

    try:
        ucx_version = (
            subprocess.check_output(
                ["ucx_info", "-v"], text=True, stderr=subprocess.DEVNULL
            )
            .splitlines()[0]
            .removeprefix("# Library version: ")
        )
    except Exception:
        ucx_version = None
    nic_numa = Path("/sys/class/infiniband/mlx5_0/device/numa_node")
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nic_numa_node": (nic_numa.read_text().strip() if nic_numa.exists() else None),
        "nixl_version": importlib.metadata.version("nixl"),
        "ucx_version": ucx_version,
        "nixl_git_sha": git_sha(),
        "nixl_git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        )
        .strip()
        .splitlines(),
        "nixl_module": nixl.__file__,
        "nixl_shard_module": nixl_shard.__file__,
        "ucx_tls": os.getenv("UCX_TLS"),
        "ucx_net_devices": os.getenv("UCX_NET_DEVICES"),
    }


def require_ucx_interface():
    config_names = {f.name for f in fields(AgentConfig)}
    route_names = {f.name for f in fields(RemoteDevice)}
    missing = {"ucx_listen_host", "ucx_listen_port"} - config_names
    if (
        missing
        or "transport" not in route_names
        or not hasattr(ShardAgent, "ucx_endpoint")
        or not hasattr(ShardAgent, "ucx_transfer_bytes")
    ):
        raise RuntimeError(
            "UCX NIXLShard interface unavailable; missing config fields "
            f"{sorted(missing)}, route transport={('transport' in route_names)}, "
            f"endpoint={hasattr(ShardAgent, 'ucx_endpoint')}, "
            f"byte counter={hasattr(ShardAgent, 'ucx_transfer_bytes')}"
        )


def aligned_buffer(length, alignment=4096):
    backing = bytearray(length + alignment)
    start = (-ctypes.addressof(ctypes.c_char.from_buffer(backing))) % alignment
    return memoryview(backing)[start : start + length]


def pattern(length, condition, slot):
    seed = hashlib.sha256(f"{condition}:{slot}".encode()).digest()
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def check(results, count, size, operation):
    if len(results) != count:
        raise AssertionError(
            f"{operation}: expected {count} completions, got {len(results)}"
        )
    for index, result in enumerate(results):
        if result.status is not CompletionStatus.OK or result.bytes_transferred != size:
            raise AssertionError(
                f"{operation} item {index}: {result.status} "
                f"bytes={result.bytes_transferred} detail={result.detail}"
            )


def run_batch(agent, operation, items, io_batch_size, timeout):
    handles = []
    submit = agent.submit_batch_store if operation == "set" else agent.submit_batch_load
    starts = []
    for offset in range(0, len(items), io_batch_size):
        starts.append(time.perf_counter_ns())
        handles.extend(submit(items[offset : offset + io_batch_size]))
    results = agent.wait(handles, timeout)
    end = time.perf_counter_ns()
    check(results, len(items), items[0].length, operation)
    return end - starts[0]


def percentiles(samples):
    samples = sorted(samples)

    def percentile(p):
        index = max(0, min(len(samples) - 1, (len(samples) * p + 99) // 100 - 1))
        return samples[index]

    return {
        "min_ns": samples[0],
        "p50_ns": percentile(50),
        "p95_ns": percentile(95),
        "p99_ns": percentile(99),
        "max_ns": samples[-1],
        "mean_ns": statistics.mean(samples),
    }


def make_server(args):
    if args.transport == "ucx":
        require_ucx_interface()
    file_path = Path(args.file)
    if file_path.exists() and not args.reuse_file:
        raise FileExistsError(f"{file_path} exists; choose a new file or --reuse-file")
    kwargs = {
        "file_path": file_path,
        "size": args.file_size,
        "logical_device_id": args.device_id,
        "create": not args.reuse_file,
        "direct_io": args.direct_io,
        "max_inflight_ops": args.max_inflight,
        "tcp_max_workers": args.workers,
    }
    if args.transport == "tcp":
        kwargs.update(tcp_listen_host=args.listen_host, tcp_listen_port=args.port)
    else:
        kwargs.update(ucx_listen_host=args.listen_host, ucx_listen_port=args.port)
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with ShardAgent.open(AgentConfig(**kwargs)) as agent:
        endpoint = agent.ucx_endpoint if args.transport == "ucx" else agent.tcp_endpoint
        if endpoint is None:
            raise RuntimeError(f"{args.transport} listener did not start")
        emit(
            "server_ready",
            transport=args.transport,
            host=endpoint.host,
            port=endpoint.port,
            device_id=args.device_id,
            file=str(file_path),
            file_size=args.file_size,
            direct_io=args.direct_io,
            workers=args.workers,
            **metadata(),
        )
        while not stop:
            time.sleep(0.2)
        emit("server_stopped", metrics=vars(agent.get_metrics()))


def make_client(args):
    if args.mode == "ucx":
        require_ucx_interface()
    if args.mode != "local" and (not args.remote_host or not args.remote_port):
        raise ValueError(
            "--remote-host and --remote-port are required for remote modes"
        )
    if args.mode == "local" and args.device_id == args.local_device_id:
        raise ValueError("local and remote logical device IDs must differ")
    file_path = Path(args.file)
    if file_path.exists() and not args.reuse_file:
        raise FileExistsError(f"{file_path} exists; choose a new file or --reuse-file")
    kwargs = {
        "file_path": file_path,
        "size": args.file_size,
        "logical_device_id": args.local_device_id,
        "create": not args.reuse_file,
        "direct_io": args.direct_io,
        "max_inflight_ops": args.max_inflight,
        "tcp_max_workers": args.workers,
    }
    if args.mode != "local":
        route_kwargs = {}
        if args.mode == "ucx":
            route_kwargs["transport"] = "ucx"
        kwargs["remote_devices"] = (
            RemoteDevice(
                args.device_id, args.remote_host, args.remote_port, 1, **route_kwargs
            ),
        )
    with ShardAgent.open(AgentConfig(**kwargs)) as agent:
        emit(
            "run_config",
            mode=args.mode,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            local_file=str(file_path),
            file_size=args.file_size,
            direct_io=args.direct_io,
            sizes=args.sizes,
            queue_depths=args.queue_depths,
            io_batch_sizes=args.io_batch_sizes,
            iterations=args.iterations,
            warmup=args.warmup,
            max_inflight=args.max_inflight,
            workers=args.workers,
            **metadata(),
        )
        for size in args.sizes:
            for depth in args.queue_depths:
                if size * depth > args.file_size:
                    raise ValueError(f"size {size} * depth {depth} exceeds file size")
                for batch_size in args.io_batch_sizes:
                    if batch_size > depth or depth % batch_size:
                        continue
                    if depth > args.max_inflight:
                        raise ValueError("queue depth exceeds max in-flight ops")
                    condition = f"{args.mode}:{size}:{depth}:{batch_size}"
                    memory = aligned_buffer(size * depth)
                    handle = agent.register_memory(
                        [buffer_region_from_writable(memory)]
                    )
                    target_device = (
                        args.local_device_id if args.mode == "local" else args.device_id
                    )
                    items = [
                        IoItem(target_device, i * size, handle, 0, i * size, size)
                        for i in range(depth)
                    ]
                    try:
                        ucx_before = (
                            agent.ucx_transfer_bytes if args.mode == "ucx" else None
                        )
                        for i in range(depth):
                            memory[i * size : (i + 1) * size] = pattern(
                                size, condition, i
                            )
                        expected = bytes(memory)
                        run_batch(agent, "set", items, batch_size, args.timeout)
                        memory[:] = bytes(len(memory))
                        run_batch(agent, "get", items, batch_size, args.timeout)
                        if memory.tobytes() != expected:
                            raise AssertionError(
                                f"{condition}: byte-for-byte GET mismatch"
                            )
                        ucx_delta = None
                        if args.mode == "ucx":
                            ucx_delta = agent.ucx_transfer_bytes - ucx_before
                            if ucx_delta < 2 * len(memory):
                                raise AssertionError(
                                    f"{condition}: UCX counter advanced {ucx_delta}, "
                                    f"expected at least {2 * len(memory)} bytes"
                                )
                        emit(
                            "correctness_pass",
                            mode=args.mode,
                            size=size,
                            queue_depth=depth,
                            io_batch_size=batch_size,
                            logical_bytes=len(memory),
                            ucx_transfer_bytes_delta=ucx_delta,
                        )
                        if args.iterations:
                            memory[:] = expected
                            for operation in ("set", "get"):
                                if operation == "get":
                                    memory[:] = bytes(len(memory))
                                for _ in range(args.warmup):
                                    run_batch(
                                        agent,
                                        operation,
                                        items,
                                        batch_size,
                                        args.timeout,
                                    )
                                before_timed_ucx = (
                                    agent.ucx_transfer_bytes
                                    if args.mode == "ucx"
                                    else None
                                )
                                durations = [
                                    run_batch(
                                        agent,
                                        operation,
                                        items,
                                        batch_size,
                                        args.timeout,
                                    )
                                    for _ in range(args.iterations)
                                ]
                                timed_ucx_delta = None
                                if args.mode == "ucx":
                                    timed_ucx_delta = (
                                        agent.ucx_transfer_bytes - before_timed_ucx
                                    )
                                    minimum = size * depth * args.iterations
                                    if timed_ucx_delta < minimum:
                                        raise AssertionError(
                                            f"{condition}: timed {operation} UCX bytes "
                                            f"{timed_ucx_delta} smaller than {minimum}"
                                        )
                                total_seconds = sum(durations) / 1e9
                                emit(
                                    "benchmark",
                                    mode=args.mode,
                                    operation=operation,
                                    measurement_scope="agent_data_path_without_metadata",
                                    size=size,
                                    queue_depth=depth,
                                    io_batch_size=batch_size,
                                    iterations=args.iterations,
                                    warmup=args.warmup,
                                    logical_bytes_per_iteration=size * depth,
                                    total_logical_bytes=size * depth * args.iterations,
                                    ucx_transfer_bytes_timed=timed_ucx_delta,
                                    throughput_mib_s=(
                                        size * depth * args.iterations / (1024**2)
                                    )
                                    / total_seconds,
                                    latency=percentiles(durations),
                                    samples_ns=durations,
                                )
                            if memory.tobytes() != expected:
                                raise AssertionError(
                                    f"{condition}: byte-for-byte post-benchmark GET mismatch"
                                )
                    finally:
                        agent.unregister_memory(handle)
        emit(
            "run_complete",
            metrics=vars(agent.get_metrics()),
            ucx_transfer_bytes=(
                agent.ucx_transfer_bytes if args.mode == "ucx" else None
            ),
        )


def parse_bytes(text):
    suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
    text = text.strip().lower()
    scale = suffixes.get(text[-1], 1)
    if scale != 1:
        text = text[:-1]
    value = int(text) * scale
    if value <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("server")
    server.add_argument("--transport", choices=("tcp", "ucx"), required=True)
    server.add_argument("--listen-host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=18741)
    server.add_argument("--device-id", type=int, default=2)
    client = sub.add_parser("client")
    client.add_argument("--mode", choices=("local", "tcp", "ucx"), required=True)
    client.add_argument("--remote-host")
    client.add_argument("--remote-port", type=int)
    client.add_argument("--device-id", type=int, default=2)
    client.add_argument("--local-device-id", type=int, default=1)
    client.add_argument(
        "--sizes",
        type=parse_bytes,
        nargs="+",
        default=[4096, 16384, 65536, 262144, 1048576, 4194304],
    )
    client.add_argument(
        "--queue-depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    client.add_argument("--io-batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    client.add_argument("--iterations", type=int, default=100)
    client.add_argument("--warmup", type=int, default=10)
    client.add_argument("--timeout", type=float, default=30)
    for command in (server, client):
        command.add_argument("--file", required=True)
        command.add_argument(
            "--output", help="write JSONL records here, separate from native logs"
        )
        command.add_argument("--file-size", type=parse_bytes, default=512 * 1024**2)
        command.add_argument("--max-inflight", type=int, default=128)
        command.add_argument(
            "--workers", type=int, default=32, help="remote request worker limit"
        )
        command.add_argument("--direct-io", action="store_true")
        command.add_argument("--reuse-file", action="store_true")
    args = parser.parse_args()
    if args.output:
        set_json_output(args.output)
    if args.command == "server":
        make_server(args)
    else:
        make_client(args)


if __name__ == "__main__":
    main()
