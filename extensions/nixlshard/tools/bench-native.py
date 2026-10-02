#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-outstanding-request debug-file benchmark. Does not measure model TTFT."""
import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import platform
import statistics
import tempfile
import time

def wait(agent, handle, limit=30):
    until = time.monotonic() + limit
    try:
        while time.monotonic() < until:
            result = agent.poll(handle)
            if result is not None:
                if result != ["success"]:
                    raise RuntimeError(f"transfer failed: {result}")
                return
            time.sleep(0.00005)
        raise TimeoutError("benchmark operation exceeded deadline")
    finally:
        agent.release(handle)

def configuration(name, path, slot_bytes, direct_io):
    return {
        "name": name,
        "disks": [{"path": path, "capacity_bytes": 256 * 1024 * 1024,
                   "unit_bytes": 4096, "metadata_bytes": 1024 * 1024, "create": True}],
        "staging_slots": 4, "staging_slot_bytes": slot_bytes,
        "max_inflight": 4, "workers": 2, "timeout_ms": 10000,
        "direct_io": direct_io,
    }

def owner_main(pipe, path, sizes, slot_bytes, direct_io):
    from nixlshard import Agent, __build_marker__
    agent = None
    try:
        agent = Agent(configuration("benchmark-owner", path, slot_bytes, direct_io))
        pattern = bytes((i * 17 + 11) & 255 for i in range(256))
        source = ctypes.create_string_buffer(pattern * (slot_bytes // 256), slot_bytes)
        token = agent.register_memory(ctypes.addressof(source), slot_bytes)
        hashes = {}
        for size in sizes:
            key = f"bench:{size}"
            wait(agent, agent.batch_store([{"key": key, "segments": [(token, 0, size)]}]))
            hashes[key] = hashlib.sha256(source.raw[:size]).hexdigest()
        if agent.checkpoint() != "success":
            raise RuntimeError("owner checkpoint failed")
        agent.deregister_memory(token)
        pipe.send({"endpoint": agent.endpoint(), "hashes": hashes, "marker": __build_marker__})
        while True:
            command = pipe.recv()
            if command == "stats":
                pipe.send(agent.stats())
            elif command == "close":
                agent.close()
                pipe.send({"closed": True})
                break
            else:
                raise ValueError("invalid parent command")
    except BaseException as error:
        try:
            pipe.send({"error": repr(error)})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        if agent is not None:
            agent.close()
        pipe.close()

def delta(after, before):
    return {key: value - before.get(key, 0)
            for key, value in after.items()
            if key.endswith("_ns") or key.endswith("_bytes") or key in ("success", "stores")}

def summarize(samples):
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "median_us": statistics.median(samples) / 1000,
        "p95_us": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] / 1000,
        "min_us": min(samples) / 1000,
    }

def run(args):
    from nixlshard import Agent, __build_marker__, __file__ as native_package
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(value <= 0 or value > 32 * 1024 * 1024 for value in sizes):
        raise ValueError("sizes must be positive and at most 32MiB")
    slot_bytes = (max(sizes) + 4095) // 4096 * 4096
    pathlib.Path(args.directory).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    results = []
    with tempfile.TemporaryDirectory(prefix="bench-", dir=args.directory) as directory:
        parent, child = ctx.Pipe()
        process = ctx.Process(target=owner_main,
                              args=(child, os.path.join(directory, "owner.ssd"), sizes,
                                    slot_bytes, args.direct_io))
        process.start()
        child.close()
        agent = None
        try:
            if not parent.poll(60):
                raise TimeoutError("owner did not initialize")
            ready = parent.recv()
            if "error" in ready:
                raise RuntimeError(ready["error"])
            cfg = configuration("benchmark-requester", os.path.join(directory, "requester.ssd"),
                                slot_bytes, args.direct_io)
            cfg["peers"] = {"benchmark-owner": ready["endpoint"]}
            agent = Agent(cfg)
            destination = ctypes.create_string_buffer(slot_bytes)
            local_source = ctypes.create_string_buffer(b"L" * slot_bytes, slot_bytes)
            dst = agent.register_memory(ctypes.addressof(destination), slot_bytes)
            src = agent.register_memory(ctypes.addressof(local_source), slot_bytes)
            for size in sizes:
                local_key = f"local:{size}"
                wait(agent, agent.batch_store([{"key": local_key, "segments": [(src, 0, size)]}]))
                for route in ("local", "remote"):
                    key = local_key if route == "local" else f"bench:{size}"
                    hint = "" if route == "local" else "benchmark-owner"
                    expected = (hashlib.sha256(local_source.raw[:size]).hexdigest() if route == "local"
                                else ready["hashes"][key])
                    until = time.monotonic() + 10
                    while not agent.batch_exists([key], [hint])[0]:
                        if time.monotonic() >= until:
                            raise TimeoutError(f"{route} preflight exists not ready")
                        time.sleep(0.02)
                    for _ in range(args.warmup):
                        wait(agent, agent.batch_load([{"key": key, "hint": hint,
                                                      "segments": [(dst, 0, size)]}]))
                    before = agent.stats()
                    parent.send("stats")
                    owner_before = parent.recv()
                    samples = []
                    for _ in range(args.iterations):
                        start = time.perf_counter_ns()
                        handle = agent.batch_load([{"key": key, "hint": hint,
                                                   "segments": [(dst, 0, size)]}])
                        wait(agent, handle)
                        samples.append(time.perf_counter_ns() - start)
                        if hashlib.sha256(destination.raw[:size]).hexdigest() != expected:
                            raise AssertionError("successful read returned incorrect bytes")
                    after = agent.stats()
                    parent.send("stats")
                    owner_after = parent.recv()
                    results.append({
                        "route": route, "bytes": size, **summarize(samples),
                        "requester_counters": delta(after, before),
                        "owner_counters": delta(owner_after, owner_before),
                    })
            agent.deregister_memory(src)
            agent.deregister_memory(dst)
            agent.close()
            parent.send("close")
            if not parent.poll(15):
                raise TimeoutError("owner did not close")
            response = parent.recv()
            if response != {"closed": True}:
                raise RuntimeError(response)
            process.join(15)
            if process.exitcode != 0:
                raise RuntimeError(f"owner process failed: {process.exitcode}")
        finally:
            if agent is not None:
                agent.close()
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
    result = {
        "schema": "nixlshard-native-debug-file-v1",
        "build_marker": __build_marker__,
        "native_package": native_package,
        "platform": platform.platform(),
        "direct_io": args.direct_io,
        "directory": args.directory,
        "one_request_outstanding": True,
        "scope": "functional/local and two-process UCX debug-file latency; not model TTFT or RDMA",
        "results": results,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        pathlib.Path(args.output).write_text(encoded + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/raid/nixlshard-v2")
    parser.add_argument("--sizes", default="4096,1048576,16777216")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.iterations < 1 or arguments.warmup < 0:
        parser.error("iterations must be positive; warmup must be nonnegative")
    run(arguments)
