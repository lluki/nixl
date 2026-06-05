#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Path-mode FILE-registration GIL micro-benchmark for NIXL.

Measures the win of NIXL path-mode FILE registration (PR #1635,
https://github.com/ai-dynamo/nixl/pull/1635) over the legacy fd-in-devId mode,
under GIL contention.

Inspired by SGLang HiCache's NIXL backend: today it opens one fd per cache key
in Python (os.open() per key) and passes the fd in the descriptor devId. Each
os.open() is a GIL crossing; with N keys that is N GIL acquisitions during
registration, which serialize behind any other thread holding the GIL in a real
serving process (tokenizer, scheduler, CUDA callbacks, HTTP server). Path mode
collapses those N opens into one C++ registerMem: Python builds N
"rw,create:<path>" metaInfo strings and the FILE_SEG backend opens every fd
itself, with the GIL released for the C++ work.

This benchmark registers N files both ways under two conditions -- (a) a quiet
main thread and (b) a background daemon thread spinning a pure-Python loop (GIL
contention) -- and sweeps N. Under contention the legacy time grows with N while
path mode stays roughly flat. nixlbench (C++) cannot show this: it times only
the transfer phase, with registerMem run once outside the timed loop.

POSIX runs everywhere; GDS / GDS_MT / HF3FS are used only when present and
skipped (with a log) otherwise.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

from nixl._api import nixl_agent, nixl_agent_config

PATTERN = 0xAB


class Spinner:
    """Background daemon thread doing GIL-bound busy work.

    Used as a context manager: the thread runs for the duration of the `with`
    block. `counter` is a live, monotonically increasing iteration count the
    main thread samples around a timed region to see how much CPU the spinner
    got (a proxy for whether the GIL was available to it).
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.counter = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.counter += 1

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False


def make_paths(directory, n):
    return [os.path.join(directory, f"nixl_gil_mb_{i}.bin") for i in range(n)]


def cleanup_paths(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def time_path_mode(agent, backend, paths, size, direct, spinner):
    """Path mode: N metaInfo strings + one register_memory, no Python open."""
    modes = "rw,create,direct" if direct else "rw,create"
    c0 = spinner.counter
    t0 = time.perf_counter()
    descs = [(0, size, i + 1, f"{modes}:{p}") for i, p in enumerate(paths)]
    reg = agent.register_memory(descs, mem_type="FILE", backends=[backend])
    elapsed = time.perf_counter() - t0
    spins = spinner.counter - c0
    agent.deregister_memory(reg, backends=[backend])
    return elapsed, spins


def time_legacy(agent, backend, paths, size, direct, spinner):
    """Legacy mode: N os.open() (the GIL cost) + one register_memory."""
    flags = os.O_RDWR | os.O_CREAT
    if direct:
        flags |= os.O_DIRECT
    c0 = spinner.counter
    t0 = time.perf_counter()
    fds = [os.open(p, flags, 0o644) for p in paths]
    descs = [(0, size, fd, "") for fd in fds]
    reg = agent.register_memory(descs, mem_type="FILE", backends=[backend])
    elapsed = time.perf_counter() - t0
    spins = spinner.counter - c0
    agent.deregister_memory(reg, backends=[backend])
    for fd in fds:
        os.close(fd)
    return elapsed, spins


def correctness_check(agent, backend, directory, size):
    """Path-mode register one file, NIXL-WRITE a known pattern, read it back.

    Guards against a silent devId collapse or a wrong-file open masquerading as
    "fast": if the backend opened the wrong fd the pattern would not land in the
    expected file.
    """
    path = os.path.join(directory, "nixl_gil_mb_check.bin")
    cleanup_paths([path])
    src = np.full(size, PATTERN, dtype=np.uint8)
    addr = src.ctypes.data
    dram = agent.register_memory(
        [(addr, size, 0, "")], mem_type="DRAM", backends=[backend]
    )
    file_reg = agent.register_memory(
        [(0, size, 1, f"rw,create:{path}")], mem_type="FILE", backends=[backend]
    )
    local = agent.get_xfer_descs([(addr, size, 0)], "DRAM")
    remote = agent.get_xfer_descs([(0, size, 1)], "FILE")
    handle = agent.initialize_xfer(
        "WRITE", local, remote, agent.name, backends=[backend]
    )
    state = agent.transfer(handle)
    while state == "PROC":
        state = agent.check_xfer_state(handle)
    agent.release_xfer_handle(handle)
    agent.deregister_memory(file_reg, backends=[backend])
    agent.deregister_memory(dram, backends=[backend])
    if state != "DONE":
        raise SystemExit(f"correctness guard FAILED: transfer state {state}")
    with open(path, "rb") as f:
        data = f.read(size)
    cleanup_paths([path])
    if data != bytes([PATTERN]) * size:
        raise SystemExit("correctness guard FAILED: path-mode round-trip mismatch")


def benchmark(agent, args):
    timers = {"legacy": time_legacy, "path": time_path_mode}
    results = []
    for n in args.counts:
        for cond, use_spin in (("quiet", False), ("contended", True)):
            for mode, fn in timers.items():
                times, spins = [], []
                for _ in range(args.repeats):
                    paths = make_paths(args.dir, n)
                    cleanup_paths(paths)
                    with Spinner(use_spin) as sp:
                        elapsed, it = fn(
                            agent, args.backend, paths, args.size, args.direct, sp
                        )
                    times.append(elapsed)
                    spins.append(it)
                    cleanup_paths(paths)
                median_s = statistics.median(times)
                results.append(
                    {
                        "n": n,
                        "mode": mode,
                        "condition": cond,
                        "total_ms": median_s * 1e3,
                        "per_file_us": median_s / n * 1e6,
                        "spinner_iters": statistics.median(spins) if use_spin else None,
                    }
                )
    return results


def print_table(results, backend, repeats):
    print(
        f"\nNIXL path-mode FILE registration micro-benchmark (backend={backend}, "
        f"median of {repeats})\n"
    )
    head = f"{'N':>6}  {'mode':<7}  {'condition':<10}  {'total(ms)':>10}  {'per-file(us)':>13}  {'spin iters':>12}"
    print(head)
    print("-" * len(head))
    for r in results:
        spin = "-" if r["spinner_iters"] is None else f"{int(r['spinner_iters']):,}"
        print(
            f"{r['n']:>6}  {r['mode']:<7}  {r['condition']:<10}  "
            f"{r['total_ms']:>10.3f}  {r['per_file_us']:>13.3f}  {spin:>12}"
        )

    # Headline: legacy-vs-path slowdown under contention, per N.
    print("\ncontended legacy/path total-time ratio (higher = bigger path-mode win):")
    by = {(r["n"], r["mode"]): r for r in results if r["condition"] == "contended"}
    for n in sorted({r["n"] for r in results}):
        leg, pth = by.get((n, "legacy")), by.get((n, "path"))
        if leg and pth and pth["total_ms"] > 0:
            print(f"  N={n:>6}: {leg['total_ms'] / pth['total_ms']:6.2f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--backend",
        default="POSIX",
        help="FILE_SEG backend: POSIX (default), GDS, GDS_MT, HF3FS",
    )
    ap.add_argument(
        "--counts",
        default="64,256,1024",
        help="comma-separated N values to sweep (add 4096 for the full divergence; "
        "contended legacy scales ~N * GIL-switch-interval, so large N is slow "
        "by design)",
    )
    ap.add_argument("--dir", default="/tmp", help="directory for the test files")
    ap.add_argument(
        "--size", type=int, default=4096, help="per-file descriptor size (bytes)"
    )
    ap.add_argument(
        "--repeats", type=int, default=3, help="runs per data point (median reported)"
    )
    ap.add_argument(
        "--direct",
        action="store_true",
        help="add O_DIRECT (legacy) / ,direct (path mode); dir must support O_DIRECT",
    )
    ap.add_argument(
        "--json", metavar="PATH", help="also write machine-readable results here"
    )
    args = ap.parse_args(argv)
    args.counts = [int(x) for x in args.counts.split(",") if x]

    agent = nixl_agent("gil_microbench", nixl_agent_config(backends=[args.backend]))
    if args.backend not in agent.get_plugin_list():
        print(
            f"backend {args.backend} unavailable (plugins: {agent.get_plugin_list()}); "
            f"skipping",
            file=sys.stderr,
        )
        return 0
    if "FILE_SEG" not in agent.get_backend_mem_types(args.backend):
        print(
            f"backend {args.backend} does not support FILE_SEG; skipping",
            file=sys.stderr,
        )
        return 0

    correctness_check(agent, args.backend, args.dir, args.size)
    results = benchmark(agent, args)
    print_table(results, args.backend, args.repeats)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "backend": args.backend,
                    "size": args.size,
                    "direct": args.direct,
                    "repeats": args.repeats,
                    "counts": args.counts,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
