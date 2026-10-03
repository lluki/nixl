#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fill a disposable native cache and measure batches that force block reuse."""
import argparse
import collections
import ctypes
import hashlib
import importlib.util
import json
import mmap
import multiprocessing
import os
import pathlib
import platform
import shutil
import statistics
import sys
import time
import uuid


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()


def counter_delta(after, before):
    # Current resource usage is a snapshot, not an event counter.
    gauges = {"staging_free_slots", "staging_quarantined_slots"}
    return {key: value - before.get(key, 0)
            for key, value in after.items() if key not in gauges}


def archive_client(prefix, output):
    # Reuse the immutable generation-match/hash-manifest implementation.
    # ADC and google.cloud.storage are imported only when uploads are enabled.
    helper_path = pathlib.Path(__file__).with_name("bench-ttft.py")
    spec = importlib.util.spec_from_file_location("saturation_gcs_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    archives = helper.GcsArchives(prefix, output)
    manifest = output / "gcs-archives.json"
    if manifest.exists():
        archives.uploads = json.loads(manifest.read_text())
    return archives


def archive_metadata(archives, output, name):
    start = time.perf_counter_ns()
    stage = output / "archive-metadata"
    stage.mkdir(exist_ok=True)
    # Explicit allowlist excludes the SSD, source/destination bytes, and any
    # credentials. Copying also gives each archive a consistent raw-log EOF.
    for filename in ("provenance.json", "batches.jsonl", "summary.json", "gcs-archives.json"):
        path = output / filename
        if path.exists():
            shutil.copyfile(path, stage / filename)
    archives.archive(stage, name)
    append(output / "batches.jsonl", {"phase": "archive_pacing", "archive": name,
           "elapsed_ns": time.perf_counter_ns() - start,
           "scope": "metadata copy/zip/upload outside timed batch; affects workload pacing and maintenance"})


def archive_final(prefix, output):
    try:
        archive_metadata(archive_client(prefix, output), output, "supervisor-final.zip")
    except BaseException as error:
        append(output / "batches.jsonl", {"phase": "archive_error", "error": repr(error)})
        raise


def checkpoint(agent, args, raw):
    append(raw, {"phase": "checkpoint_started"})
    before = agent.stats()
    start = time.perf_counter_ns()
    deadline = time.monotonic() + args.operation_watchdog_seconds
    attempts = []
    while True:
        status = agent.checkpoint()
        attempts.append({"status": status, "elapsed_ns": time.perf_counter_ns() - start})
        append(raw, {"phase": "checkpoint_attempt", "attempt": len(attempts),
                     **attempts[-1]})
        if status != "busy" or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    append(raw, {"phase": "checkpoint", "status": status, "attempts": attempts,
                 "elapsed_ns": time.perf_counter_ns() - start,
                 "counter_deltas": counter_delta(agent.stats(), before)})
    return status


def batch(agent, operation, items, deadline_seconds, raw, phase):
    keys = [item["key"] for item in items]
    append(raw, {"phase": phase, "operation": "batch_started",
                 "native_operation": operation, "keys": keys})
    before = agent.stats()
    start = time.perf_counter_ns()
    handle = getattr(agent, operation)(items)
    submitted = time.perf_counter_ns()
    statuses = None
    try:
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            statuses = agent.poll(handle)
            if statuses is not None:
                break
            time.sleep(0.00005)
        if statuses is None:
            append(raw, {"phase": phase, "operation": "poll_watchdog",
                         "native_operation": operation, "keys": keys,
                         "elapsed_ns": time.perf_counter_ns() - start,
                         "observed_statuses": None})
            raise TimeoutError("poll exceeded harness operation deadline")
        terminal = time.perf_counter_ns()
        append(raw, {"phase": phase, "operation": "batch_terminal",
                     "native_operation": operation, "keys": keys,
                     "statuses": statuses, "terminal_ns": terminal - start})
    finally:
        # Buffer contents/registration remain live through quiescent release,
        # including timeout/error paths. The supervisor bounds a stuck release.
        release_start = time.perf_counter_ns()
        agent.release(handle)
    released = time.perf_counter_ns()
    record = {"phase": phase, "operation": operation,
              "keys": keys, "statuses": statuses,
              "submit_ns": submitted - start, "terminal_ns": terminal - start,
              "release_ns": released - release_start,
              "journal_gap_ns": release_start - terminal,
              "elapsed_ns": terminal - start + released - release_start,
              "wall_elapsed_ns": released - start,
              "counter_deltas": counter_delta(agent.stats(), before)}
    append(raw, record)
    if len(statuses) != len(items):
        raise AssertionError("native status count differs from submitted batch")
    return record


class Buffer:
    def __init__(self, agent, pages, payload):
        self.agent = agent
        self.payload = payload
        self.mapping = mmap.mmap(-1, pages * payload)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        self.token = agent.register_memory(self.address, pages * payload)

    def fill(self, keys):
        expected = []
        for index, key in enumerate(keys):
            seed = hashlib.sha256(key.encode()).digest()
            content = (seed * ((self.payload + len(seed) - 1) // len(seed)))[:self.payload]
            ctypes.memmove(self.address + index * self.payload, content, self.payload)
            expected.append(hashlib.sha256(content).hexdigest())
        return expected

    def items(self, keys):
        return [{"key": key, "segments": [(self.token, i * self.payload, self.payload)]}
                for i, key in enumerate(keys)]

    def load(self, keys, expected, args, raw, phase):
        ctypes.memset(self.address, 0xCD, len(keys) * self.payload)
        record = batch(self.agent, "batch_load", self.items(keys),
                       args.operation_watchdog_seconds, raw, phase)
        validation = []
        for index, (key, status, digest) in enumerate(zip(keys, record["statuses"], expected)):
            if status == "success":
                view = memoryview(self.mapping)[index * self.payload:(index + 1) * self.payload]
                try:
                    actual = hashlib.sha256(view).hexdigest()
                finally:
                    view.release()
                validation.append({"key": key, "expected_sha256": digest,
                                   "actual_sha256": actual, "matches": actual == digest})
        append(raw, {"phase": phase, "operation": "verify_successful_reads",
                     "validation": validation})
        if any(not item["matches"] for item in validation):
            raise AssertionError("successful native load returned stale/incorrect bytes")
        record["successful_reads_verified"] = len(validation)
        return record

    def close(self):
        self.agent.deregister_memory(self.token)
        self.mapping.close()


def exists(agent, keys):
    result = []
    for start in range(0, len(keys), 128):
        result.extend(agent.batch_exists(keys[start:start + 128]))
    if len(result) != len(keys):
        raise AssertionError("native exists status count differs from submitted keys")
    return result


def run_worker(args, output, run_id):
    from nixlshard import Agent, __build_marker__, __file__ as package
    raw = output / "batches.jsonl"
    padded_payload = ((args.payload_bytes + args.unit_bytes - 1) //
                      args.unit_bytes * args.unit_bytes)
    usable_units = (args.capacity_bytes - args.metadata_bytes) // args.unit_bytes
    capacity_pages = usable_units // (padded_payload // args.unit_bytes)
    cfg = {"name": "saturation-" + run_id,
           "disks": [{"path": str(output / "disposable.ssd"),
                      "capacity_bytes": args.capacity_bytes,
                      "unit_bytes": args.unit_bytes,
                      "metadata_bytes": args.metadata_bytes, "create": True}],
           "workers": 2, "staging_slots": 4,
           "staging_slot_bytes": args.staging_slot_bytes,
           "max_inflight": 1, "timeout_ms": args.timeout_ms,
           "direct_io": args.direct_io}
    provenance = {"schema": "nixlshard-native-saturation-v1",
                  "run_id": run_id, "build_marker": __build_marker__,
                  "tool_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  "native_package": package, "python": sys.version,
                  "platform": platform.platform(), "arguments": vars(args),
                  "agent_config": cfg, "usable_units": usable_units,
                  "capacity_pages": capacity_pages,
                  "padded_payload_bytes": padded_payload,
                  "one_batch_outstanding": True,
                  "batch_latency_scope": "submit-through-terminal plus release-call elapsed; terminal-journal gap separately recorded and excluded; wall_elapsed_ns includes it",
                  "incremental_gcs_prefix": args.gcs_prefix,
                  "upload_pacing_scope": "copy/zip/upload gaps after fill chunks and measured iterations; compare identical upload flags across native prefixes",
                  "scope": "local debug-file cache under reuse; not SGLang/TTFT or physical SSD claims",
                  "timer_scope": "aggregate native worker/background elapsed intervals overlap; do not sum into request elapsed"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    agent = None
    buffer = None
    history = []
    measured = []
    fill_records = []
    try:
        archives = archive_client(args.gcs_prefix, output) if args.gcs_prefix else None
        agent = Agent(cfg)
        buffer = Buffer(agent, max(args.batch_pages + [args.fill_batch_pages]), args.payload_bytes)
        for offset in range(0, capacity_pages, args.fill_batch_pages):
            count = min(args.fill_batch_pages, capacity_pages - offset)
            keys = [f"{run_id}:fill:{offset + i:08d}" for i in range(count)]
            digests = buffer.fill(keys)
            record = batch(agent, "batch_store", buffer.items(keys),
                           args.operation_watchdog_seconds, raw, "fill")
            fill_records.append(record)
            history.extend(zip(keys, digests))
            # Fill success is required before calling subsequent trials saturated.
            if any(status != "success" for status in record["statuses"]):
                raise RuntimeError("initial fill incomplete; see raw status artifacts")
            verified = buffer.load(keys, digests, args, raw, "fill_verify")
            if any(status != "success" for status in verified["statuses"]):
                raise AssertionError("freshly filled keys were not retrievable")
            if archives is not None:
                archive_metadata(archives, output, f"fill-{offset:08d}.zip")
        fill_stats = agent.stats()
        append(raw, {"phase": "fill_complete", "known_capacity_pages": capacity_pages,
                     "stats": fill_stats})
        for pages in args.batch_pages:
            for iteration in range(args.iterations):
                keys = [f"{run_id}:reuse:{pages}:{iteration}:{i:08d}" for i in range(pages)]
                digests = buffer.fill(keys)
                store = batch(agent, "batch_store", buffer.items(keys),
                              args.operation_watchdog_seconds, raw, "saturated_store")
                load = buffer.load(keys, digests, args, raw, "saturated_load")
                history.extend(zip(keys, digests))
                measured.append({"batch_pages": pages, "iteration": iteration,
                                 "store": store, "load": load})
                if archives is not None:
                    archive_metadata(archives, output, f"reuse-{pages}-{iteration:04d}.zip")
        final_stats = agent.stats()
        checkpoint_status = checkpoint(agent, args, raw)
        if checkpoint_status != "success":
            raise RuntimeError("explicit checkpoint failed")
        all_keys = [key for key, digest in history]
        before_restart = exists(agent, all_keys)
        buffer.close()
        buffer = None
        agent.close()
        agent = None
        cfg["disks"][0]["create"] = False
        agent = Agent(cfg)
        buffer = Buffer(agent, max(args.batch_pages + [args.fill_batch_pages]), args.payload_bytes)
        after_restart = exists(agent, all_keys)
        append(raw, {"phase": "restart_index", "keys": all_keys,
                     "before": before_restart, "after": after_restart})
        if before_restart != after_restart:
            raise AssertionError("explicitly checkpointed live mappings changed on restart")
        latest = [(key, digest) for (key, digest), live in zip(history, after_restart) if live]
        probe_count = max(args.batch_pages)
        probe = latest[-probe_count:]
        if probe:
            verified = buffer.load([key for key, digest in probe], [digest for key, digest in probe],
                                   args, raw, "restart_latest_verify")
            if any(status != "success" for status in verified["statuses"]):
                raise AssertionError("retained checkpointed keys were not retrievable")
        oldest = history[:probe_count]
        buffer.load([key for key, digest in oldest], [digest for key, digest in oldest],
                    args, raw, "restart_oldest_probe")
        groups = {}
        for pages in args.batch_pages:
            group = [item for item in measured if item["batch_pages"] == pages]
            groups[str(pages)] = {}
            for operation in ("store", "load"):
                durations = sorted(item[operation]["elapsed_ns"] / 1e6 for item in group)
                counts = collections.Counter(status for item in group
                                             for status in item[operation]["statuses"])
                groups[str(pages)][operation] = {
                    "samples": len(group), "elapsed_ms_samples": durations,
                    "median_ms": statistics.median(durations),
                    "p95_ms": durations[max(0, (95 * len(durations) + 99) // 100 - 1)],
                    "statuses": dict(counts)}
        summary = {"completed": True, "correctness_passed": True,
                   "percentile_method": "nearest-rank for p95; median_ms uses statistics.median",
                   "all_measured_stores_success": all(status == "success"
                      for item in measured for status in item["store"]["statuses"]),
                   "all_measured_loads_success": all(status == "success"
                      for item in measured for status in item["load"]["statuses"]),
                   "fill_pages": capacity_pages,
                   "evictions_during_measured": final_stats.get("evictions", 0) - fill_stats.get("evictions", 0),
                   "oldest_probe_live_after_restart": sum(after_restart[:probe_count]),
                   "checkpoint_status": checkpoint_status,
                   "restart_mapping_equality": before_restart == after_restart,
                   "restart_latest_verified": len(probe),
                   "groups": groups,
                   "measurement_counter_deltas": counter_delta(final_stats, fill_stats),
                   "limitations": ["non-success statuses are explicit cache failures/misses, not verified reads",
                                   "hash verification is outside the native batch latency interval",
                                   "counter timers overlap, and metadata sync/maintenance can affect throughput",
                                   "successful reads checked against distinct immutable key content"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except BaseException as error:
        append(raw, {"phase": "error", "error": repr(error)})
        (output / "summary.json").write_text(json.dumps({
            "completed": False, "correctness_passed": False, "error": repr(error),
            "fill_batches_completed": len(fill_records),
            "measured_batches_completed": len(measured)}, indent=2) + "\n")
        raise
    finally:
        try:
            if buffer is not None:
                buffer.close()
        finally:
            if agent is not None:
                agent.close()
        if not args.keep_mock_file:
            (output / "disposable.ssd").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/raid/nixlshard-v2/saturation")
    parser.add_argument("--capacity-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--metadata-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--unit-bytes", type=int, default=64 * 1024)
    parser.add_argument("--payload-bytes", type=int, default=16 * 1024**2)
    parser.add_argument("--staging-slot-bytes", type=int, default=32 * 1024**2)
    parser.add_argument("--batch-pages", type=lambda x: [int(v) for v in x.split(",")], default=[32, 64])
    parser.add_argument("--fill-batch-pages", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--operation-watchdog-seconds", type=float, default=30)
    parser.add_argument("--overall-watchdog-seconds", type=float, default=1800)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--keep-mock-file", action="store_true")
    parser.add_argument("--gcs-prefix", help="approved gs://bucket/unique-run-prefix; metadata-only incremental immutable ZIP uploads")
    args = parser.parse_args()
    positive = [args.capacity_bytes, args.metadata_bytes, args.unit_bytes,
                args.payload_bytes, args.staging_slot_bytes, args.fill_batch_pages,
                args.iterations, args.timeout_ms, args.operation_watchdog_seconds,
                args.overall_watchdog_seconds]
    if any(value <= 0 for value in positive) or not args.batch_pages:
        parser.error("geometry, counts and watchdogs must be positive")
    if args.unit_bytes < 512 or args.unit_bytes & (args.unit_bytes - 1):
        parser.error("unit bytes must be a power of two >=512")
    if args.metadata_bytes < 16384 or args.metadata_bytes % max(4096, args.unit_bytes):
        parser.error("metadata must be >=16384 and aligned to unit bytes and 4096")
    padded = ((args.payload_bytes + args.unit_bytes - 1) // args.unit_bytes * args.unit_bytes)
    capacity_pages = (args.capacity_bytes - args.metadata_bytes) // padded
    if capacity_pages <= max(args.batch_pages + [args.fill_batch_pages]):
        parser.error("usable cache must exceed each batch size")
    if padded > args.staging_slot_bytes or args.staging_slot_bytes % args.unit_bytes:
        parser.error("aligned staging slot must fit padded payload")
    if any(value < 1 or value > 128 for value in args.batch_pages + [args.fill_batch_pages]):
        parser.error("batches must contain 1..128 pages")
    if len(set(args.batch_pages)) != len(args.batch_pages):
        parser.error("batch sizes must be distinct")
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    output = pathlib.Path(args.directory).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"artifact_dir": str(output), "watchdog_seconds": args.overall_watchdog_seconds}), flush=True)
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker, args=(args, output, run_id))
    process.start()
    process.join(args.overall_watchdog_seconds)
    watchdog_fired = False
    if process.is_alive():
        watchdog_fired = True
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        append(output / "batches.jsonl", {"phase": "supervisor_watchdog", "terminated": True})
        (output / "summary.json").write_text(json.dumps({
            "completed": False, "correctness_passed": False,
            "error": "overall watchdog exceeded; child terminated; disposable file retained for inspection"}, indent=2) + "\n")
    if not watchdog_fired and process.exitcode != 0:
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary.update({"completed": False, "correctness_passed": False,
                        "process_exitcode": process.exitcode})
        summary.setdefault("error", "worker failed during initialization or cleanup")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.gcs_prefix:
        uploader = context.Process(target=archive_final, args=(args.gcs_prefix, output))
        uploader.start()
        uploader.join(args.operation_watchdog_seconds)
        if uploader.is_alive():
            uploader.terminate()
            uploader.join(5)
            if uploader.is_alive():
                uploader.kill()
                uploader.join(5)
        if uploader.exitcode != 0:
            append(output / "batches.jsonl", {"phase": "final_archive_failed", "exitcode": uploader.exitcode})
            if not watchdog_fired:
                return 1
    if watchdog_fired:
        return 124
    return 0 if process.exitcode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
