#!/usr/bin/env python3
"""Measure client streaming TTFT with verified cold/SSD/host/GPU/remote cache sources.

Use a dedicated, already-running SGLang server with metrics enabled, stream
interval one, immutable NIXLShard model revision, and enough storage capacity.
This script serializes all generation requests. Preparation and HTTP metric
scrapes are outside measured requests. Host preparation uses explicit cache
pressure, because the pinned SGLang HTTP API cannot evict only GPU cache.

HiCache 'direct' denotes GPU/host movement, independently of agent.direct_io.
Cache-level TTFT differences do not measure attributable NIXLShard overhead.
"""

import argparse
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.parse
import uuid
import zipfile


class GcsArchives:
    """One immutable archive per completed sample, outside measured generation."""

    def __init__(self, prefix, artifact_dir):
        parsed = urllib.parse.urlsplit(prefix)
        if (
            parsed.scheme != "gs"
            or not parsed.netloc
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
            or ".." in parsed.path.split("/")
        ):
            raise ValueError("GCS prefix must be gs://bucket/unique-run-prefix")
        from google.cloud import storage

        self.bucket = storage.Client().bucket(parsed.netloc)
        self.prefix = parsed.path.strip("/") + "/" + artifact_dir.name
        self.artifact_dir = artifact_dir
        self.uploads = []

    def archive(self, directory, name, recursive=True):
        stream = io.BytesIO()
        paths = directory.rglob("*") if recursive else directory.iterdir()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                if path.is_file():
                    archive.writestr(
                        str(path.relative_to(directory)), path.read_bytes()
                    )
        body = stream.getvalue()
        blob = self.bucket.blob(self.prefix + "/" + name)
        blob.upload_from_string(
            body, content_type="application/zip", if_generation_match=0
        )
        self.uploads.append(
            {
                "uri": "gs://" + self.bucket.name + "/" + blob.name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
        )
        (self.artifact_dir / "gcs-archives.json").write_text(
            json.dumps(self.uploads, indent=2) + "\n"
        )


class Client:
    def __init__(self, base, timeout=120):
        parsed = urllib.parse.urlsplit(base)
        if parsed.scheme not in ("http", "https") or parsed.path not in ("", "/"):
            raise ValueError("base URL must be an http(s) origin")
        connection = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection(parsed.hostname, parsed.port, timeout=timeout)
        self.timeout = timeout

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        # The other server can be idle longer than Uvicorn's keepalive timeout
        # while generation runs. Retry safe observations/upper-tier flush once
        # on a fresh connection, never retry generation or arbitrary POSTs.
        safe_retry = body is None or path.split("?", 1)[0] == "/flush_cache"
        for attempt in range(2 if safe_retry else 1):
            try:
                self.connection.request(
                    "GET" if body is None else "POST", path, body,
                    {"Content-Type": "application/json"},
                )
                response = self.connection.getresponse()
                raw = response.read().decode()
                break
            except (http.client.HTTPException, ConnectionError, OSError):
                self.connection.close()
                if not safe_retry or attempt:
                    raise
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {path}: {raw[:2000]}")
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    def generate(self, payload, incremental=False):
        body = json.dumps(dict(payload, stream=True)).encode()
        # Include network submission/response delay, exclude JSON encoding/preparation.
        start_ns = time.perf_counter_ns()
        self.connection.request(
            "POST",
            "/generate",
            body,
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = self.connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"generation HTTP {response.status}: {response.read()[:2000]!r}"
            )
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise RuntimeError("generation did not return a streaming SSE response")
        deadline = time.monotonic() + self.timeout

        def lines():
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("generation stream exceeded request deadline")
                if self.connection.sock is not None:
                    self.connection.sock.settimeout(remaining)
                line = response.readline(1 << 20)
                if not line:
                    break
                yield line, time.perf_counter_ns()

        try:
            result = collect_stream(lines(), start_ns, incremental)
            # Consume trailing bytes before reusing the connection.
            response.read()
            return result
        finally:
            response.close()

    def close(self):
        self.connection.close()


def sse_events(lines):
    """Yield complete SSE data fields and their client receive timestamps."""
    fields, bytes_in_event, last_ns = [], 0, None
    for raw, received_ns in lines:
        bytes_in_event += len(raw)
        if bytes_in_event > (1 << 20):
            raise ValueError("SSE event exceeds one MiB")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if fields:
                yield "\n".join(fields), received_ns
            fields, bytes_in_event = [], 0
        elif line.startswith("data:"):
            fields.append(line[5:].removeprefix(" "))
        last_ns = received_ns
    if fields:
        raise ValueError(f"truncated SSE event at {last_ns}")


def collect_stream(lines, start_ns, incremental=False):
    first_ns, last, frames, output_ids, text, done_ns = None, {}, [], [], "", None
    for data, received_ns in sse_events(lines):
        if data == "[DONE]":
            done_ns = received_ns
            break
        event = json.loads(data)
        if not isinstance(event, dict) or "error" in event:
            raise RuntimeError(f"invalid/error generation event: {event!r}")
        ids = event.get("output_ids") or []
        if first_ns is None and (ids or event.get("text")):
            first_ns = received_ns
        if incremental:
            output_ids.extend(ids)
            text += event.get("text") or ""
        else:
            if ids:
                # Pinned SGLang defaults to cumulative streaming output.
                if ids[: len(output_ids)] != output_ids:
                    raise ValueError(
                        "cumulative output IDs changed; verify incremental streaming mode"
                    )
                output_ids = list(ids)
            if event.get("text") is not None:
                text = event["text"]
        last = event
        frames.append({"elapsed_ns": received_ns - start_ns, "event": event})
    if done_ns is None:
        raise ValueError("generation ended without SSE [DONE]")
    if first_ns is None or not output_ids:
        raise ValueError("generation produced no output token IDs")
    meta = last.get("meta_info") or {}
    if not meta.get("finish_reason"):
        raise ValueError("stream has no terminal finish reason")
    return {
        "ttft_ms": (first_ns - start_ns) / 1e6,
        "stream_latency_ms": (done_ns - start_ns) / 1e6,
        "first_token_elapsed_ns": first_ns - start_ns,
        "output_ids": output_ids,
        "text": text,
        "meta_info": meta,
        "frames": frames,
    }


def cache_details(result):
    meta = result["meta_info"]
    reported = meta.get("cached_tokens_details") or {}
    values = {
        name: int(reported.get(name, 0)) for name in ("device", "host", "storage")
    }
    values["cached_tokens"] = int(meta.get("cached_tokens", 0))
    values["storage_backend"] = reported.get("storage_backend")
    return values


def validate_cache(scenario, result, expected_prefix):
    values = cache_details(result)
    if scenario == "cold":
        if values["cached_tokens"] or any(
            values[name] for name in ("device", "host", "storage")
        ):
            raise AssertionError(f"cold request had cache hits: {values}")
    else:
        desired = {"ssd": "storage", "remote": "storage", "host": "host", "gpu": "device"}[scenario]
        if values[desired] < expected_prefix or any(
            values[name] for name in ("device", "host", "storage") if name != desired
        ):
            raise AssertionError(
                f"{scenario} request was not an isolated >= {expected_prefix}-token hit: {values}"
            )
        if scenario in ("ssd", "remote") and values["storage_backend"] != "HiCacheNixlShard":
            raise AssertionError(f"SSD hit did not identify HiCacheNixlShard: {values}")
        if scenario == "remote" and values["storage"] != expected_prefix:
            raise AssertionError(f"remote storage prefix differs from expected complete pages: {values}")
    return values


def metrics_snapshot(text):
    result = {}
    for line in text.splitlines():
        if line.startswith("#") or not any(
            word in line
            for word in (
                "backuped_tokens",
                "prefetch",
                "cached_tokens",
                "storage",
                "hicache",
                "nixlshard",
            )
        ):
            continue
        match = re.match(r"^(.+?)\s+([-+0-9.eE]+)(?:\s+\S+)?$", line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                result[match[1]] = value
    return result


NATIVE_COUNTER_FAMILIES = frozenset(
    (
        "sglang:nixlshard_component_seconds_total",
        "sglang:nixlshard_component_bytes_total",
        "sglang:nixlshard_events_total",
    )
)


def native_component_availability(metric_names, diagnostic_url=None, received=False):
    observed = sorted(
        {name.split("{", 1)[0] for name in metric_names} & NATIVE_COUNTER_FAMILIES
    )
    return {
        "prometheus_counter_families": observed,
        "prometheus_counters_available": bool(observed),
        "prometheus_all_fixed_families_observed": set(observed)
        == NATIVE_COUNTER_FAMILIES,
        "agent_diagnostic_url": diagnostic_url,
        "diagnostic_snapshot_received": received,
        "scope": "fixed cumulative native counters; aggregate worker/background intervals "
        "overlap and exclude framework/model work; do not sum into TTFT",
    }


def metric_total(snapshot, name):
    return sum(value for key, value in snapshot.items() if key.split("{", 1)[0] == name)


def metric_deltas(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def native_bytes(snapshot, component):
    series = [value for key, value in snapshot.items()
              if key.startswith("sglang:nixlshard_component_bytes_total{")
              and f'component="{component}"' in key]
    if not series:
        raise ValueError(f"native byte counter missing for {component}")
    return sum(series)


def remote_counter_proof(requester, owner, expected_bytes):
    observed = {
        "requester_remote_read_bytes": native_bytes(requester, "remote_read"),
        "requester_staging_copy_bytes": native_bytes(requester, "staging_copy"),
        "requester_posix_read_bytes": native_bytes(requester, "posix_read"),
        "requester_ucx_write_bytes": native_bytes(requester, "ucx_write"),
        "requester_posix_write_bytes": native_bytes(requester, "posix_write"),
        "owner_posix_read_bytes": native_bytes(owner, "posix_read"),
        "owner_ucx_write_bytes": native_bytes(owner, "ucx_write"),
        "owner_posix_write_bytes": native_bytes(owner, "posix_write"),
        "owner_staging_copy_bytes": native_bytes(owner, "staging_copy"),
    }
    expected = {key: (0 if key in ("requester_posix_read_bytes", "requester_ucx_write_bytes",
                                 "requester_posix_write_bytes", "owner_posix_write_bytes",
                                 "owner_staging_copy_bytes")
                      else expected_bytes) for key in observed}
    if any(value < 0 or value > expected[key] for key, value in observed.items()):
        raise AssertionError(f"remote counter window has incorrect/extra payload I/O: {observed}")
    return {"expected_bytes": expected_bytes, "observed": observed,
            "exact": observed == expected,
            "scope": "cumulative native bytes through full generation/export settling; not a first-token timing window"}


def percentile(values, percent):
    """Linear interpolation on sorted samples; raw samples remain authoritative."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Experiment:
    def __init__(
        self, args, client, tokenizer, artifact_dir, incremental, archives=None
    ):
        self.args, self.client, self.tokenizer = args, client, tokenizer
        self.artifact_dir, self.incremental = artifact_dir, incremental
        self.sequence, self.samples = 0, []
        self.archives = archives
        self.native_prometheus_families = set()
        self.native_diagnostic_received = False

    def prompt(self, length):
        # The nonce appears before the first complete cache page: no shared page keys.
        prefix = self.tokenizer.encode(
            f"Probe {uuid.uuid4().hex}: ", add_special_tokens=False
        )
        body = self.tokenizer.encode(
            "The cache preserves a complete immutable sequence of key value tensors. ",
            add_special_tokens=False,
        )
        if not body or len(prefix) >= length:
            raise ValueError("context length is too small for a unique prefix")
        return (prefix + body * ((length // len(body)) + 2))[:length]

    def payload(self, ids, generated=None):
        return {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": generated or self.args.output_tokens,
                "ignore_eos": True,
            },
        }

    def snapshot(self, directory, label, client=None):
        client = client or self.client
        metrics = client.request("/metrics")
        (directory / f"{label}-metrics.txt").write_text(metrics)
        result = {"metrics": metrics_snapshot(metrics)}
        diagnostic_url = self.args.native_stats_url if client is self.client else None
        if diagnostic_url:
            result["native_stats"] = client.request(diagnostic_url)
            self.native_diagnostic_received = True
        result["native_component_counters"] = native_component_availability(
            result["metrics"],
            diagnostic_url,
            self.native_diagnostic_received,
        )
        self.native_prometheus_families.update(
            result["native_component_counters"]["prometheus_counter_families"]
        )
        (directory / f"{label}-stats.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result

    def flush(self, client=None):
        result = (client or self.client).request("/flush_cache?timeout=10", {})
        if "Cache flushed" not in str(result):
            raise RuntimeError(f"upper-tier flush failed: {result}")
        return result

    def await_backup(self, before, expected, client=None):
        baseline = metric_total(before["metrics"], "sglang:backuped_tokens_total")
        deadline = time.monotonic() + self.args.backup_timeout
        last_total, stable_since = None, None
        while time.monotonic() < deadline:
            snapshot = metrics_snapshot((client or self.client).request("/metrics"))
            total = metric_total(snapshot, "sglang:backuped_tokens_total")
            if total - baseline >= expected:
                if total != last_total:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= self.args.settle_seconds:
                    return {
                        "baseline": baseline,
                        "completed": total,
                        "settle_seconds": self.args.settle_seconds,
                    }
            last_total = total
            time.sleep(0.05)
        raise TimeoutError("seed backup did not complete and settle before SSD flush")

    def generate_record(self, directory, role, ids, client=None, incremental=None):
        result = (client or self.client).generate(
            self.payload(ids), self.incremental if incremental is None else incremental
        )
        (directory / f"{role}.json").write_text(json.dumps(result, indent=2) + "\n")
        if len(result["output_ids"]) != self.args.output_tokens:
            raise AssertionError(
                "generation did not produce the configured number of output tokens"
            )
        if result["meta_info"].get("finish_reason", {}).get("type") != "length":
            raise AssertionError(
                "generation did not finish at the configured output length"
            )
        return result

    def pressure(self, directory):
        remaining = self.args.host_pressure_tokens
        requests = []
        while remaining > 0:
            length = min(remaining, self.args.pressure_context)
            if length <= self.args.page_size:
                length = self.args.page_size + 1
            result = self.client.generate(
                self.payload(self.prompt(length), 1), self.incremental
            )
            requests.append(
                {
                    "prompt_tokens": result["meta_info"].get("prompt_tokens"),
                    "cache": cache_details(result),
                    "output_ids": result["output_ids"],
                }
            )
            remaining -= length
        (directory / "host-pressure.json").write_text(
            json.dumps(requests, indent=2) + "\n"
        )
        time.sleep(self.args.settle_seconds)
        return requests

    def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
        self.sequence += 1
        directory = self.artifact_dir / f"{self.sequence:04d}-{context}-{scenario}"
        directory.mkdir()
        ids = ids if ids is not None else self.prompt(context)
        (directory / "request.json").write_text(
            json.dumps(self.payload(ids), indent=2) + "\n"
        )
        expected = ((context - 1) // self.args.page_size) * self.args.page_size
        record = {
            "scenario": scenario,
            "context_tokens": context,
            "repeat": repetition,
            "warmup": warmup,
            "artifact": directory.name,
            "passed": False,
        }
        try:
            if scenario == "cold":
                record["preparation"] = {"flush": self.flush(), "unique_prefix": True}
            elif scenario in ("ssd", "host"):
                self.flush()
                seed_stats = self.snapshot(directory, "before-seed")
                reference = self.generate_record(directory, "seed", ids)
                validate_cache("cold", reference, expected)
                record["backup"] = self.await_backup(seed_stats, expected)
                if scenario == "ssd":
                    record["preparation"] = {
                        "flush": self.flush(),
                        "storage_preserved": True,
                    }
                else:
                    record["preparation"] = {
                        "pressure": self.pressure(directory),
                        "gpu_capacity_tokens": self.args.gpu_capacity_tokens,
                    }
            elif reference is None:
                self.flush()
                seed_stats = self.snapshot(directory, "before-seed")
                reference = self.generate_record(directory, "seed", ids)
                validate_cache("cold", reference, expected)
                record["backup"] = self.await_backup(seed_stats, expected)
                record["preparation"] = {"no_generation_since_seed": True}
            else:
                # Wait for the preceding cold request's backup without evicting GPU KV.
                record["backup"] = self.await_backup(
                    self.samples[-1]["before"], expected
                )
                record["preparation"] = {"no_generation_since_seed": True}
            before = self.snapshot(directory, "before")
            result = self.generate_record(directory, "measured", ids)
            if result["meta_info"].get("prompt_tokens") != context:
                raise AssertionError(
                    "served prompt token count differs from configured context"
                )
            after = self.snapshot(directory, "after")
            record.update(
                {
                    "ttft_ms": result["ttft_ms"],
                    "stream_latency_ms": result["stream_latency_ms"],
                    "cache": validate_cache(scenario, result, expected),
                    "before": before,
                    "after": after,
                    "metric_deltas": {
                        key: value - before["metrics"].get(key, 0)
                        for key, value in after["metrics"].items()
                    },
                    "output_sha256": hashlib.sha256(
                        json.dumps(result["output_ids"]).encode()
                    ).hexdigest(),
                }
            )
            if reference is not None:
                if (
                    result["output_ids"] != reference["output_ids"]
                    or result["text"] != reference["text"]
                ):
                    raise AssertionError(
                        "cached replay output differs from its deterministic cold seed"
                    )
                record["output_matches_cold"] = True
            record["passed"] = True
            return ids, result
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            self.samples.append(record)
            (directory / "sample.json").write_text(json.dumps(record, indent=2) + "\n")
            with (self.artifact_dir / "samples.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            if self.archives is not None:
                self.archives.archive(directory, directory.name + ".zip")

    def summary(self):
        groups = {}
        for sample in self.samples:
            if not sample["warmup"] and sample["passed"]:
                key = f"{sample['context_tokens']}:{sample['scenario']}"
                groups.setdefault(key, []).append(sample["ttft_ms"])
        return {
            key: {
                "count": len(values),
                "ttft_ms_p50": percentile(values, 50),
                "ttft_ms_p95": percentile(values, 95),
                "ttft_ms_samples": values,
            }
            for key, values in groups.items()
        }


class RemoteExperiment(Experiment):
    """Matched requester cold control, owner seed, then validated remote replay."""

    def __init__(self, *args, owner_client, owner_incremental, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner_client = owner_client
        self.owner_incremental = owner_incremental

    @staticmethod
    def journal(directory, phase, **details):
        with (directory / "remote-journal.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": phase, "monotonic_ns": time.monotonic_ns(),
                                     **details}) + "\n")
            stream.flush()

    def archive_phase(self, directory, phase):
        if self.archives is not None:
            self.archives.archive(directory, directory.name + "-" + phase + ".zip")

    def settle_remote(self, directory, requester_before, owner_before,
                      expected_bytes, role):
        start = time.monotonic()
        deadline = start + self.args.backup_timeout
        poll = 0
        while True:
            label = role if not poll else f"{role}-poll-{poll:03d}"
            requester_after = self.snapshot(directory, label)
            owner_after = self.snapshot(directory, "owner-" + label, self.owner_client)
            requester_delta = metric_deltas(requester_before["metrics"], requester_after["metrics"])
            owner_delta = metric_deltas(owner_before["metrics"], owner_after["metrics"])
            proof = remote_counter_proof(requester_delta, owner_delta, expected_bytes)
            self.journal(directory, "remote_counter_observation", role=role,
                         observation=poll, proof=proof)
            if proof["exact"]:
                proof["export_settle_seconds"] = time.monotonic() - start
                proof["requester_metric_deltas"] = requester_delta
                proof["owner_metric_deltas"] = owner_delta
                return requester_after, owner_after, proof
            if time.monotonic() >= deadline:
                raise TimeoutError("remote payload counters never reached the exact expected transfer")
            time.sleep(0.05)
            poll += 1

    def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
        if scenario != "remote":
            raise ValueError("remote experiment only supports the remote scenario")
        self.sequence += 1
        directory = self.artifact_dir / f"{self.sequence:04d}-{context}-remote"
        directory.mkdir()
        ids = self.prompt(context) if ids is None else ids
        (directory / "request.json").write_text(json.dumps(self.payload(ids), indent=2) + "\n")
        expected_tokens = ((context - 1) // self.args.page_size) * self.args.page_size
        expected_bytes = expected_tokens // self.args.page_size * self.args.kv_bytes_per_page
        record = {"scenario": "remote", "context_tokens": context,
                  "repeat": repetition, "warmup": warmup, "artifact": directory.name,
                  "passed": False, "discovery_preflight": []}
        try:
            self.journal(directory, "cold_control_started")
            self.flush()
            cold_before = self.snapshot(directory, "cold-control-before")
            cold = self.generate_record(directory, "cold-control", ids)
            cold_after = self.snapshot(directory, "cold-control-after")
            cold_cache = validate_cache("cold", cold, expected_tokens)
            cold_delta = metric_deltas(cold_before["metrics"], cold_after["metrics"])
            if any(native_bytes(cold_delta, component) != 0
                   for component in ("remote_read", "posix_read", "ucx_write")):
                raise AssertionError("cold control performed payload reads or UCX writes")
            record.update(cold_control_ttft_ms=cold["ttft_ms"], cold_control_cache=cold_cache,
                          cold_control_metric_deltas=cold_delta)
            self.journal(directory, "cold_control_completed", ttft_ms=cold["ttft_ms"])

            self.journal(directory, "owner_seed_started")
            self.flush(self.owner_client)
            owner_seed_before = self.snapshot(directory, "owner-before-seed", self.owner_client)
            reference = self.generate_record(directory, "owner-seed", ids, self.owner_client,
                                             self.owner_incremental)
            validate_cache("cold", reference, expected_tokens)
            if reference["output_ids"] != cold["output_ids"] or reference["text"] != cold["text"]:
                raise AssertionError("owner seed differs from requester deterministic cold control")
            backup_tokens = ((context + self.args.output_tokens - 1) //
                             self.args.page_size) * self.args.page_size
            record["owner_backup"] = self.await_backup(owner_seed_before, backup_tokens,
                                                       self.owner_client)
            record["owner_expected_backup_tokens"] = backup_tokens
            self.snapshot(directory, "owner-after-seed", self.owner_client)
            self.journal(directory, "owner_seed_completed", backup=record["owner_backup"])
            self.archive_phase(directory, "owner-seed")

            # Unknown native MD keys are resolved asynchronously. Preserve every
            # discovery request, exclude it from measured TTFT, and flush upper
            # tiers before each attempt. Requester disks=[] prevents local SSD hits.
            deadline = time.monotonic() + self.args.backup_timeout
            ready = False
            for attempt in range(1, self.args.remote_discovery_attempts + 1):
                if time.monotonic() >= deadline:
                    break
                self.journal(directory, "discovery_started", attempt=attempt)
                self.flush()
                before = self.snapshot(directory, f"discovery-{attempt:03d}-before")
                owner_before = self.snapshot(directory, f"owner-discovery-{attempt:03d}-before",
                                             self.owner_client)
                probe = self.generate_record(directory, f"discovery-{attempt:03d}", ids)
                if probe["output_ids"] != reference["output_ids"] or probe["text"] != reference["text"]:
                    raise AssertionError("discovery output differs from deterministic cold output")
                detail = {"attempt": attempt, "cache": cache_details(probe),
                          "ttft_ms": probe["ttft_ms"]}
                record["discovery_preflight"].append(detail)
                try:
                    validate_cache("remote", probe, expected_tokens)
                except AssertionError as error:
                    detail["not_ready"] = str(error)
                    self.journal(directory, "discovery_not_ready", **detail)
                    time.sleep(self.args.settle_seconds)
                    continue
                _, _, proof = self.settle_remote(
                    directory, before, owner_before, expected_bytes,
                    f"discovery-{attempt:03d}-after"
                )
                detail["remote_proof"] = proof
                self.journal(directory, "discovery_ready", attempt=attempt)
                ready = True
                break
            if not ready:
                raise TimeoutError("requester never observed a complete isolated remote cache hit")
            self.archive_phase(directory, "discovery-ready")

            self.flush()
            time.sleep(self.args.settle_seconds)
            requester_before = self.snapshot(directory, "before")
            owner_before = self.snapshot(directory, "owner-before", self.owner_client)
            self.journal(directory, "measured_remote_started")
            result = self.generate_record(directory, "measured", ids)
            self.journal(directory, "measured_remote_stream_completed", ttft_ms=result["ttft_ms"])
            if result["meta_info"].get("prompt_tokens") != context:
                raise AssertionError("served prompt tokens differ from requested context")
            cache = validate_cache("remote", result, expected_tokens)
            after, owner_after, proof = self.settle_remote(
                directory, requester_before, owner_before, expected_bytes, "after"
            )
            if result["output_ids"] != reference["output_ids"] or result["text"] != reference["text"]:
                raise AssertionError("remote replay differs from requester/owner cold output")
            record.update(ttft_ms=result["ttft_ms"], stream_latency_ms=result["stream_latency_ms"],
                          cache=cache, before=requester_before, after=after,
                          owner_before=owner_before, owner_after=owner_after,
                          metric_deltas=proof["requester_metric_deltas"],
                          owner_metric_deltas=proof["owner_metric_deltas"],
                          remote_proof=proof, output_matches_cold=True,
                          output_sha256=hashlib.sha256(json.dumps(result["output_ids"]).encode()).hexdigest(),
                          passed=True)
            return ids, result
        except BaseException as error:
            record["error"] = repr(error)
            self.journal(directory, "failed", error=record["error"])
            raise
        finally:
            self.samples.append(record)
            (directory / "sample.json").write_text(json.dumps(record, indent=2) + "\n")
            with (self.artifact_dir / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            if self.archives is not None:
                self.archives.archive(directory, directory.name + ".zip")

    def summary(self):
        groups = super().summary()
        for context in self.args.contexts:
            values = [sample["cold_control_ttft_ms"] for sample in self.samples
                      if sample["context_tokens"] == context and sample["passed"]
                      and not sample["warmup"]]
            if values:
                groups[f"{context}:cold-control"] = {
                    "count": len(values), "ttft_ms_p50": percentile(values, 50),
                    "ttft_ms_p95": percentile(values, 95), "ttft_ms_samples": values,
                    "scope": "same requester, unique prefix before owner seeding; no L3 hit, backend still enabled"}
        return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31001")
    parser.add_argument("--owner-base-url", help="second SGLang endpoint used to seed actual remote KV")
    parser.add_argument("--owner-provenance-json", type=Path)
    parser.add_argument("--kv-bytes-per-page", type=int, default=16 * 1024**2,
                        help="logical page payload for exact remote byte proof")
    parser.add_argument("--remote-discovery-attempts", type=int, default=8)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--artifact-dir", type=Path, required=True, help="new unique directory"
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        required=True,
        help="server launch/runtime/config/build-marker evidence; excludes credentials",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["cold", "ssd", "host", "gpu", "remote"],
        default=["cold", "ssd", "gpu"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--backup-timeout", type=float, default=60)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--host-pressure-tokens", type=int, default=0)
    parser.add_argument("--gpu-capacity-tokens", type=int, default=0)
    parser.add_argument("--pressure-context", type=int, default=2048)
    parser.add_argument(
        "--stream-output", choices=["auto", "cumulative", "incremental"], default="auto"
    )
    parser.add_argument(
        "--native-stats-url",
        help="optional HTTP path providing native counter snapshots",
    )
    parser.add_argument(
        "--gcs-prefix",
        help="gs://bucket/unique-run-prefix for incremental immutable sample archives (ADC credentials)",
    )
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.warmups < 0
        or args.output_tokens < 1
        or args.page_size < 1
    ):
        parser.error("invalid repeats/warmups/output/page settings")
    if any(context <= args.page_size for context in args.contexts):
        parser.error("every context must exceed one cache page")
    if "remote" in args.scenarios:
        if args.scenarios != ["remote"] or not args.owner_base_url or not args.owner_provenance_json:
            parser.error("remote requires an exclusive --scenarios remote profile and both owner options")
        if args.owner_base_url.rstrip("/") == args.base_url.rstrip("/"):
            parser.error("remote owner and requester endpoints must differ")
        if args.kv_bytes_per_page <= 0 or args.remote_discovery_attempts < 1:
            parser.error("remote bytes/page and discovery attempts must be positive")
    elif args.owner_base_url or args.owner_provenance_json:
        parser.error("owner options require --scenarios remote")
    if "host" in args.scenarios and (
        args.gpu_capacity_tokens <= 0
        or args.host_pressure_tokens <= args.gpu_capacity_tokens
        or args.pressure_context <= args.page_size
    ):
        parser.error(
            "host scenario requires known GPU capacity and pressure exceeding it"
        )
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(args.provenance_json.read_text())
    if provenance.get("model_revision") != args.model_revision:
        parser.error(
            "provenance model_revision must match the requested immutable revision"
        )
    owner_provenance = None
    if args.owner_base_url:
        owner_provenance = json.loads(args.owner_provenance_json.read_text())
        if owner_provenance.get("model_revision") != args.model_revision:
            parser.error("owner provenance model revision differs from requested immutable revision")
        if any(item.get("kv_bytes_per_page") != args.kv_bytes_per_page
               for item in (provenance, owner_provenance)):
            parser.error("both provenance files must confirm the configured logical KV bytes/page")
    client = Client(args.base_url, args.request_timeout)
    owner_client = Client(args.owner_base_url, args.request_timeout) if args.owner_base_url else None
    experiment, passed, error = None, False, None
    runtime = None
    archives = None
    try:
        if args.gcs_prefix:
            archives = GcsArchives(args.gcs_prefix, args.artifact_dir)
        info = client.request("/server_info")
        (args.artifact_dir / "server-info.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        server_args = info.get("server_args", info)
        mode = server_args.get("incremental_streaming_output")
        if args.stream_output == "auto" and not isinstance(mode, bool):
            raise ValueError(
                "server did not report streaming mode; pass --stream-output explicitly"
            )
        incremental = (
            mode
            if args.stream_output == "auto"
            else args.stream_output == "incremental"
        )
        if server_args.get("stream_interval") != 1:
            raise ValueError(
                "true first-token timing requires server stream_interval=1"
            )
        if server_args.get("page_size") != args.page_size:
            raise ValueError(
                "client page size differs from the serving cache page size"
            )
        capacity = server_args.get("max_total_tokens")
        if capacity is not None and max(args.contexts) + args.output_tokens > capacity:
            raise ValueError(
                "contexts plus generated tokens exceed server GPU token capacity"
            )
        if not server_args.get("enable_metrics"):
            raise ValueError("cache preparation requires an --enable-metrics server")
        if server_args.get("hicache_storage_backend") != "nixlshard":
            raise ValueError("server does not report the NIXLShard storage backend")
        owner_incremental = None
        owner_initial_metrics = None
        if owner_client is not None:
            owner_info = owner_client.request("/server_info")
            (args.artifact_dir / "owner-server-info.json").write_text(json.dumps(owner_info, indent=2) + "\n")
            owner_args = owner_info.get("server_args", owner_info)
            for key in ("model_path", "dtype", "kv_cache_dtype", "tp_size", "hicache_mem_layout"):
                if owner_args.get(key) != server_args.get(key):
                    raise ValueError(f"owner/requester serving namespace setting differs: {key}")
            if (owner_args.get("stream_interval") != 1 or owner_args.get("page_size") != args.page_size
                    or not owner_args.get("enable_metrics")
                    or owner_args.get("hicache_storage_backend") != "nixlshard"):
                raise ValueError("owner must expose compatible page/stream/cache/metrics settings")
            owner_mode = owner_args.get("incremental_streaming_output")
            if args.stream_output == "auto" and not isinstance(owner_mode, bool):
                raise ValueError("owner did not report streaming mode")
            owner_incremental = owner_mode if args.stream_output == "auto" else args.stream_output == "incremental"
            owner_capacity = owner_args.get("max_total_tokens")
            if owner_capacity is not None and max(args.contexts) + args.output_tokens > owner_capacity:
                raise ValueError("contexts plus output tokens exceed owner GPU token capacity")
            owner_initial_metrics = owner_client.request("/metrics")
            (args.artifact_dir / "owner-initial-metrics.txt").write_text(owner_initial_metrics)
        if (
            "host" in args.scenarios
            and server_args.get("max_total_tokens") != args.gpu_capacity_tokens
        ):
            raise ValueError(
                "host preparation GPU capacity must match server max_total_tokens"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, local_files_only=True
        )
        initial_metrics = client.request("/metrics")
        if owner_client is not None:
            for role, metrics in (("requester", initial_metrics), ("owner", owner_initial_metrics)):
                if not native_component_availability(metrics_snapshot(metrics))["prometheus_all_fixed_families_observed"]:
                    raise ValueError(f"{role} requires native metrics export for remote payload proof")
        (args.artifact_dir / "initial-metrics.txt").write_text(initial_metrics)
        runtime = {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "python": sys.version,
            "client": "stdlib HTTP/1.1 persistent connection",
            "model_revision": args.model_revision,
            "server_provenance": provenance,
            "owner_provenance": owner_provenance,
            "owner_incremental_streaming_output": owner_incremental,
            "incremental_streaming_output": incremental,
            "measurement": "request submission to first received nonempty output-token SSE event",
            "one_generation_outstanding": True,
            "incremental_gcs_archives": args.gcs_prefix,
            "native_component_counters": native_component_availability(
                metrics_snapshot(initial_metrics), args.native_stats_url
            ),
            "limitations": [
                "cache-level TTFT differences do not isolate NIXLShard overhead",
                "backup counter settling is not proof of zero background I/O",
                "host pressure requests are preparation and excluded from TTFT",
                "HiCache direct GPU/host path is independent of SSD O_DIRECT",
                "remote profiles explicitly prepare async MD/peer discovery before measured replay",
                "remote byte-proof windows include full generation and metrics export settling, not only first-token latency",
            ],
        }
        (args.artifact_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        if archives is not None:
            archives.archive(args.artifact_dir, "initial.zip", recursive=False)
        if owner_client is not None:
            experiment = RemoteExperiment(args, client, tokenizer, args.artifact_dir, incremental, archives,
                                          owner_client=owner_client, owner_incremental=owner_incremental)
        else:
            experiment = Experiment(args, client, tokenizer, args.artifact_dir, incremental, archives)
        for context in args.contexts:
            for repeat in range(args.warmups + args.repeats):
                warmup = repeat < args.warmups
                # Independent unique prefixes keep previous repeats out of the cold baseline.
                for scenario in args.scenarios:
                    if scenario == "gpu" and "cold" in args.scenarios:
                        continue  # Already measured as the immediate cold replay.
                    ids, result = experiment.sample(scenario, context, repeat, warmup)
                    if scenario == "cold" and "gpu" in args.scenarios:
                        # GPU hits must immediately replay the same cold request.
                        experiment.sample("gpu", context, repeat, warmup, ids, result)
                # 'gpu' is handled above when cold is selected, otherwise independently seeded.
        passed = all(sample["passed"] for sample in experiment.samples)
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        client.close()
        if owner_client is not None:
            owner_client.close()
        if runtime is not None and experiment is not None:
            observed = (
                set(runtime["native_component_counters"]["prometheus_counter_families"])
                | experiment.native_prometheus_families
            )
            runtime["native_component_counters"] = native_component_availability(
                observed, args.native_stats_url, experiment.native_diagnostic_received
            )
            (args.artifact_dir / "runtime.json").write_text(
                json.dumps(runtime, indent=2) + "\n"
            )
        summary = {
            "passed": passed,
            "error": error,
            "groups": experiment.summary() if experiment else {},
            "failed_samples": (
                sum(not sample["passed"] for sample in experiment.samples)
                if experiment
                else 0
            ),
            "percentile_method": "linear interpolation on ordered samples",
            "ttft_scope": "client streaming latency; includes network and server work",
        }
        (args.artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        if archives is not None:
            try:
                archives.archive(args.artifact_dir, "summary.zip", recursive=False)
            except Exception as caught:
                summary.update(passed=False, archive_error=repr(caught))
                (args.artifact_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                if error is None:
                    raise
                print("Final GCS archive also failed: " + repr(caught), file=sys.stderr)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
