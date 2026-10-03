#!/usr/bin/env python3
"""Measure client streaming TTFT with verified cold/SSD/host/GPU cache sources.

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
        self.connection.request(
            "GET" if body is None else "POST",
            path,
            body,
            {"Content-Type": "application/json"},
        )
        response = self.connection.getresponse()
        raw = response.read().decode()
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
        desired = {"ssd": "storage", "host": "host", "gpu": "device"}[scenario]
        if values[desired] < expected_prefix or any(
            values[name] for name in ("device", "host", "storage") if name != desired
        ):
            raise AssertionError(
                f"{scenario} request was not an isolated >= {expected_prefix}-token hit: {values}"
            )
        if scenario == "ssd" and values["storage_backend"] != "HiCacheNixlShard":
            raise AssertionError(f"SSD hit did not identify HiCacheNixlShard: {values}")
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
                "nixl",
            )
        ):
            continue
        match = re.match(r"^(.+?)\s+([-+0-9.eE]+)(?:\s+\S+)?$", line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                result[match[1]] = value
    return result


def metric_total(snapshot, name):
    return sum(value for key, value in snapshot.items() if key.split("{", 1)[0] == name)


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

    def snapshot(self, directory, label):
        metrics = self.client.request("/metrics")
        (directory / f"{label}-metrics.txt").write_text(metrics)
        result = {"metrics": metrics_snapshot(metrics)}
        if self.args.native_stats_url:
            result["native_stats"] = self.client.request(self.args.native_stats_url)
        (directory / f"{label}-stats.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result

    def flush(self):
        result = self.client.request("/flush_cache?timeout=10", {})
        if "Cache flushed" not in str(result):
            raise RuntimeError(f"upper-tier flush failed: {result}")
        return result

    def await_backup(self, before, expected):
        baseline = metric_total(before["metrics"], "sglang:backuped_tokens_total")
        deadline = time.monotonic() + self.args.backup_timeout
        last_total, stable_since = None, None
        while time.monotonic() < deadline:
            snapshot = metrics_snapshot(self.client.request("/metrics"))
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

    def generate_record(self, directory, role, ids):
        result = self.client.generate(self.payload(ids), self.incremental)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31001")
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
        choices=["cold", "ssd", "host", "gpu"],
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
    client = Client(args.base_url, args.request_timeout)
    experiment, passed, error = None, False, None
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
        runtime = {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "python": sys.version,
            "client": "stdlib HTTP/1.1 persistent connection",
            "model_revision": args.model_revision,
            "server_provenance": provenance,
            "incremental_streaming_output": incremental,
            "measurement": "request submission to first received nonempty output-token SSE event",
            "one_generation_outstanding": True,
            "incremental_gcs_archives": args.gcs_prefix,
            "native_component_counters": (
                "optional diagnostic endpoint snapshots"
                if args.native_stats_url
                else "unavailable: generic SGLang metrics do not export adapter native_* counters"
            ),
            "limitations": [
                "cache-level TTFT differences do not isolate NIXLShard overhead",
                "backup counter settling is not proof of zero background I/O",
                "host pressure requests are preparation and excluded from TTFT",
                "HiCache direct GPU/host path is independent of SSD O_DIRECT",
            ],
        }
        (args.artifact_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        if archives is not None:
            archives.archive(args.artifact_dir, "initial.zip", recursive=False)
        experiment = Experiment(
            args, client, tokenizer, args.artifact_dir, incremental, archives
        )
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
