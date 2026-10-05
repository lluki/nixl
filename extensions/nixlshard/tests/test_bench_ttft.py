"""Streaming timing and cache-provenance tests, independent of GPU packages."""

import importlib.util
import io
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import tempfile
import sys
import socket
import threading
import time
import unittest
import zipfile
from unittest import mock

path = Path(__file__).parents[1] / "tools" / "bench-ttft.py"
spec = importlib.util.spec_from_file_location("bench_ttft", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def event(ids, **meta):
    return {"output_ids": ids, "text": "x" * len(ids), "meta_info": meta}


def encoded_frames(events):
    lines = []
    for index, value in enumerate(events, 1):
        data = value if isinstance(value, str) else json.dumps(value)
        lines.extend(
            [(f"data: {data}\r\n".encode(), index * 1000), (b"\r\n", index * 1000)]
        )
    return iter(lines)


class Tests(unittest.TestCase):
    def test_admin_authentication_and_error_echo_do_not_record_token(self):
        key = "test-private-control-token"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                authorized = self.headers.get("Authorization") == "Bearer " + key
                body = json.dumps({"admin_api_key": key, "page_size": 64}).encode()
                self.send_response(200 if authorized and self.path == "/server_info" else 403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = bench.Client(f"http://127.0.0.1:{server.server_port}", auth_token=key)
        try:
            observation = bench.redact_credentials(client.request("/server_info"))
            self.assertEqual(observation["page_size"], 64)
            self.assertNotIn(key, json.dumps(observation))
            with self.assertRaises(RuntimeError) as error:
                client.request("/failure")
            self.assertNotIn(key, str(error.exception))
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(1)

    def test_serving_credentials_are_removed_before_artifacts_and_archives(self):
        key = "test-private-admin-value"
        source = {"admin_api_key": key, "api_key": None, "page_size": 64,
                  "launch_command": ["python", "--admin-api-key", key],
                  "nested": {"echo": "arguments: " + key},
                  "list_only": ["--api-key", "other-private-token"]}
        safe = bench.redact_credentials(source)
        serialized = json.dumps(safe)
        self.assertNotIn(key, serialized)
        self.assertNotIn("other-private-token", serialized)
        self.assertEqual(safe["page_size"], 64)
        self.assertIsNone(safe["api_key"])
        self.assertEqual(source["admin_api_key"], key)
        self.assertEqual(safe["launch_command"][-1], "<redacted>")
        self.assertEqual(safe["nested"]["echo"], "arguments: <redacted>")

    def test_idle_http_keepalive_reconnects_observations_without_retrying_generation(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append(self.path)
                body = b"metrics"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if len(requests) == 1:
                    def idle_close():
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        self.connection.close()
                    threading.Timer(0.02, idle_close).start()

            def do_POST(self):
                requests.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = bench.Client(f"http://127.0.0.1:{server.server_port}", timeout=1)
        try:
            self.assertEqual(client.request("/metrics"), "metrics")
            time.sleep(0.04)
            self.assertEqual(client.request("/metrics"), "metrics")
            with self.assertRaises(bench.http.client.RemoteDisconnected):
                client.generate({"input_ids": [1]}, False)
            self.assertEqual(requests, ["/metrics", "/metrics", "/generate"])
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(1)

    def test_incremental_archives_preserve_complete_samples_and_fail_on_upload_error(
        self,
    ):
        uploads = []

        def blob(name):
            return SimpleNamespace(
                name=name,
                upload_from_string=lambda body, **kwargs: uploads.append(
                    (name, body, kwargs)
                ),
            )

        bucket = SimpleNamespace(name="test-bucket", blob=blob)
        storage = SimpleNamespace(
            Client=lambda: SimpleNamespace(bucket=lambda name: bucket)
        )
        modules = {
            "google": SimpleNamespace(cloud=SimpleNamespace(storage=storage)),
            "google.cloud": SimpleNamespace(storage=storage),
            "google.cloud.storage": storage,
        }
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(sys.modules, modules),
        ):
            artifact = Path(root) / "run"
            sample = artifact / "0001-512-ssd"
            sample.mkdir(parents=True)
            (sample / "sample.json").write_text('{"passed": true}')
            (sample / "measured.json").write_text('{"output_ids": [1, 2]}')
            (sample / "before-metrics.txt").write_text("storage_hits 448\n")
            archives = bench.GcsArchives("gs://test-bucket/runs/unique", artifact)
            archives.archive(sample, sample.name + ".zip")
            name, body, options = uploads[0]
            self.assertEqual(name, "runs/unique/run/0001-512-ssd.zip")
            self.assertEqual(options["if_generation_match"], 0)
            with zipfile.ZipFile(io.BytesIO(body)) as saved:
                self.assertEqual(
                    set(saved.namelist()),
                    {"sample.json", "measured.json", "before-metrics.txt"},
                )
                self.assertEqual(
                    json.loads(saved.read("measured.json"))["output_ids"], [1, 2]
                )
            self.assertEqual(
                json.loads((artifact / "gcs-archives.json").read_text())[0]["bytes"],
                len(body),
            )

            def fail(*_, **__):
                raise OSError("upload failed")

            bucket.blob = lambda name: SimpleNamespace(
                name=name, upload_from_string=fail
            )
            with self.assertRaisesRegex(OSError, "upload failed"):
                archives.archive(sample, "failed.zip")
            self.assertEqual(len(archives.uploads), 1)

    def test_first_nonempty_token_ignores_metadata_and_final_completion(self):
        response = bench.collect_stream(
            encoded_frames(
                [
                    event([], prompt_tokens=128),
                    event([0], prompt_tokens=128),
                    event([0, 7], prompt_tokens=128, finish_reason={"type": "length"}),
                    "[DONE]",
                ]
            ),
            0,
        )
        self.assertEqual(response["first_token_elapsed_ns"], 2000)
        self.assertEqual(response["output_ids"], [0, 7])
        self.assertEqual(response["stream_latency_ms"], 0.004)

    def test_cumulative_and_incremental_repeated_ids_remain_distinct(self):
        cumulative = bench.collect_stream(
            encoded_frames(
                [event([7]), event([7, 7], finish_reason={"type": "length"}), "[DONE]"]
            ),
            0,
        )
        incremental = bench.collect_stream(
            encoded_frames(
                [event([7]), event([7], finish_reason={"type": "length"}), "[DONE]"]
            ),
            0,
            True,
        )
        self.assertEqual(cumulative["output_ids"], [7, 7])
        self.assertEqual(incremental["output_ids"], cumulative["output_ids"])
        self.assertEqual(incremental["text"], cumulative["text"])

    def test_truncated_error_and_nonterminal_streams_fail(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            bench.collect_stream(iter([(b'data: {"output_ids": [1]}\n', 1000)]), 0)
        with self.assertRaisesRegex(RuntimeError, "error"):
            bench.collect_stream(
                encoded_frames([{"error": "backend failed"}, "[DONE]"]), 0
            )
        with self.assertRaisesRegex(ValueError, "finish reason"):
            bench.collect_stream(encoded_frames([event([1]), "[DONE]"]), 0)

    def test_sse_comments_and_multiline_data(self):
        lines = iter(
            [
                (b": heartbeat\n", 1000),
                (b"\n", 1000),
                (b'data: {"output_ids": [7],\n', 2000),
                (b'data: "meta_info": {"finish_reason": {"type": "length"}}}\n', 2000),
                (b"\n", 2000),
                (b"data: [DONE]\n", 3000),
                (b"\n", 3000),
            ]
        )
        self.assertEqual(bench.collect_stream(lines, 0)["first_token_elapsed_ns"], 2000)

    def test_cache_sources_reject_mixed_or_incorrectly_named_hits(self):
        result = {
            "meta_info": {
                "cached_tokens": 448,
                "cached_tokens_details": {
                    "storage": 448,
                    "storage_backend": "HiCacheNixlShard",
                },
            }
        }
        self.assertEqual(bench.validate_cache("ssd", result, 448)["storage"], 448)
        with self.assertRaises(AssertionError):
            bench.validate_cache("gpu", result, 448)
        result["meta_info"]["cached_tokens_details"]["device"] = 1
        with self.assertRaises(AssertionError):
            bench.validate_cache("ssd", result, 448)
        result["meta_info"]["cached_tokens_details"] = {"host": 448}
        self.assertEqual(bench.validate_cache("host", result, 448)["host"], 448)
        with self.assertRaises(AssertionError):
            bench.validate_cache("cold", result, 448)

    def test_failed_cache_sample_keeps_raw_response_and_excludes_summary(self):
        class FakeClient:
            def request(self, path, payload=None):
                return "Cache flushed" if path.startswith("/flush_cache") else ""

            def generate(self, payload, incremental):
                return {
                    "ttft_ms": 1,
                    "stream_latency_ms": 2,
                    "output_ids": [1],
                    "text": "x",
                    "meta_info": {
                        "prompt_tokens": 128,
                        "cached_tokens": 1,
                        "finish_reason": {"type": "length"},
                    },
                    "frames": [],
                }

        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text.split())))

        args = SimpleNamespace(native_stats_url=None, output_tokens=1, page_size=64)
        with tempfile.TemporaryDirectory() as directory:
            experiment = bench.Experiment(
                args, FakeClient(), Tokenizer(), Path(directory), False
            )
            with self.assertRaises(AssertionError):
                experiment.sample("cold", 128, 0, False)
            self.assertFalse(experiment.samples[0]["passed"])
            self.assertEqual(experiment.summary(), {})
            self.assertTrue(
                (Path(directory) / "0001-128-cold" / "measured.json").is_file()
            )
            self.assertIn(
                "cold request had cache hits",
                (Path(directory) / "samples.jsonl").read_text(),
            )

    def test_live_http_receives_first_token_before_delayed_final_event(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(value):
                    payload = (
                        "data: "
                        + (value if isinstance(value, str) else json.dumps(value))
                        + "\n\n"
                    ).encode()
                    self.wfile.write(
                        f"{len(payload):x}\r\n".encode() + payload + b"\r\n"
                    )
                    self.wfile.flush()

                time.sleep(0.02)
                send(event([], prompt_tokens=128))
                time.sleep(0.04)
                send(event([7], prompt_tokens=128))
                time.sleep(0.08)
                send(event([7, 8], prompt_tokens=128, finish_reason={"type": "length"}))
                send("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = bench.Client(f"http://127.0.0.1:{server.server_port}", timeout=2)
        try:
            result = client.generate(
                {"input_ids": [1], "sampling_params": {"max_new_tokens": 2}}
            )
            self.assertEqual(result["output_ids"], [7, 8])
            self.assertGreater(result["ttft_ms"], 40)
            self.assertGreater(result["stream_latency_ms"] - result["ttft_ms"], 60)
        finally:
            client.close()
            server.shutdown()
            worker.join()
            server.server_close()

    def test_cli_pairs_one_gpu_sample_and_preserves_ssd_seed_evidence(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())

        class FakeClient:
            def __init__(self, *args):
                self.device, self.storage = set(), set()
                self.backed_up = 0

            def request(self, path, payload=None):
                if path == "/server_info":
                    return {
                        "stream_interval": 1,
                        "incremental_streaming_output": False,
                        "page_size": 64,
                        "max_total_tokens": 4096,
                        "enable_metrics": True,
                        "hicache_storage_backend": "nixlshard",
                    }
                if path.startswith("/flush_cache"):
                    self.device.clear()
                    return "Cache flushed"
                native = (
                    'sglang:nixlshard_component_seconds_total{component="posix_read"} 1\n'
                    'sglang:nixlshard_component_bytes_total{component="posix_read"} 8192\n'
                    'sglang:nixlshard_events_total{event="success"} 1\n'
                    if self.backed_up
                    else ""
                )
                return (
                    f'sglang:backuped_tokens_total{{rank="0"}} {self.backed_up}\n'
                    + native
                )

            def generate(self, payload, incremental):
                key = tuple(payload["input_ids"])
                desired = (
                    "device"
                    if key in self.device
                    else "storage" if key in self.storage else None
                )
                details = {desired: 64} if desired else {}
                if desired == "storage":
                    details["storage_backend"] = "HiCacheNixlShard"
                self.device.add(key)
                self.storage.add(key)
                self.backed_up += 128
                return {
                    "ttft_ms": 1,
                    "stream_latency_ms": 2,
                    "output_ids": [7],
                    "text": "x",
                    "meta_info": {
                        "prompt_tokens": 128,
                        "cached_tokens": 64 if desired else 0,
                        "cached_tokens_details": details,
                        "finish_reason": {"type": "length"},
                    },
                    "frames": [],
                }

            def close(self):
                pass

        auto_tokenizer = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()
        )
        module = SimpleNamespace(AutoTokenizer=auto_tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            provenance = Path(temporary) / "provenance.json"
            provenance.write_text('{"model_revision": "frozen-test"}')
            artifact = Path(temporary) / "run"
            argv = [
                "bench-ttft",
                "--tokenizer-path",
                temporary,
                "--model-revision",
                "frozen-test",
                "--provenance-json",
                str(provenance),
                "--artifact-dir",
                str(artifact),
                "--contexts",
                "128",
                "--output-tokens",
                "1",
                "--warmups",
                "0",
                "--repeats",
                "1",
                "--settle-seconds",
                "0",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(bench, "Client", FakeClient),
                mock.patch.dict(sys.modules, {"transformers": module}),
                mock.patch("builtins.print"),
            ):
                bench.main()
            summary = json.loads((artifact / "summary.json").read_text())
            self.assertTrue(summary["passed"])
            runtime = json.loads((artifact / "runtime.json").read_text())
            components = runtime["native_component_counters"]
            self.assertTrue(components["prometheus_counters_available"])
            self.assertTrue(components["prometheus_all_fixed_families_observed"])
            self.assertIsNone(components["agent_diagnostic_url"])
            self.assertFalse(components["diagnostic_snapshot_received"])
            self.assertEqual(
                {key: value["count"] for key, value in summary["groups"].items()},
                {"128:cold": 1, "128:gpu": 1, "128:ssd": 1},
            )
            records = [
                json.loads(line)
                for line in (artifact / "samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["scenario"] for record in records], ["cold", "gpu", "ssd"]
            )
            self.assertTrue(
                all(record["output_matches_cold"] for record in records[1:])
            )
            self.assertTrue(
                (artifact / records[-1]["artifact"] / "seed.json").is_file()
            )
            self.assertTrue(
                (
                    artifact / records[-1]["artifact"] / "before-seed-stats.json"
                ).is_file()
            )

    def test_native_component_metric_families_are_captured_without_unrelated_nixl_fields(
        self,
    ):
        seconds = 'sglang:nixlshard_component_seconds_total{component="posix_read",model_name="test model",tp_rank="0"}'
        byte_counts = (
            'sglang:nixlshard_component_bytes_total{component="posix_read",tp_rank="0"}'
        )
        events = 'sglang:nixlshard_events_total{event="timeout",tp_rank="0"}'
        snapshot = bench.metrics_snapshot(
            f"{seconds} 1.75\n{byte_counts} 8192\n{events} 3\n"
            "nixl_unrelated_diagnostic 99\n"
            "# HELP sglang:nixlshard_component_seconds_total observed wall seconds\n"
        )
        self.assertEqual(snapshot, {seconds: 1.75, byte_counts: 8192, events: 3})

    def test_prometheus_native_availability_is_distinct_from_full_diagnostic_url(self):
        unavailable = bench.native_component_availability(
            {
                "sglang:backuped_tokens_total": 64,
                "sglang:nixlshard_events_created": 123,
            },
            "/native-diagnostic",
        )
        self.assertFalse(unavailable["prometheus_counters_available"])
        self.assertEqual(unavailable["agent_diagnostic_url"], "/native-diagnostic")
        self.assertFalse(unavailable["diagnostic_snapshot_received"])
        partial = bench.native_component_availability(
            {'sglang:nixlshard_events_total{event="timeout"}': 0}
        )
        self.assertTrue(partial["prometheus_counters_available"])
        self.assertFalse(partial["prometheus_all_fixed_families_observed"])
        complete = bench.native_component_availability(
            bench.NATIVE_COUNTER_FAMILIES, "/native-diagnostic", received=True
        )
        self.assertTrue(complete["prometheus_all_fixed_families_observed"])
        self.assertTrue(complete["diagnostic_snapshot_received"])

    @staticmethod
    def remote_fixture(directory, corrupt_measured=False):
        calls = []
        components = ("staging_copy", "posix_read", "posix_write", "remote_read", "ucx_write")

        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())

        class Node:
            def __init__(self, role):
                self.role, self.generations, self.flushes = role, 0, 0
                self.actual = {name: 0 for name in components}
                self.exported = dict(self.actual)
                self.export_delay = 0
                self.backup = 0
                self.seeded = False

            def request(self, path, payload=None):
                if path.startswith("/flush_cache"):
                    self.flushes += 1
                    return "Cache flushed"
                if self.export_delay:
                    self.export_delay -= 1
                else:
                    self.exported = dict(self.actual)
                return (f"sglang:backuped_tokens_total {self.backup}\n" +
                        "".join(f'sglang:nixlshard_component_bytes_total{{component="{key}"}} {value}\n'
                                for key, value in self.exported.items()) +
                        'sglang:nixlshard_component_seconds_total{component="posix_read"} 0.1\n' +
                        'sglang:nixlshard_events_total{event="success"} 0\n')

            def generate(self, payload, incremental):
                self.generations += 1
                calls.append(self.role)
                hit = self.role == "requester" and owner.seeded and self.generations >= 3
                if self.role == "owner":
                    self.seeded = True
                    self.backup += 128
                    self.actual["posix_write"] += 8192
                    self.actual["staging_copy"] += 8192
                elif hit:
                    amount = 8192 if corrupt_measured and self.generations == 4 else 4096
                    self.actual["remote_read"] += amount
                    self.actual["staging_copy"] += amount
                    owner.actual["posix_read"] += amount
                    owner.actual["ucx_write"] += amount
                    self.export_delay = owner.export_delay = 1
                details = {"storage": 64, "storage_backend": "HiCacheNixlShard"} if hit else {}
                return {"ttft_ms": 2 if hit else 10, "stream_latency_ms": 20,
                        "output_ids": [7], "text": "x", "frames": [],
                        "meta_info": {"prompt_tokens": 128, "cached_tokens": 64 if hit else 0,
                                      "cached_tokens_details": details,
                                      "finish_reason": {"type": "length"}}}

        owner, requester = Node("owner"), Node("requester")
        args = SimpleNamespace(native_stats_url=None, output_tokens=1, page_size=64,
                               kv_bytes_per_page=4096, backup_timeout=1, settle_seconds=0,
                               remote_discovery_attempts=3, contexts=[128])
        experiment = bench.RemoteExperiment(args, requester, Tokenizer(), directory, False,
                                           owner_client=owner, owner_incremental=False)
        return experiment, owner, requester, calls

    def test_remote_discovery_preparation_and_export_lag_preserve_matched_controls(self):
        with tempfile.TemporaryDirectory() as root:
            experiment, owner, requester, calls = self.remote_fixture(Path(root))
            experiment.sample("remote", 128, 0, False)
            record = experiment.samples[0]
            self.assertTrue(record["passed"])
            self.assertEqual(calls, ["requester", "owner", "requester", "requester", "requester"])
            self.assertEqual(requester.flushes, 4)
            self.assertEqual(len(record["discovery_preflight"]), 2)
            self.assertIn("not_ready", record["discovery_preflight"][0])
            self.assertEqual(record["cold_control_ttft_ms"], 10)
            self.assertEqual(record["ttft_ms"], 2)
            self.assertTrue(record["remote_proof"]["exact"])
            self.assertEqual(record["remote_proof"]["observed"]["owner_ucx_write_bytes"], 4096)
            self.assertEqual(record["remote_proof"]["observed"]["requester_posix_read_bytes"], 0)
            self.assertTrue(record["output_matches_cold"])
            groups = experiment.summary()
            self.assertEqual(groups["128:remote"]["ttft_ms_samples"], [2])
            self.assertEqual(groups["128:cold-control"]["ttft_ms_samples"], [10])
            sample_dir = Path(root) / record["artifact"]
            for name in ("owner-before-stats.json", "owner-after-stats.json",
                         "before-stats.json", "after-stats.json", "cold-control.json",
                         "owner-seed.json", "remote-journal.jsonl"):
                self.assertTrue((sample_dir / name).exists(), name)
            observations = [json.loads(line) for line in
                            (sample_dir / "remote-journal.jsonl").read_text().splitlines()]
            measured = [item for item in observations
                        if item["phase"] == "remote_counter_observation" and item["role"] == "after"]
            self.assertEqual(len(measured), 2)
            self.assertFalse(measured[0]["proof"]["exact"])
            self.assertTrue(measured[-1]["proof"]["exact"])

    def test_remote_wrong_transfer_amount_fails_without_hiding_raw_result(self):
        with tempfile.TemporaryDirectory() as root:
            experiment, owner, requester, calls = self.remote_fixture(Path(root), True)
            with self.assertRaisesRegex(AssertionError, "incorrect/extra payload"):
                experiment.sample("remote", 128, 0, False)
            self.assertFalse(experiment.samples[0]["passed"])
            self.assertEqual(experiment.summary(), {})
            sample_dir = Path(root) / experiment.samples[0]["artifact"]
            self.assertTrue((sample_dir / "measured.json").exists())
            self.assertTrue((sample_dir / "owner-after-poll-001-stats.json").exists())
            self.assertTrue((sample_dir / "sample.json").exists())

    def test_remote_byte_proof_rejects_missing_or_local_only_evidence(self):
        def counters(**values):
            return {f'sglang:nixlshard_component_bytes_total{{component="{name}"}}': values.get(name, 0)
                    for name in ("staging_copy", "posix_read", "posix_write", "remote_read", "ucx_write")}
        with self.assertRaisesRegex(ValueError, "missing"):
            bench.remote_counter_proof({}, {}, 4096)
        requester = counters(staging_copy=4096, remote_read=4096, posix_read=4096)
        owner = counters(posix_read=4096, ucx_write=4096)
        with self.assertRaisesRegex(AssertionError, "incorrect/extra payload"):
            bench.remote_counter_proof(requester, owner, 4096)
        requester = counters(staging_copy=4096, remote_read=4096)
        self.assertTrue(bench.remote_counter_proof(requester, owner, 4096)["exact"])
        owner = counters(posix_read=4096, ucx_write=4096, posix_write=64)
        with self.assertRaisesRegex(AssertionError, "incorrect/extra payload"):
            bench.remote_counter_proof(requester, owner, 4096)

    def test_direct_remote_proof_requires_positive_scatter_evidence_and_zero_copy(self):
        def counters(**values):
            return {f'sglang:nixlshard_component_bytes_total{{component="{name}"}}': values.get(name, 0)
                    for name in ("staging_copy", "posix_read", "posix_write", "remote_read", "ucx_write", "direct_receive")}
        owner = counters(posix_read=4096, ucx_write=4096)
        requester = counters(remote_read=4096, direct_receive=4096)
        with self.assertRaisesRegex(ValueError, "event counter missing"):
            bench.remote_counter_proof(requester, owner, 4096, True, 2)
        requester['sglang:nixlshard_events_total{event="direct_receive_segments"}'] = 2
        proof = bench.remote_counter_proof(requester, owner, 4096, True, 2)
        self.assertTrue(proof["exact"])
        self.assertTrue(proof["direct_receive"])
        self.assertEqual(proof["expected_destination_segments"], 2)
        for field, value in (("staging_copy", 4096), ("direct_receive", 8192)):
            corrupted = dict(requester)
            corrupted[f'sglang:nixlshard_component_bytes_total{{component="{field}"}}'] = value
            with self.assertRaisesRegex(AssertionError, "incorrect/extra payload"):
                bench.remote_counter_proof(corrupted, owner, 4096, True, 2)
        # Missing direct bytes cannot be inferred from zero copy. Counter export
        # lag stays explicitly incomplete rather than creating a false proof.
        lagged = counters(remote_read=4096)
        lagged['sglang:nixlshard_events_total{event="direct_receive_segments"}'] = 2
        self.assertFalse(bench.remote_counter_proof(lagged, owner, 4096, True, 2)["exact"])

    def test_local_direct_proof_counts_explicit_fallbacks_without_zero_copy_claim(self):
        def counters(direct_bytes=128, segments=4, fallback=0, copy_bytes=0):
            values = dict(posix_read=128, staging_copy=copy_bytes, remote_read=0, ucx_write=0,
                          direct_receive=direct_bytes, direct_local_read=direct_bytes)
            result = {f'sglang:nixlshard_component_bytes_total{{component="{name}"}}': value
                      for name, value in values.items()}
            result.update({f'sglang:nixlshard_events_total{{event="{name}"}}': value
                           for name, value in dict(direct_receive_segments=segments,
                                                  local_direct_fallbacks=fallback).items()})
            return result
        direct = bench.local_counter_proof(counters(), 128, 64, True)
        self.assertTrue(direct["exact"])
        self.assertTrue(direct["zero_copy_verified"])
        mixed = bench.local_counter_proof(counters(64, 2, 1, 64), 128, 64, True)
        self.assertTrue(mixed["exact"])
        self.assertEqual((mixed["direct_pages"], mixed["fallback_pages"]), (1, 1))
        self.assertFalse(mixed["zero_copy_verified"])
        all_fallback = bench.local_counter_proof(counters(0, 0, 2, 128), 128, 64, True)
        self.assertTrue(all_fallback["exact"])
        self.assertFalse(all_fallback["zero_copy_verified"])
        # A staged runtime with no direct-mode fallback counters cannot silently
        # qualify; partial exports also remain incomplete until counts agree.
        self.assertFalse(bench.local_counter_proof(counters(0, 0, 0, 128), 128, 64, True)["exact"])
        self.assertFalse(bench.local_counter_proof(counters(128, 2), 128, 64, True)["exact"])
        # Full-generation write gathers are not receiver copies. The per-RID
        # load trace, tested separately, must still prove read copies are zero.
        background = bench.local_counter_proof(counters(copy_bytes=64), 128, 64, True)
        self.assertTrue(background["exact"])
        self.assertEqual(background["observed_bytes"]["staging_copy"], 64)

    def test_percentiles_and_available_counter_deltas(self):
        self.assertEqual(bench.percentile([30, 10, 20], 50), 20)
        self.assertEqual(bench.percentile([30, 10, 20], 95), 29)
        metrics = bench.metrics_snapshot(
            'sglang:backuped_tokens_total{rank="0"} 448.0\n# comment\n'
        )
        self.assertEqual(
            bench.metric_total(metrics, "sglang:backuped_tokens_total"), 448
        )


if __name__ == "__main__":
    unittest.main()
