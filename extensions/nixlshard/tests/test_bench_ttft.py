"""Streaming timing and cache-provenance tests, independent of GPU packages."""

import importlib.util
import io
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import tempfile
import sys
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
                return f'sglang:backuped_tokens_total{{rank="0"}} {self.backed_up}\n'

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
