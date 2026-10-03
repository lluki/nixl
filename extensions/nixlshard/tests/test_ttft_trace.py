"""Validate real-clock joins and reject fabricated additive trace breakdowns."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

TOOLS = Path(__file__).parents[1] / "tools"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


trace = module("ttft_trace", "ttft_trace.py")
bench = module("bench_ttft_trace", "bench-ttft.py")


def fixture():
    result = dict(
        request_id="r1",
        clock="CLOCK_MONOTONIC",
        request_start_ns=100,
        first_token_ns=1000,
    )

    def event(stage, a, b=None, **extra):
        return dict(
            rid="r1",
            clock="CLOCK_MONOTONIC",
            stage=stage,
            start_ns=a,
            end_ns=a if b is None else b,
            **extra,
        )

    native = [
        dict(
            stage="queue",
            start_ns=220,
            end_ns=250,
            bytes=0,
            first_object=0,
            object_count=1,
            request_id="",
        ),
        dict(
            stage="remote_rpc",
            start_ns=250,
            end_ns=500,
            bytes=64,
            first_object=0,
            object_count=1,
            request_id="wire-ticket",
            owner_posix_ns=100,
            owner_ucx_ns=80,
            owner_read_bytes=64,
        ),
        dict(
            stage="staging_copy",
            start_ns=500,
            end_ns=550,
            bytes=64,
            first_object=0,
            object_count=1,
            request_id="",
        ),
    ]
    records = [
        event("api_request_received", 150),
        event("scheduler_received", 175),
        event("metadata_query", 200, 350),
        event("first_forward_entry", 600),
        event("first_prefill_result", 850),
        event("api_first_nonempty_output", 900),
        event(
            "native_batch", 560, batch_handle=7, terminal_observed=True, events=native
        ),
    ]
    return result, records


class TraceTests(unittest.TestCase):
    def test_tier_masks_require_consistent_totals_and_exact_storage_prefix(self):
        result = {"meta_info": {"cached_tokens": 64, "cached_tokens_details": {
            "storage": 64, "storage_backend": "HiCacheNixlShard"}}}
        self.assertEqual(bench.validate_cache("ssd", result, 64)["storage"], 64)
        result["meta_info"]["cached_tokens"] = 128
        with self.assertRaisesRegex(AssertionError, "totals differ"):
            bench.validate_cache("ssd", result, 64)
        result["meta_info"]["cached_tokens_details"]["storage"] = 128
        with self.assertRaisesRegex(AssertionError, "prefix differs"):
            bench.validate_cache("ssd", result, 64)

    def test_rpc_composition_survives_overlapping_metadata_queue_boundaries(self):
        result, records = fixture()
        sample = trace.normalize(result, records, "remote", 128, 0, False)
        blocks = sample["critical_blocks"]
        rpc = [b for b in blocks if b["category"] == "remote_rpc"]
        self.assertEqual(len(rpc), 1)
        self.assertEqual((rpc[0]["start_ns"], rpc[0]["end_ns"]), (250, 500))
        self.assertTrue(rpc[0]["composition_only"])
        self.assertEqual(sum(b["end_ns"] - b["start_ns"] for b in blocks), 900)
        self.assertEqual(rpc[0]["owner_posix_ns"] + rpc[0]["owner_ucx_ns"], 180)
        self.assertNotIn("owner_start_ns", rpc[0])
        self.assertEqual([(b["start_ns"], b["end_ns"]) for b in blocks
                          if b["category"] == "frontend_dispatch"], [(150, 175)])
        self.assertEqual([(b["start_ns"], b["end_ns"]) for b in blocks
                          if b["category"] == "framework_wait"], [(560, 600)])
        self.assertEqual([(b["start_ns"], b["end_ns"]) for b in blocks
                          if b["category"] == "output_handoff"], [(850, 900)])
        self.assertEqual(sample["service_windows"][0]["evidence"], ["native:7:1"])
        self.assertEqual(
            sample["service_windows"][0]["pattern_id"], "descriptor_count_1"
        )

    def test_wrong_clock_anchors_and_missing_or_excess_owner_timing_fail(self):
        for mutation in (
            lambda r, e: r.update(clock="CLOCK_REALTIME"),
            lambda r, e: e.append(copy.deepcopy(e[0])),
            lambda r, e: e[-1]["events"][1].pop("owner_posix_ns"),
            lambda r, e: e[-1]["events"][1].update(owner_ucx_ns=200),
            lambda r, e: e[-1].update(terminal_observed=False),
        ):
            result, records = fixture()
            mutation(result, records)
            with self.assertRaises(ValueError):
                trace.normalize(result, records, "remote", 128, 0, False)

    def test_forward_overlap_remains_combined_and_owner_composition_is_not_fabricated(
        self,
    ):
        result, records = fixture()
        records[-1]["events"][1]["end_ns"] = 700
        records[-1]["events"][2].update(start_ns=700, end_ns=750)
        sample = trace.normalize(result, records, "remote", 128, 0, False)
        self.assertTrue(sample["unresolved_overlaps"])
        self.assertFalse(
            any(b.get("composition_only") for b in sample["critical_blocks"])
        )
        forward = [
            b
            for b in sample["critical_blocks"]
            if b["category"] == "framework_h2d_forward"
        ]
        self.assertEqual([(b["start_ns"], b["end_ns"]) for b in forward], [(600, 850)])
        self.assertEqual(
            sum(b["end_ns"] - b["start_ns"] for b in sample["critical_blocks"]), 900
        )

    def test_capture_remote_requires_exact_covered_bytes_and_retains_failed_sample(
        self,
    ):
        result, records = fixture()
        args = SimpleNamespace(page_size=64, kv_bytes_per_page=64)
        result.update(output_ids=[7], meta_info={"cached_tokens":64,
                      "cached_tokens_details":{"storage":64,"storage_backend":"HiCacheNixlShard"}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args.request_trace_dir = root
            (root / "request-timeline-1.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            experiment = bench.Experiment(args, None, None, root, False)
            record = {}
            experiment.capture_trace(root, result, record, "remote", 128, 0, False)
            self.assertTrue(record["critical_path"]["verified"])
            records[-1]["events"][2]["bytes"] = 63
            (root / "request-timeline-1.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            with self.assertRaisesRegex(AssertionError, "exact successful remote"):
                experiment.capture_trace(root, result, {}, "remote", 128, 0, False)

    def test_local_and_hot_tier_native_proofs_distinguish_read_from_background_write(
        self,
    ):
        def series(**values):
            return {
                f'sglang:nixlshard_component_bytes_total{{component="{k}"}}': v
                for k, v in values.items()
            }

        args = SimpleNamespace(page_size=64, kv_bytes_per_page=64, backup_timeout=0.1)
        baseline = dict(
            metrics=series(
                posix_read=0, staging_copy=0, remote_read=0, ucx_write=0, posix_write=0
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            experiment = bench.Experiment(args, None, None, Path(directory), False)
            experiment.snapshot = lambda *a: dict(
                metrics=series(
                    posix_read=64,
                    staging_copy=64,
                    remote_read=0,
                    ucx_write=0,
                    posix_write=128,
                )
            )
            self.assertTrue(
                experiment.local_payload_proof(Path(directory), "ssd", baseline, 64)[
                    "exact"
                ]
            )
            with self.assertRaises(AssertionError):
                experiment.local_payload_proof(Path(directory), "host", baseline, 64)
            experiment.snapshot = lambda *a: dict(
                metrics=series(
                    posix_read=0,
                    staging_copy=0,
                    remote_read=0,
                    ucx_write=0,
                    posix_write=128,
                )
            )
            self.assertTrue(
                experiment.local_payload_proof(Path(directory), "host", baseline, 64)[
                    "exact"
                ]
            )


if __name__ == "__main__":
    unittest.main()
