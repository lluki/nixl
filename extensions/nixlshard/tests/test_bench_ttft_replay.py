"""Agent-written regressions for archived-ID replay validation and injection."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools" / "bench-ttft-replay.py"
spec = importlib.util.spec_from_file_location("ttft_replay_tests_module", PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.params = dict(temperature=0, max_new_tokens=16, ignore_eos=True)
        for repeat in range(2):
            directory = self.root / f"{repeat + 1:04d}-128-remote"
            directory.mkdir()
            request = dict(input_ids=[repeat + 1] + [42] * 127, sampling_params=self.params)
            for name in ("request.json", "cold-control-request.json", "owner-seed-request.json", "measured-request.json"):
                (directory / name).write_text(json.dumps(request))
            output = dict(output_ids=[5, 6], text="result")
            for name in ("cold-control.json", "owner-seed.json", "measured.json"):
                (directory / name).write_text(json.dumps(output))
            sample = dict(context_tokens=128, repeat=repeat, warmup=repeat == 0, scenario="remote",
                          passed=True, receiver_mode="staged", cache=dict(device=0, host=0, storage=64),
                          remote_proof=dict(exact=True))
            (directory / "sample.json").write_text(json.dumps(sample))

    def plan(self):
        return replay.validate_plan(self.root, [128], 1, 1, 16)

    def change(self, repeat, file, key, value):
        path = self.root / f"{repeat + 1:04d}-128-remote" / file
        data = json.loads(path.read_text()); data[key] = value
        path.write_text(json.dumps(data))

    def test_complete_plan_preserves_order_ids_params_and_hashes(self):
        plan = self.plan()
        self.assertEqual([entry["warmup"] for entry in plan], [True, False])
        self.assertEqual(plan[1]["input_ids"][0], 2)
        self.assertEqual(plan[1]["sampling_params"], self.params)
        self.assertEqual(plan[1]["input_ids_sha256"], replay.digest(plan[1]["input_ids"]))

    def test_late_candidate_mismatched_seed_rejected_before_injection(self):
        self.change(1, "owner-seed-request.json", "input_ids", [99] * 128)
        with self.assertRaisesRegex(ValueError, "request IDs or parameters differ"):
            self.plan()

    def test_wrong_sampling_parameters_and_incomplete_plan_rejected(self):
        self.change(1, "measured-request.json", "sampling_params", dict(self.params, temperature=0.5))
        with self.assertRaisesRegex(ValueError, "sampling parameters"):
            self.plan()
        (self.root / "0002-128-remote" / "sample.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.plan()

    def test_duplicate_candidate_and_non_integer_tokens_rejected(self):
        self.change(1, "sample.json", "repeat", 0)
        self.change(1, "sample.json", "warmup", True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.plan()
        self.change(1, "sample.json", "repeat", 1)
        self.change(1, "sample.json", "warmup", False)
        self.change(1, "measured-request.json", "input_ids", [True] * 128)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            self.plan()

    def test_output_mismatch_and_shared_prefix_reference_rejected(self):
        self.change(1, "cold-control.json", "output_ids", [7])
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            self.plan()
        self.change(1, "cold-control.json", "output_ids", [5, 6])
        self.change(1, "sample.json", "cache", dict(device=64, host=0, storage=0))
        with self.assertRaisesRegex(ValueError, "pure storage"):
            self.plan()

    def test_explicit_new_base_pin_rejects_changed_bytes_before_import(self):
        path = self.root / "new-base.py"
        path.write_text("approved_value = 7\n")
        expected = replay.file_sha(path)
        self.assertEqual(replay.load_base(path, expected).approved_value, 7)
        path.write_text("raise AssertionError('must never execute changed source')\n")
        with self.assertRaisesRegex(ValueError, "frozen"):
            replay.load_base(path, expected)
        with self.assertRaises(ValueError):
            replay.load_base(path, "not-a-sha256")

    def test_changed_base_bytes_rejected_before_import(self):
        path = self.root / "base.py"
        path.write_text("raise RuntimeError('must never run')")
        with self.assertRaisesRegex(ValueError, "frozen"):
            replay.load_base(path)

    def fake_base(self, result_ids=(5, 6)):
        root = self.root / "new-artifacts"; root.mkdir()
        calls = []
        class Base:
            def __init__(self):
                self.artifact_dir = root
                self.archives = None
                self.sequence = 0
            def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
                calls.append((scenario, context, repetition, warmup, list(ids)))
                self.sequence += 1
                directory = root / f"{self.sequence:04d}-{context}-remote"
                directory.mkdir()
                (directory / "measured.json").write_text(json.dumps(dict(output_ids=list(result_ids), text="result")))
                return ids, dict(passed=True)
        return types.SimpleNamespace(RemoteExperiment=Base), calls

    def test_existing_ids_hook_receives_exact_plan_and_wrong_order_never_submits(self):
        plan = self.plan()
        base, calls = self.fake_base()
        replay.install_replay(base, plan, dict(candidate_count=2))
        instance = base.RemoteExperiment()
        with self.assertRaisesRegex(ValueError, "order"):
            instance.sample("remote", 128, 1, False)
        self.assertFalse(calls)
        instance.sample("remote", 128, 0, True)
        instance.sample("remote", 128, 1, False)
        self.assertEqual([call[4] for call in calls], [entry["input_ids"] for entry in plan])
        self.assertEqual(json.loads((instance.artifact_dir / "replay-plan.json").read_text())["candidate_count"], 2)
        with self.assertRaisesRegex(ValueError, "extra"):
            instance.sample("remote", 128, 2, False)

    def test_cross_mode_output_difference_is_detected(self):
        base, calls = self.fake_base(result_ids=(99,))
        replay.install_replay(base, self.plan(), {})
        with self.assertRaisesRegex(AssertionError, "archived staged output"):
            base.RemoteExperiment().sample("remote", 128, 0, True)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

