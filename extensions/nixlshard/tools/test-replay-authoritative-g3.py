"""Agent written: archived-plan and replay-hook safety regressions; no serving traffic."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("driver", Path(__file__).with_name("replay-authoritative-g3.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
LOCAL = Path(os.environ["G3_REPLAY_LOCAL_REFERENCE"]) if os.environ.get("G3_REPLAY_LOCAL_REFERENCE") else None
REMOTE = Path(os.environ["G3_REPLAY_REMOTE_REFERENCE"]) if os.environ.get("G3_REPLAY_REMOTE_REFERENCE") else None
BASE = Path(os.environ.get("G3_REPLAY_BASE_HARNESS", str(Path(__file__).with_name("bench-ttft.py"))))
REPLAY = Path(os.environ.get("G3_REPLAY_HELPER", str(BASE.with_name("bench-ttft-replay.py"))))
FIXTURES_AVAILABLE = all(path is not None and path.exists() for path in (LOCAL, REMOTE, BASE, REPLAY))
FIXTURE_SKIP = "Configure G3_REPLAY_LOCAL_REFERENCE and G3_REPLAY_REMOTE_REFERENCE with the actual archived baseline fixtures; optional G3_REPLAY_BASE_HARNESS/G3_REPLAY_HELPER override sibling frozen sources. No synthetic performance data is substituted."


@unittest.skipUnless(FIXTURES_AVAILABLE, FIXTURE_SKIP)
class Plans(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("NIXLSHARD_TEST_DIR"))
        self.root = Path(self.temp.name)
        shutil.copytree(LOCAL / "context-512", self.root / "context-512")

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return driver.validate_local(self.root, [512], 5, 1, 16)

    def edit(self, path, change):
        item = json.loads(path.read_text())
        change(item)
        path.write_text(json.dumps(item))

    def test_real_complete_local120_and_remote30_preflight_without_http(self):
        self.assertEqual(len(driver.validate_local(LOCAL, [512, 1024, 2048, 4096, 8192], 5, 1, 16)), 120)
        replay = driver.load_replay(REPLAY)
        self.assertEqual(len(replay.validate_plan(REMOTE, [512, 1024, 2048, 4096, 8192], 5, 1, 16)), 30)

    def test_cold_gpu_pair_changed_consistently_is_rejected(self):
        for name in ("request.json", "measured-request.json"):
            self.edit(self.root / "context-512/0002-512-gpu" / name,
                      lambda item: item["input_ids"].__setitem__(0, item["input_ids"][0] + 1))
        with self.assertRaisesRegex(ValueError, "cold/GPU pair"):
            self.validate()

    def test_order_journal_and_sample_disagreement_is_rejected(self):
        path = self.root / "context-512/samples.jsonl"
        lines = path.read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "order/journal"):
            self.validate()

    def test_forged_matching_record_fingerprint_is_rejected(self):
        directory = self.root / "context-512"
        self.edit(directory / "0001-512-cold/sample.json", lambda item: item.update(output_sha256="0" * 64))
        lines = (directory / "samples.jsonl").read_text().splitlines()
        lines[0] = (directory / "0001-512-cold/sample.json").read_text()
        (directory / "samples.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "output fingerprint"):
            self.validate()

    def test_seed_sampling_mutation_is_rejected(self):
        self.edit(self.root / "context-512/0003-512-host/seed-request.json",
                  lambda item: item["sampling_params"].update(temperature=1))
        with self.assertRaisesRegex(ValueError, "seed/measured inputs or sampling"):
            self.validate()

    def test_missing_whole_sample_is_rejected(self):
        shutil.rmtree(self.root / "context-512/0024-512-ssd")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.validate()

    def test_base_bytes_and_accepted_plan_gate(self):
        replay = driver.load_replay(REPLAY)
        changed = self.root / "bench.py"
        changed.write_bytes(BASE.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "frozen SHA"):
            replay.checked_base_sha(changed, driver.BASE_SHA)
        args = SimpleNamespace(mode="local", reference_artifact_dir=self.root, reference_contexts=[512],
                               replay_helper=REPLAY, base_harness=BASE, expected_plan_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "complete-plan fingerprint"):
            driver.prepare(args, ["--contexts", "512", "--scenarios", "cold", "gpu", "host", "ssd",
                                  "--warmups", "1", "--repeats", "5"])

    def test_plan_identity_survives_exact_archive_relocation(self):
        old = driver.validate_local(LOCAL, [512], 5, 1, 16)
        self.assertEqual(driver.plan_digest(old), driver.plan_digest(self.validate()))


@unittest.skipUnless(FIXTURES_AVAILABLE, FIXTURE_SKIP)
class Hook(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("NIXLSHARD_TEST_DIR"))
        self.root = Path(self.temp.name)
        self.plan = driver.validate_local(LOCAL, [512], 5, 1, 16)[:2]
        root = self.root

        class Original:
            def __init__(self):
                self.artifact_dir, self.archives = root, None
                self.sequence, self.calls, self.samples = 0, [], []
                self.bad_output = False

            def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
                self.sequence += 1
                self.calls.append((scenario, list(ids), reference))
                result = {"output_ids": self_plan[self.sequence - 1]["reference_output_ids"],
                          "text": self_plan[self.sequence - 1]["reference_text"]}
                if self.bad_output:
                    result["text"] += "changed"
                directory = root / f"{self.sequence:04d}-{context}-{scenario}"
                directory.mkdir()
                sample = {"passed": True}
                self.samples.append(sample)
                (directory / "sample.json").write_text(json.dumps(sample))
                with (root / "samples.jsonl").open("a") as file:
                    file.write(json.dumps(sample) + "\n")
                return ids, result

        self_plan = self.plan
        base = SimpleNamespace(Experiment=Original)
        driver.install_local(base, self.plan, {"test": "no HTTP"})
        self.instance = base.Experiment()

    def tearDown(self):
        self.temp.cleanup()

    def test_hook_rejects_order_before_original_call(self):
        with self.assertRaisesRegex(ValueError, "runtime sample order"):
            self.instance.sample("gpu", 512, 0, True)
        self.assertEqual(self.instance.calls, [])

    def test_hook_preserves_cold_gpu_ids_and_reference(self):
        ids, result = self.instance.sample("cold", 512, 0, True)
        self.instance.sample("gpu", 512, 0, True, ids, result)
        self.assertEqual(self.instance.calls[0][1], self.plan[0]["input_ids"])
        self.assertEqual(self.instance.calls[1][1], self.plan[1]["input_ids"])
        self.assertIs(self.instance.calls[1][2], result)

    def test_hook_rejects_caller_alias_inputs_before_request(self):
        self.instance.sample("cold", 512, 0, True)
        with self.assertRaisesRegex(ValueError, "caller cold/GPU"):
            self.instance.sample("gpu", 512, 0, True, [1] * 512)
        self.assertEqual(len(self.instance.calls), 1)

    def test_output_mismatch_marks_saved_sample_failed(self):
        self.instance.bad_output = True
        with self.assertRaisesRegex(AssertionError, "accepted baseline"):
            self.instance.sample("cold", 512, 0, True)
        self.assertFalse(self.instance.samples[-1]["passed"])
        saved = json.loads((self.root / "0001-512-cold/sample.json").read_text())
        self.assertFalse(saved["passed"])
        self.assertFalse(json.loads((self.root / "samples.jsonl").read_text())["passed"])
        self.assertFalse(json.loads((self.root / "0001-512-cold/replay-verification.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
