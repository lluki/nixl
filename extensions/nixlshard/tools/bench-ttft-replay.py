#!/usr/bin/env python3
"""Replay archived remote requests through the byte-pinned TTFT harness.

Agent written. This driver changes only prompt selection; native/serving/base
harness bytes and the base harness's timing and correctness checks stay intact.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE_SHA256 = "db8263a1bcf463759b5ee4d83436adb80a6e005b2adf53708dbf3109210403f6"


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(reference, contexts, repeats, warmups, output_tokens):
    """Read and validate the complete candidate set before any HTTP call."""
    if len(set(contexts)) != len(contexts) or repeats < 1 or warmups < 0:
        raise ValueError("invalid expected context/repetition plan")
    expected_order = [(context, repeat, repeat < warmups)
                      for context in contexts for repeat in range(warmups + repeats)]
    expected = set(expected_order)
    params = {"temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}
    plan = {}
    for path in sorted(reference.glob("*/sample.json")):
        sample = json.loads(path.read_text())
        key = (sample["context_tokens"], sample["repeat"], sample["warmup"])
        if key not in expected or key in plan:
            raise ValueError(f"duplicate or unexpected archived candidate: {path.parent.name}")
        if sample.get("scenario") != "remote" or not sample.get("passed") or sample.get("receiver_mode") != "staged":
            raise ValueError("reference must contain successful staged remote candidates")
        requests = {}
        for name in ("request.json", "cold-control-request.json", "owner-seed-request.json", "measured-request.json"):
            requests[name] = json.loads((path.parent / name).read_text())
        request = requests["measured-request.json"]
        ids = request["input_ids"]
        if (len(ids) != key[0] or any(type(token) is not int or token < 0 for token in ids)
                or request.get("sampling_params") != params):
            raise ValueError("archived input length, token IDs or sampling parameters do not match this run")
        for other in requests.values():
            if other.get("input_ids") != ids or other.get("sampling_params") != params:
                raise ValueError("archived owner/cold/measured request IDs or parameters differ")
        outputs = [json.loads((path.parent / name).read_text())
                   for name in ("cold-control.json", "owner-seed.json", "measured.json")]
        if any(output["output_ids"] != outputs[0]["output_ids"]
               or output["text"] != outputs[0]["text"] for output in outputs[1:]):
            raise ValueError("archived staged owner/cold/measured outputs differ")
        expected_tokens = ((key[0] - 1) // 64) * 64
        cache = sample["cache"]
        if cache["device"] or cache["host"] or cache["storage"] != expected_tokens:
            raise ValueError("reference is not a pure storage hit")
        if not sample["remote_proof"]["exact"]:
            raise ValueError("reference has no exact native byte proof")
        plan[key] = {"context_tokens": key[0], "repeat": key[1], "warmup": key[2],
                     "input_ids": ids, "sampling_params": params,
                     "input_ids_sha256": digest(ids),
                     "sampling_params_sha256": digest(params),
                     "reference_artifact": str(path.parent.resolve()),
                     "reference_request_sha256": file_sha(path.parent / "measured-request.json"),
                     "reference_sample_sha256": file_sha(path),
                     "reference_output_ids": outputs[0]["output_ids"],
                     "reference_text": outputs[0]["text"]}
    if set(plan) != expected:
        raise ValueError(f"incomplete archived plan: expected {len(expected)} candidates, found {len(plan)}")
    if len({entry["input_ids_sha256"] for entry in plan.values()}) != len(plan):
        raise ValueError("reference reuses a complete prompt across candidates")
    return [plan[key] for key in expected_order]


def load_base(path):
    if file_sha(path) != BASE_SHA256:
        raise ValueError("base harness bytes do not match the frozen fcacef2 tool")
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("nixlshard_frozen_ttft", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_replay(base, plan, manifest):
    """Use the existing sample(ids=...) API; do not alter payload or validation."""
    original = base.RemoteExperiment
    class ExactReplay(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replay_index = 0
            (self.artifact_dir / "replay-plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if self.archives is not None:
                self.archives.archive(self.artifact_dir, "replay-plan.zip", recursive=False)

        def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
            if self.replay_index >= len(plan):
                raise ValueError("base harness requested an extra candidate")
            entry = plan[self.replay_index]
            if (context, repetition, warmup) != (entry["context_tokens"], entry["repeat"], entry["warmup"]):
                raise ValueError("base harness candidate order differs from the validated replay plan")
            if ids is not None and ids != entry["input_ids"]:
                raise ValueError("caller-provided prompt differs from replay plan")
            self.replay_index += 1
            answer = super().sample(scenario, context, repetition, warmup,
                                    ids=list(entry["input_ids"]), reference=reference)
            candidate = self.artifact_dir / f"{self.sequence:04d}-{context}-remote"
            measured = json.loads((candidate / "measured.json").read_text())
            if (measured["output_ids"] != entry["reference_output_ids"]
                    or measured["text"] != entry["reference_text"]):
                raise AssertionError("direct replay output differs from the archived staged output")
            return answer
    base.RemoteExperiment = ExactReplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--reference-artifact-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="no imports, HTTP or generation")
    parser.add_argument("base_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.base_args[1:] if args.base_args[:1] == ["--"] else args.base_args
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument("--contexts", type=int, nargs="+", required=True)
    config.add_argument("--scenarios", nargs="+", required=True)
    config.add_argument("--warmups", type=int, required=True)
    config.add_argument("--repeats", type=int, required=True)
    config.add_argument("--output-tokens", type=int, default=16)
    config.add_argument("--page-size", type=int, default=64)
    config.add_argument("--direct-receive", action="store_true")
    settings, _ = config.parse_known_args(argv)
    if settings.scenarios != ["remote"] or not settings.direct_receive or settings.page_size != 64:
        parser.error("replay requires exclusive direct remote profile with page64")
    if file_sha(args.base_harness) != BASE_SHA256:
        parser.error("base harness SHA does not match the frozen fcacef2 tool")
    plan = validate_plan(args.reference_artifact_dir, settings.contexts, settings.repeats,
                         settings.warmups, settings.output_tokens)
    manifest = {"schema_version": 1, "mode": "exact_archived_staged_inputs",
                "base_harness_sha256": BASE_SHA256, "replay_driver_sha256": file_sha(Path(__file__)),
                "reference_artifact_dir": str(args.reference_artifact_dir.resolve()),
                "candidate_count": len(plan), "plan_sha256": digest(plan), "candidates": plan,
                "scope": "same archived token IDs/sampling parameters and checked output equality; all timing and native/cache validation from byte-pinned base harness"}
    if args.validate_only:
        print(json.dumps({k:v for k,v in manifest.items() if k != "candidates"}, indent=2))
        return
    base = load_base(args.base_harness)
    install_replay(base, plan, manifest)
    sys.argv = [str(args.base_harness)] + argv
    base.main()


if __name__ == "__main__":
    main()

