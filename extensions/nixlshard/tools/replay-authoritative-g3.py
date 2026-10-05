#!/usr/bin/env python3
"""Agent written: exact accepted-input replay over the immutable G3 TTFT harness.

Only prompt selection/output comparison changes. No timing, tier preparation,
native validation, or runtime source is replaced. Source belongs in Git/RAID,
never in the credential-free GCS artifact directories.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE_SHA = "d5ad6397fd37dc63d7cc82eb90d7c2ea8e00cee3a024a7731c542a0cbe984fc0"
REPLAY_SHA = "53d030a177c642eee59617a9395cd29c1dfc8915b8310274fbf2c8f0b22febb8"
TRACE_SHA = "5ba14e66ed3f09595f6306011a47b8b9a51256f4ffd219e8b2e35d611ad661bb"
TIERS = ("cold", "gpu", "host", "ssd")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_digest(plan):
    # Location is provenance, not identity: exact archive bytes may be restored elsewhere.
    return digest([{k: v for k, v in entry.items() if k != "reference_artifact"} for entry in plan])


def read(path):
    return json.loads(path.read_text())


def load_replay(path):
    if sha(path) != REPLAY_SHA:
        raise ValueError("frozen replay helper SHA mismatch")
    spec = importlib.util.spec_from_file_location("frozen_g3_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_records(directory, expected):
    """Cross-check individual records against the archived ordered journal."""
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    paths = sorted(directory.glob("*/sample.json"))
    if len(paths) != len(expected) or len(rows) != len(expected):
        raise ValueError("incomplete or extra archived sample plan")
    answer = []
    for ordinal, (path, row, key) in enumerate(zip(paths, rows, expected), 1):
        record = read(path)
        actual = (record["context_tokens"], record["repeat"], record["warmup"], record["scenario"])
        if actual != key or record != row or not record.get("passed"):
            raise ValueError("archived sample order/journal fingerprint differs")
        if path.parent.name != f"{ordinal:04d}-{key[0]}-{key[3]}" or record.get("artifact") != path.parent.name:
            raise ValueError("archived sample sequence or artifact fingerprint differs")
        measured = read(path.parent / "measured.json")
        fingerprint = hashlib.sha256(json.dumps(measured["output_ids"]).encode()).hexdigest()
        if record.get("output_sha256") != fingerprint:
            raise ValueError("archived output fingerprint differs")
        answer.append((path, record, measured))
    return answer


def validate_local(reference, contexts, repeats, warmups, output_tokens):
    if len(set(contexts)) != len(contexts) or repeats < 1 or warmups < 0:
        raise ValueError("invalid replay plan")
    params = {"temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}
    plan = []
    unique = set()
    for context in contexts:
        expected = [(context, repeat, repeat < warmups, tier)
                    for repeat in range(warmups + repeats) for tier in TIERS]
        records = validated_records(reference / f"context-{context}", expected)
        preceding_cold = None
        for path, sample, measured in records:
            directory = path.parent
            requests = [read(directory / name) for name in ("request.json", "measured-request.json")]
            request = requests[-1]
            ids = request["input_ids"]
            if (len(ids) != context or any(type(token) is not int or token < 0 for token in ids)
                    or any(x.get("input_ids") != ids or x.get("sampling_params") != params for x in requests)):
                raise ValueError("archived input length/token IDs/sampling mismatch")
            tier = sample["scenario"]
            cached = ((context - 1) // 64) * 64
            cache = sample["cache"]
            if (any(cache.get(name) != (cached if name == {"gpu": "device", "host": "host", "ssd": "storage"}.get(tier) else 0)
                    for name in ("device", "host", "storage"))):
                raise ValueError("reference has an invalid tier mask")
            if tier == "cold":
                if digest(ids) in unique:
                    raise ValueError("reference reuses a cold/seed prompt")
                unique.add(digest(ids))
                preceding_cold = (ids, measured)
            elif tier == "gpu":
                if preceding_cold is None or ids != preceding_cold[0]:
                    raise ValueError("cold/GPU pair input IDs differ")
                seed = preceding_cold[1]
                if measured["output_ids"] != seed["output_ids"] or measured["text"] != seed["text"]:
                    raise ValueError("cold/GPU pair output differs")
            else:
                seed_request = read(directory / "seed-request.json")
                seed = read(directory / "seed.json")
                if seed_request.get("input_ids") != ids or seed_request.get("sampling_params") != params:
                    raise ValueError("seed/measured inputs or sampling differ")
                if measured["output_ids"] != seed["output_ids"] or measured["text"] != seed["text"]:
                    raise ValueError("seed/measured outputs differ")
                if digest(ids) in unique:
                    raise ValueError("reference reuses a cold/seed prompt")
                unique.add(digest(ids))
            if tier == "ssd" and not sample.get("local_proof", {}).get("exact"):
                raise ValueError("SSD reference has no exact byte proof")
            plan.append(dict(context_tokens=context, repeat=sample["repeat"], warmup=sample["warmup"],
                             scenario=tier, input_ids=ids, sampling_params=params,
                             input_ids_sha256=digest(ids), sampling_params_sha256=digest(params),
                             reference_artifact=str(directory.resolve()),
                             reference_request_sha256=sha(directory / "measured-request.json"),
                             reference_sample_sha256=sha(path),
                             reference_output_ids=measured["output_ids"], reference_text=measured["text"]))
    return plan


def install_local(base, plan, manifest):
    original = base.Experiment

    class ExactLocal(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replay_index = 0
            (self.artifact_dir / "replay-plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if self.archives is not None:
                self.archives.archive(self.artifact_dir, "replay-plan.zip", recursive=False)

        def sample(self, scenario, context, repetition, warmup, ids=None, reference=None):
            if self.replay_index >= len(plan):
                raise ValueError("extra replay sample")
            entry = plan[self.replay_index]
            if (context, repetition, warmup, scenario) != (
                    entry["context_tokens"], entry["repeat"], entry["warmup"], entry["scenario"]):
                raise ValueError("runtime sample order differs from replay plan")
            if ids is not None and ids != entry["input_ids"]:
                raise ValueError("caller cold/GPU IDs differ from replay plan")
            self.replay_index += 1
            answer = super().sample(scenario, context, repetition, warmup,
                                    ids=list(entry["input_ids"]), reference=reference)
            result = answer[1]
            directory = self.artifact_dir / f"{self.sequence:04d}-{context}-{scenario}"
            passed = result["output_ids"] == entry["reference_output_ids"] and result["text"] == entry["reference_text"]
            verification = dict(passed=passed, reference_request_sha256=entry["reference_request_sha256"],
                                reference_sample_sha256=entry["reference_sample_sha256"],
                                input_ids_sha256=entry["input_ids_sha256"],
                                sampling_params_sha256=entry["sampling_params_sha256"])
            (directory / "replay-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
            if not passed:
                self.samples[-1].update(passed=False, error="output differs from accepted baseline")
                (directory / "sample.json").write_text(json.dumps(self.samples[-1], indent=2) + "\n")
                journal = self.artifact_dir / "samples.jsonl"
                lines = journal.read_text().splitlines()
                lines[-1] = json.dumps(self.samples[-1])
                journal.write_text("\n".join(lines) + "\n")
            if self.archives is not None:
                self.archives.archive(directory, directory.name + "-replay-verification.zip")
            if not passed:
                raise AssertionError("replay output differs from accepted baseline")
            return answer
    base.Experiment = ExactLocal


def prepare(args, argv):
    replay = load_replay(args.replay_helper)
    replay.checked_base_sha(args.base_harness, BASE_SHA)
    if sha(args.base_harness.with_name("ttft_trace.py")) != TRACE_SHA:
        raise ValueError("frozen trace normalizer SHA mismatch")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=64)
    settings, _ = parser.parse_known_args(argv)
    if settings.page_size != 64 or not settings.contexts or len(set(settings.contexts)) != len(settings.contexts):
        raise ValueError("replay requires distinct page64 contexts")
    contexts = args.reference_contexts
    if args.mode == "remote":
        if settings.scenarios != ["remote"]:
            raise ValueError("remote replay requires exclusive remote scenario")
        full_plan = replay.validate_plan(args.reference_artifact_dir, contexts,
                                        settings.repeats, settings.warmups, settings.output_tokens)
        records = validated_records(args.reference_artifact_dir,
                                    [(x["context_tokens"], x["repeat"], x["warmup"], "remote") for x in full_plan])
        for (path, _, measured), entry in zip(records, full_plan):
            if measured["output_ids"] != entry["reference_output_ids"] or measured["text"] != entry["reference_text"]:
                raise ValueError("remote output differs from validated plan")
    else:
        if settings.scenarios != list(TIERS):
            raise ValueError("local replay requires cold gpu host ssd in that order")
        full_plan = validate_local(args.reference_artifact_dir, contexts,
                                   settings.repeats, settings.warmups, settings.output_tokens)
    if any(x not in contexts for x in settings.contexts):
        raise ValueError("requested context is absent from accepted plan")
    plan = [entry for context in settings.contexts for entry in full_plan if entry["context_tokens"] == context]
    manifest = dict(schema_version=1, mode=args.mode, base_harness_sha256=BASE_SHA,
                    trace_normalizer_sha256=TRACE_SHA,
                    frozen_replay_helper_sha256=REPLAY_SHA, driver_sha256=sha(Path(__file__)),
                    reference_artifact_dir=str(args.reference_artifact_dir.resolve()),
                    complete_plan_sha256=plan_digest(full_plan), complete_candidate_count=len(full_plan),
                    plan_sha256=plan_digest(plan), candidate_count=len(plan), candidates=plan,
                    scope="Exact accepted token IDs, sampling and output. Frozen harness timings/validation unchanged; host-pressure preparation prompts remain random and excluded.")
    if args.expected_plan_sha256 is not None and args.expected_plan_sha256 != manifest["complete_plan_sha256"]:
        raise ValueError("accepted complete-plan fingerprint differs")
    return replay, plan, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["remote", "local"], required=True)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--replay-helper", type=Path, required=True)
    parser.add_argument("--reference-artifact-dir", type=Path, required=True)
    parser.add_argument("--reference-contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--expected-plan-sha256", help="complete accepted reference-plan fingerprint; required for traffic")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("base_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.base_args[1:] if args.base_args[:1] == ["--"] else args.base_args
    replay, plan, manifest = prepare(args, argv)
    if args.validate_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "candidates"}, indent=2))
        return
    if args.expected_plan_sha256 is None:
        parser.error("--expected-plan-sha256 is required before traffic")
    base = replay.load_base(args.base_harness, BASE_SHA)
    if args.mode == "remote":
        replay.install_replay(base, plan, manifest)
    else:
        install_local(base, plan, manifest)
    sys.argv = [str(args.base_harness)] + argv
    base.main()


if __name__ == "__main__":
    main()
