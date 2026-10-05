"""Agent written: saved-artifact augmentation of input/plugin witnesses and means.

No source artifacts or credentials are copied into report data. No serving calls.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

role, baseline = map(Path, sys.argv[1:3])
if not __debug__: raise RuntimeError("audit requires Python assertions enabled")
root = role / "remote-full"
def read(path):
    return json.loads(path.read_text())
def fs(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

normalized_path = root / "remote-only-normalized.json"
audit_path = root / "independent-audit.json"
normal = read(normalized_path)
audit = read(audit_path)
previous_normal_sha, previous_audit_sha = fs(normalized_path), fs(audit_path)
launch = read(role / "launch.json")
owner_launch = read(role / "owner-launch-for-bench.json")
owner_dir = Path(os.environ.get("AUDIT_OWNER_ARTIFACTS",str(Path("/workspace/audit-inputs") / launch["run"])))
plugins = {}
for name, path, source in [
    ("requester", role / "native-plugin-maps-final.json", launch),
    ("owner", owner_dir / "owner-native-plugin-maps-final.json", owner_launch)]:
    maps = read(path)
    paths = [line for item in maps for line in item["plugins"]]
    for plugin in ["libplugin_POSIX.so", "libplugin_UCX.so"]:
        assert any(source["native_prefix"] + "/lib/plugins/" + plugin in line for line in paths)
    plugins[name] = dict(artifact=str(path), sha256=fs(path), native_prefix=source["native_prefix"],
                         actual_posix_ucx_maps_verified=True)
plan = read(root / "replay-plan.json")
assert plan["driver_sha256"] == "e1855a942cca2122d33ebd7a5f4ffe835bb5d2e71d3612e6e5c77164dc312873"
assert plan["base_harness_sha256"] == "d5ad6397fd37dc63d7cc82eb90d7c2ea8e00cee3a024a7731c542a0cbe984fc0"
assert plan["complete_plan_sha256"] == "498fe74c9de9fb46f1775f4ece27d5332775f672545cfa6a455c9ff72c5b3997"
assert plan["candidate_count"] == 30
accepted = {}
for path in (baseline / "remote-full").glob("*/sample.json"):
    sample = read(path)
    accepted[(sample["context_tokens"], sample["repeat"], sample["warmup"])] = path.parent
rows = []
for row in normal["samples"]:
    key = (row["context_tokens"], row["repeat"], row["warmup"])
    original = accepted[key]
    request_path = Path(row["input_ids_artifact"])
    request = read(request_path)
    candidate_output = read(request_path.with_name("measured.json"))
    accepted_request = read(original / "measured-request.json")
    accepted_output = read(original / "measured.json")
    assert request["input_ids"] == accepted_request["input_ids"]
    assert request["sampling_params"] == accepted_request["sampling_params"]
    assert candidate_output["output_ids"] == accepted_output["output_ids"]
    assert candidate_output["text"] == accepted_output["text"]
    entry = next(x for x in plan["candidates"] if (x["context_tokens"], x["repeat"], x["warmup"]) == key)
    plan_original = Path(entry["reference_artifact"])
    assert entry["reference_sample_sha256"] == fs(plan_original / "sample.json")
    assert entry["reference_request_sha256"] == fs(plan_original / "measured-request.json")
    plan_request = read(plan_original / "measured-request.json")
    assert request["input_ids"] == plan_request["input_ids"] and request["sampling_params"] == plan_request["sampling_params"]
    row["accepted_baseline"] = dict(input_ids_artifact=str(original / "measured-request.json"),
                                   input_ids_artifact_sha256=fs(original / "measured-request.json"),
                                   sample_artifact_sha256=fs(original / "sample.json"),
                                   output_artifact_sha256=fs(original / "measured.json"))
    blocks = row["critical_blocks"]
    values = dict(ttft_ms=(row["first_token_ns"]-row["request_start_ns"])/1e6,
                  owner_metadata_ms=0, owner_payload_ms=0, owner_ucx_ms=0, owner_copy_ms=0,
                  receiver_copy_ms=0, rpc_residual_ms=0, other_ms=0, framework_h2d_forward_ms=0)
    for block in blocks:
        ns = block["end_ns"]-block["start_ns"]
        if block["category"] == "remote_rpc":
            for field, target in [("owner_metadata_ns", "owner_metadata_ms"), ("owner_posix_ns", "owner_payload_ms"),
                                  ("owner_ucx_ns", "owner_ucx_ms"), ("owner_staging_copy_ns", "owner_copy_ms")]:
                values[target] += block[field]/1e6
            child = sum(block[field] for field in ["owner_metadata_ns","owner_posix_ns","owner_ucx_ns","owner_staging_copy_ns"])
            assert child <= ns
            values["rpc_residual_ms"] += (ns-child)/1e6
        elif block["category"] == "staging_copy":
            values["receiver_copy_ms"] += ns/1e6
        elif block["category"] == "other":
            values["other_ms"] += ns/1e6
        elif block["category"] == "framework_h2d_forward":
            values["framework_h2d_forward_ms"] += ns/1e6
    rows.append(dict(context_tokens=key[0], repeat=key[1], warmup=key[2], **values))
means = {}
for context in [512,1024,2048,4096,8192]:
    selected = [row for row in rows if row["context_tokens"]==context and not row["warmup"]]
    assert len(selected)==5
    means[str(context)] = {field:statistics.mean(row[field] for row in selected) for field in values}
normal.update(replay_provenance=dict(artifact=str(root/"replay-plan.json"), sha256=fs(root/"replay-plan.json"),
                                     driver_sha256=plan["driver_sha256"],
                                     complete_plan_sha256=plan["complete_plan_sha256"]),
              allowed_baseline_implementation_difference="native build only",
              accepted_baseline_role=str(baseline),
              derivation=dict(previous_normalized_sha256=previous_normal_sha, previous_audit_sha256=previous_audit_sha))
normalized_path.write_text(json.dumps(normal,indent=2)+"\n")
audit.update(actual_plugin_maps=plugins, exact_accepted_input_sampling_output_all30_verified=True,
             accepted_baseline_role=str(baseline), accepted_replay_plan_sha256=plan["complete_plan_sha256"],
             replay_driver_sha256=plan["driver_sha256"], mean_measured=means,
             normalized_sha256=fs(normalized_path), previous_audit_sha256=previous_audit_sha,
             baseline_plugin_maps_scope="Baseline archived filtered maps prove native library/binding prefix; POSIX/UCX plugin mapping was not captured there. Candidate mappings are explicit.",
             row_mean_scope="Means of per-request exclusive requester-clock intervals and nested owner durations; no sum of independent medians or pure CPU overhead claim.")
audit_path.write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(dict(normalized_sha256=fs(normalized_path),audit_sha256=fs(audit_path),
                      maximum_other_percent=audit["maximum_other_percent"],means=means),indent=2))
