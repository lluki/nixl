"""Agent written: add saved plugin/replay witnesses and exact per-row baseline links."""
import hashlib
import json
from pathlib import Path
import sys

if not __debug__:raise RuntimeError("audit requires assertions")
role,baseline=map(Path,sys.argv[1:3])
def read(path):return json.loads(path.read_text())
def fs(path):return hashlib.sha256(path.read_bytes()).hexdigest()
normal_path=role/"local-normalized-120.json";audit_path=role/"local-independent-audit.json"
normal=read(normal_path);audit=read(audit_path)
previous_normal,previous_audit=fs(normal_path),fs(audit_path)
launch=read(role/"launch.json")
maps=read(role/"native-plugin-maps-final.json")
lines=[line for record in maps for line in record["plugins"]]
for name in ["libplugin_POSIX.so","libplugin_UCX.so"]:
 assert any(launch["native_prefix"]+"/lib/plugins/"+name in line for line in lines)
baseline_profile=audit["baseline_local_profile_provenance"]
assert baseline_profile["native_build"]=="3310a0cd924619f60f42978247c1817206a1f008"
assert normal["local_profile_provenance"]["native_build"]=="128a65a5066a35d774b86241c5f2668cdd816965"
expected=json.loads(json.dumps(baseline_profile));expected["native_build"]=normal["local_profile_provenance"]["native_build"]
assert expected==normal["local_profile_provenance"]
accepted={}
for sample in baseline.glob("context-*/*/sample.json"):
 record=read(sample);accepted[(record["context_tokens"],record["repeat"],record["warmup"],record["scenario"])]=sample.parent
for row in normal["samples"]:
 scenario="ssd" if row["tier"]=="local_ssd" else row["tier"]
 key=(row["context_tokens"],row["repeat"],row["warmup"],scenario)
 original=accepted[key];candidate=Path(row["input_ids_artifact"])
 a=read(original/"measured-request.json");b=read(candidate)
 x=read(original/"measured.json");y=read(candidate.with_name("measured.json"))
 assert a["input_ids"]==b["input_ids"] and a["sampling_params"]==b["sampling_params"]
 assert x["output_ids"]==y["output_ids"] and x["text"]==y["text"]
 row["accepted_baseline"]=dict(input_ids_artifact=str(original/"measured-request.json"),
                               input_ids_artifact_sha256=fs(original/"measured-request.json"),
                               sample_artifact_sha256=fs(original/"sample.json"),
                               output_artifact_sha256=fs(original/"measured.json"))
context_replay=[]
for context in [512,1024,2048,4096,8192]:
 path=role/f"context-{context}"/"replay-plan.json"
 plan=read(path)
 assert plan["complete_plan_sha256"]=="883144a7d891c22b27a708e6cf13d93378730a0219eaad613470dac625469249"
 assert plan["driver_sha256"]=="e1855a942cca2122d33ebd7a5f4ffe835bb5d2e71d3612e6e5c77164dc312873"
 assert plan["candidate_count"]==24 and plan["complete_candidate_count"]==120
 context_replay.append(dict(artifact=str(path),sha256=fs(path),driver_sha256=plan["driver_sha256"],
                           complete_plan_sha256=plan["complete_plan_sha256"],selected_plan_sha256=plan["plan_sha256"]))
plugin_witness=dict(artifact=str(role/"native-plugin-maps-final.json"),
                    sha256=fs(role/"native-plugin-maps-final.json"),
                    native_prefix=launch["native_prefix"],actual_posix_ucx_maps_verified=True)
normal.update(accepted_baseline_role=str(baseline),allowed_baseline_implementation_difference="native build only",
              baseline_local_profile_provenance=baseline_profile,
              replay_provenance=context_replay,actual_plugin_maps=plugin_witness,
              derivation=dict(previous_normalized_sha256=previous_normal,previous_audit_sha256=previous_audit))
normal_path.write_text(json.dumps(normal,indent=2)+"\n")
audit.update(actual_plugin_maps=plugin_witness,replay_provenance=context_replay,
             exact_accepted_input_sampling_output_all120_verified=True,
             normalized_sha256=fs(normal_path),previous_audit_sha256=previous_audit)
audit_path.write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(dict(normalized_sha256=fs(normal_path),audit_sha256=fs(audit_path),
                      maximum_measured_other_percent=audit["maximum_measured_other_percent"],
                      other_outliers=audit["other_outliers"]),indent=2))
