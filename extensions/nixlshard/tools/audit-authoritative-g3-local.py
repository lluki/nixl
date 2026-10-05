"""Agent written: audit saved local120 and exact accepted baseline comparisons.

Saved JSON only; no HTTP/native/model calls. See audit-authoritative-g3.md.
"""
import json,hashlib,statistics,sys,os
if not __debug__: raise RuntimeError("audit requires Python assertions enabled")
from pathlib import Path
from collections import Counter
def read(p):return json.loads(Path(p).read_text())
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def fs(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric_delta(a,b,component):
 return sum(v-a.get(k,0) for k,v in b.items() if k.startswith("sglang:nixlshard_component_bytes_total") and ('component="'+component+'"') in k)
def validate_cache_mask(cache,tier,context):
 expected=((context-1)//64)*64
 fields=["device","host","storage"]
 if tier=="cold":assert cache["cached_tokens"]==0 and all(cache[k]==0 for k in fields)
 else:
  field={"gpu":"device","host":"host","local_ssd":"storage"}[tier]
  assert cache[field]==expected and all(cache[k]==0 for k in fields if k!=field)
  assert cache["cached_tokens"]==sum(cache[k] for k in fields)
root=Path(sys.argv[1]);assert (root/"epochs.exit").read_text().strip()=="0"
launch=read(root/"launch.json");cfg=read(root/"config.json");domain=read(root/"domain-witness.json")[0];maps=read(root/"runtime-maps.json")
native=os.environ.get("AUDIT_EXPECTED_NATIVE","3310a0cd924619f60f42978247c1817206a1f008");sg="72726e814f09016d4ad077f2f3cd4ef7e7968f5e";tool="7125d6b4c5935b4067a690bacffae6096f232dd4"
assert len(native)==40 and all(c in "0123456789abcdef" for c in native)
assert launch["source_heads"]["nixl"]==native and launch["serving_source"]==sg and launch["tool_source"]==tool
assert any(launch["native_prefix"]+"/lib/libnixlshard.so" in str(x) and "/nixlshard/_bindings." in str(x) for x in maps)
assert cfg["agent"]["numa_node"]==domain["numa_node"]==0 and cfg["agent"]["direct_receive"] and domain["registration_mode"]=="EXPLICIT"
assert domain["unit_bytes"]==domain["min_object_bytes"]==domain["max_object_bytes"]==16777216 and domain["key_bytes"]==32
remote_root=Path(os.environ.get("AUDIT_REMOTE_ROLE","/scratch/nixlshard-v2/authoritative-g3/20261005-a10-b7-remote-direct-exact/requester"))
remote_launch=read(remote_root/"launch.json");remote_domain=read(remote_root/"domain-witness.json")[0]
assert remote_domain==domain and launch["model_revision"]==remote_launch["model_revision"]
for k in ["boot_id","gpu","gpu_numa","hca_node_guid","hca_port","hca_numa","scratch_mount","architecture"]:assert launch["hardware"][k]==remote_launch["hardware"][k]
control=[json.loads(x) for x in (root/"epoch-supervisor/events.jsonl").read_text().splitlines()]
assert control[-1]["phase"]=="supervisor_completed" and control[-1]["passed"]
assert all(x["status"]==200 for x in control if x["phase"]=="control_returned")
for c in [512,1024,2048,4096,8192]:
 phases={x["phase"]:x for x in control if x.get("context")==c}
 assert phases["closed_file_descriptors"]["remaining"]==[]
 assert phases["closed_file_descriptors"]["monotonic_ns"]<phases["unlink_started"]["monotonic_ns"]<phases["unlink_completed"]["monotonic_ns"]<phases["epoch_ready"]["monotonic_ns"]<phases["benchmark_started"]["monotonic_ns"]
 assert phases["epoch_completed"]["returncode"]==0 and phases["epoch_completed"]["samples"]==24
allrows=[];details=[];refs=[];comparison=[];outliers=[]
def local_profile(value,config,witness,pin):
 command=value["command"][:];command[command.index("--hicache-storage-backend-extra-config")+1]="@<role-config>"
 config=json.loads(json.dumps(config));config["agent"].pop("name",None)
 for disk in config["agent"]["disks"]:disk["path"]="<fresh-disposable-local-file>";disk.pop("reset",None)
 hardware={k:value["hardware"][k] for k in ["boot_id","gpu","gpu_numa","hca_node_guid","hca_port","hca_numa","scratch_mount","architecture"]}
 return {"native_build":pin,"sglang_commit":sg,"model_revision":value["model_revision"],"hardware":hardware,"serving":{"command":command,"packages":value["packages"],"environment":{k:v for k,v in value["environment"].items() if k not in ["TMPDIR","SGLANG_REQUEST_TIMELINE_DIR"]},"kv_bytes_per_page":value["kv_bytes_per_page"]},"agent":config,"namespace":witness}
profile=local_profile(launch,cfg,domain,native)
reference=Path(sys.argv[2]) if len(sys.argv)>2 else None
if reference:
 ref_launch=read(reference/"launch.json");ref_cfg=read(reference/"config.json");ref_domain=read(reference/"domain-witness.json")[0]
 assert ref_domain==domain and ref_launch["model_revision"]==launch["model_revision"]
 for k in ["boot_id","gpu","gpu_numa","hca_node_guid","hca_port","hca_numa","scratch_mount","architecture"]:assert ref_launch["hardware"][k]==launch["hardware"][k]
 def normalized_cfg(value):
  value=json.loads(json.dumps(value));value["agent"].pop("name",None)
  for disk in value["agent"]["disks"]:disk["path"]="<disposable-file>";disk.pop("reset",None)
  return value
 assert normalized_cfg(ref_cfg)==normalized_cfg(cfg)
 ref_profile=local_profile(ref_launch,ref_cfg,ref_domain,ref_launch["source_heads"]["nixl"])
 assert ref_profile["native_build"]=="3310a0cd924619f60f42978247c1817206a1f008"
 allowed_profile=json.loads(json.dumps(ref_profile));allowed_profile["native_build"]=native
 assert allowed_profile==profile
for ctx in [512,1024,2048,4096,8192]:
 b=root/("context-"+str(ctx));ss=[json.loads(x) for x in (b/"samples.jsonl").read_text().splitlines()]
 assert len(ss)==24 and all(s["passed"] for s in ss)
 assert Counter((s["scenario"],s["repeat"],s["warmup"]) for s in ss)==Counter((sc,i,i==0) for sc in ["cold","gpu","host","ssd"] for i in range(6))
 normal=read(b/"request-diagnostics.json");assert len(normal["samples"])==24
 refs.append({"artifact":str(b/"request-diagnostics.json"),"sha256":fs(b/"request-diagnostics.json"),"samples_sha256":fs(b/"samples.jsonl")})
 rows={(r["tier"],r["repeat"],r["warmup"]):r for r in normal["samples"]}
 for s in ss:
  tier={"cold":"cold","gpu":"gpu","host":"host","ssd":"local_ssd"}[s["scenario"]];n=rows[(tier,s["repeat"],s["warmup"])];p=b/s["artifact"]
  assert n["storage_contract"]=="authoritative_g3_v2" and n["receiver_mode"]=="direct" and n["verified"]
  result=read(p/"measured.json");request=read(p/"measured-request.json");base=read(p/"request.json");ids=request["input_ids"];params=request["sampling_params"]
  assert ids==base["input_ids"] and params==base["sampling_params"] and len(ids)==ctx
  if reference:
   original=reference/("context-"+str(ctx))/s["artifact"]
   accepted=read(original/"measured-request.json");accepted_result=read(original/"measured.json")
   assert ids==accepted["input_ids"] and params==accepted["sampling_params"]
   assert result["output_ids"]==accepted_result["output_ids"] and result["text"]==accepted_result["text"]
   comparison.append({"context_tokens":ctx,"scenario":s["scenario"],"repeat":s["repeat"],"warmup":s["warmup"],"accepted_request_artifact":str(original/"measured-request.json"),"accepted_request_sha256":fs(original/"measured-request.json"),"accepted_sample_sha256":fs(original/"sample.json")})
  if tier!="cold":
   if tier=="gpu":
    cold=next(x for x in ss if x["scenario"]=="cold" and x["repeat"]==s["repeat"] and x["warmup"]==s["warmup"])
    cp=b/cold["artifact"];seed=read(cp/"measured.json");seedreq=read(cp/"measured-request.json")
   else:seed=read(p/"seed.json");seedreq=read(p/"seed-request.json")
   assert seedreq["input_ids"]==ids and seedreq["sampling_params"]==params
   assert seed["output_ids"]==result["output_ids"] and seed["text"]==result["text"]
  assert s["output_sha256"]==n["output_sha256"]==hashlib.sha256(json.dumps(result["output_ids"]).encode()).hexdigest()
  assert n["request_start_ns"]==result["request_start_ns"] and n["first_token_ns"]==result["first_token_ns"] and n["request_id"]==request["rid"]==result["request_id"]
  assert n["trace_events"]==read(p/"critical-path-raw.json") and all(e["rid"]==n["request_id"] for e in n["trace_events"])
  forward=next(e["start_ns"] for e in n["trace_events"] if e["stage"]=="first_forward_entry")
  events={}
  for batch in n["native_batches"]:
   assert batch["rid"]==n["request_id"]
   for i,e in enumerate(batch["events"]):
    ref="native:%s:%s"%(batch["batch_handle"],i);events[ref]=e
    assert n["request_start_ns"]<=e["start_ns"]<=e["end_ns"]<=forward
  assert [b["events"] for b in n["native_batches"]]==[e["events"] for e in n["trace_events"] if e["stage"]=="native_batch"]
  assert all(e["build_marker"]==native[:12] for e in n["trace_events"] if e["stage"]=="native_batch")
  cursor=n["request_start_ns"];other=0;localblocks=[]
  for block in n["critical_blocks"]:
   assert block["start_ns"]==cursor and block["end_ns"]>=cursor;cursor=block["end_ns"]
   if block["category"]=="other":other+=block["end_ns"]-block["start_ns"]
   if block["category"]=="local_ssd":
    e=events[block["evidence"][0]];localblocks.append(e)
    assert block["start_ns"]==e["start_ns"] and block["end_ns"]==e["end_ns"] and block["composition_only"]
    for k in ["owner_posix_ns","owner_metadata_ns","owner_metadata_bytes","owner_staging_copy_ns","owner_staging_copy_bytes"]:assert block[k]==e[k]
  assert cursor==n["first_token_ns"] and not n["unresolved_overlaps"]
  assert abs((cursor-n["request_start_ns"])/1e6-s["ttft_ms"])<1e-8
  if 100*other/(cursor-n["request_start_ns"])>10:
   outliers.append({"context_tokens":ctx,"tier":tier,"repeat":s["repeat"],"warmup":s["warmup"],"other_percent":100*other/(cursor-n["request_start_ns"]),"spans":[block for block in n["critical_blocks"] if block["category"]=="other"],"scope":"unattributed requester CLOCK_MONOTONIC wall intervals; no CPU/kernel cause inferred; all samples retained"})
  local=[e for e in events.values() if e["stage"]=="local_posix"]
  assert not any(e["stage"] in ["remote_rpc","staging_copy"] for e in events.values())
  cache=s["cache"];pages=(ctx-1)//64;expected=pages*16777216
  validate_cache_mask(cache,tier,ctx)
  before=read(p/"before-stats.json")["metrics"];after=read(p/"after-stats.json")["metrics"]
  if tier=="local_ssd":
   proof=s["local_proof"];assert proof["exact"] and proof["zero_copy_verified"] and proof["direct_receive"]
   assert proof["expected_bytes"]==expected and proof["direct_pages"]==pages and proof["fallback_pages"]==0
   assert proof["observed_events"]["direct_receive_segments"]==pages*2 and proof["observed_events"]["local_direct_fallbacks"]==0
   obs=proof["observed_bytes"]
   for k in ["posix_read","direct_receive","direct_local_read"]:assert obs[k]==expected
   assert obs["remote_read"]==obs["ucx_write"]==0
   assert len(local)==len(localblocks)==pages and sum(e["bytes"] for e in local)==expected
   assert n["source_proof"]["metadata_io"]==proof["metadata_io"]
   metadata=proof["metadata_io"];assert metadata["available"] and all(v>=0 for v in metadata["observed"].values())
   for e in local:
    assert e["direct_receive"] and e["destination_segments"]==2 and e["owner_read_bytes"]==e["bytes"]==16777216
    assert e["owner_metadata_bytes"]==4096 and e["owner_metadata_ns"]>0 and e["owner_posix_ns"]>0
    assert e["owner_staging_copy_ns"]==e["owner_staging_copy_bytes"]==0
    assert e["owner_posix_ns"]+e["owner_metadata_ns"]<=e["end_ns"]-e["start_ns"]
   assert metadata["observed"]["metadata_read_bytes"]>0 and sum(metadata["observed"].values())>=pages*4096
   for comp in ["metadata_read","metadata_write"]:assert metric_delta(before,proof["after"]["metrics"],comp)==metadata["observed"][comp+"_bytes"]
   for comp in ["posix_read","direct_receive","direct_local_read","remote_read","ucx_write","staging_copy"]:assert metric_delta(before,proof["after"]["metrics"],comp)==obs[comp]
   for w in n["service_windows"]:
    e=events[w["evidence"][0]];exp={"ssd":(e["owner_read_bytes"],e["owner_posix_ns"]),"metadata":(e["owner_metadata_bytes"],e["owner_metadata_ns"]),"staging_copy":(0,0)}
    assert (w["bytes"],w["duration_ns"])==exp[w["stage"]]
  else:
   assert not local
   for comp in ["posix_read","remote_read","direct_receive","direct_local_read"]:assert metric_delta(before,after,comp)==0
  n.update(input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),sampling_params_sha256=sha(params),input_ids_artifact=str(p/"measured-request.json"),input_ids_artifact_sha256=fs(p/"measured-request.json"),source_study_id=normal["study_id"],source_artifact=str(b/"request-diagnostics.json"))
  allrows.append(n)
  details.append({"context_tokens":ctx,"tier":tier,"repeat":s["repeat"],"warmup":s["warmup"],"ttft_ms":s["ttft_ms"],"other_percent":100*other/(cursor-n["request_start_ns"]),"metadata_ms":sum(e["owner_metadata_ns"] for e in local)/1e6,"payload_ms":sum(e["owner_posix_ns"] for e in local)/1e6,"receiver_copy_ms":0,"expected_payload_bytes":expected if tier=="local_ssd" else 0})
normalized={k:v for k,v in normal.items() if k not in ["samples","contexts","study_id"]};normalized.update(samples=allrows,contexts=[512,1024,2048,4096,8192],study_id=launch["run"],comparison_cohort="20261005-a10-b7-authoritative-g3",matched_hardware=launch["hardware"],source_pins={"native":native,"sglang":sg,"helper":tool},domain_witness=domain,local_profile_provenance=profile,profile_difference="local requester assigned512GiB SSD; remote requester diskless; remaining model/pool/rank/NUMA/native sources equal",source_artifacts=refs)
np=root/"local-normalized-120.json";np.write_text(json.dumps(normalized,indent=2)+"\n")
groups={}
for tier in ["cold","gpu","host","local_ssd"]:
 groups[tier]={}
 for ctx in [512,1024,2048,4096,8192]:
  ds=[x for x in details if x["tier"]==tier and x["context_tokens"]==ctx and not x["warmup"]]
  groups[tier][str(ctx)]={"mean_ttft_ms":statistics.mean(x["ttft_ms"] for x in ds),"p50_ttft_ms":statistics.median(x["ttft_ms"] for x in ds),"mean_metadata_ms":statistics.mean(x["metadata_ms"] for x in ds),"max_other_percent":max(x["other_percent"] for x in ds)}
audit={"passed":True,"samples":120,"measured":100,"storage_contract":"authoritative_g3_v2","source_pins":normalized["source_pins"],"actual_maps_domain_verified":True,"epoch_api_reset_fd_closure_unlink_order_verified":True,"input_comparison_scope":"GPU reuses preceding cold request exactly; host/local SSD each reuse exact own seed request. Prefixes across all four tiers are shape matched, not asserted identical.","same_remote_gpu_hardware_verified":True,"local_direct_posix_all30_samples_positive":True,"per_request_read_copy_bytes":0,"fallback_pages":0,"counter_scope":"full-generation counters include background backup metadata and store copies; positive per-RID trace proves receiver path","normalized_sha256":fs(np),"maximum_other_percent":max(x["other_percent"] for x in details),"maximum_measured_other_percent":max(x["other_percent"] for x in details if not x["warmup"]),"other_outlier":{"tier":"cold","context_tokens":512,"repeat":0,"warmup":True,"scheduler_received_to_metadata_query_ms":12.004662,"scope":"actual unknown scheduler wall gap before metadata_query; no CPU/kernel cause inferred; warmup retained, excluded only by planned warmup policy"},"measured_groups":groups,"rows":details,"source_artifacts":refs,"hardware":launch["hardware"],"domain":domain}
audit.pop("other_outlier",None)
audit.update(other_outliers=outliers,exact_accepted_baseline_replay_verified=bool(reference),accepted_input_comparisons=comparison,baseline_role=str(reference) if reference else None,allowed_implementation_difference="native build only" if reference else None,local_profile_provenance=profile,baseline_local_profile_provenance=ref_profile if reference else None,local_profile_normalization_scope="Disposable SSD path, config-file argument, process agent name and timeline/TMPDIR locations normalized; all model/serving/package/NUMA/domain/disk geometry settings retained.")
ap=root/"local-independent-audit.json";ap.write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps({"normalized_sha256":fs(np),"audit_sha256":fs(ap),"max_other":audit["maximum_other_percent"],"groups":groups}))
