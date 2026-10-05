"""Agent written: independently audit saved authoritative G3 remote TTFT artifacts.

No HTTP, model imports, native execution, or timing changes. Usage is documented
in audit-authoritative-g3.md. Assertions are correctness gates, so -O is rejected.
"""
import json,hashlib,statistics,sys
from pathlib import Path
from collections import Counter
def read(p): return json.loads(Path(p).read_text())
def sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
import os
if not __debug__: raise RuntimeError("audit requires Python assertions enabled")
NATIVE=os.environ.get("AUDIT_EXPECTED_NATIVE","3310a0cd924619f60f42978247c1817206a1f008")
assert len(NATIVE)==40 and all(c in "0123456789abcdef" for c in NATIVE)
SG="72726e814f09016d4ad077f2f3cd4ef7e7968f5e"
TOOL="7125d6b4c5935b4067a690bacffae6096f232dd4"
def configuration(role):
 launch=read(role/"launch.json"); owner=read(role/"owner-launch-for-bench.json")
 run=launch["run"]; op=Path(os.environ.get("AUDIT_OWNER_ARTIFACTS",str(Path("/workspace/audit-inputs")/run)))
 configs={"requester":read(role/"config.json"),"owner":read(op/"owner-config.json")}
 domains={"requester":read(role/"domain-witness.json"),"owner":read(op/"owner-domain-witness.json")}
 assert domains["requester"]==domains["owner"] and len(domains["owner"])==1
 domain=domains["owner"][0]; schema=json.loads(domain["namespace_id"])
 assert json.dumps(schema,sort_keys=True,separators=(",",":"))==domain["namespace_id"]
 assert domain["numa_node"]==0 and domain["registration_mode"]=="EXPLICIT" and domain["key_bytes"]==32
 assert domain["unit_bytes"]==domain["min_object_bytes"]==domain["max_object_bytes"]==16777216
 assert schema["segments"]==[8388608,8388608] and schema["revision"]==launch["model_revision"]
 assert not configs["requester"]["agent"]["disks"] and not configs["owner"]["agent"]["direct_receive"]
 result={"native_build":NATIVE,"sglang_commit":SG,"model_revision":launch["model_revision"],"hardware":{},"serving":{},"agent":{},"namespace":domain}
 refs={}
 for name,l in [("requester",launch),("owner",owner)]:
  assert l["source_heads"]["nixl"]==NATIVE and l["serving_source"]==SG and l["tool_source"]==TOOL
  assert l["native_build"]==NATIVE[:12]
  maps=read(role/"runtime-maps.json" if name=="requester" else op/"owner-runtime-maps.json")
  assert any(l["native_prefix"]+"/lib/libnixlshard.so" in z and "/nixlshard/_bindings." in z for z in [str(x) for x in maps])
  h=l["hardware"]; result["hardware"][name]={k:h[k] for k in ["boot_id","gpu","gpu_numa","hca_node_guid","hca_port","hca_numa","scratch_mount","architecture"]}
  cmd=l["command"][:]
  idx=cmd.index("--hicache-storage-backend-extra-config")+1;cmd[idx]="@<role-config>"
  result["serving"][name]={"command":cmd,"packages":l["packages"],"environment":{k:v for k,v in l["environment"].items() if k not in ["TMPDIR","SGLANG_REQUEST_TIMELINE_DIR"]},"kv_bytes_per_page":l["kv_bytes_per_page"]}
  cfg=configs[name];cfg=json.loads(json.dumps(cfg))
  if name=="requester":cfg["agent"].pop("direct_receive")
  for d in cfg["agent"]["disks"]:d["path"]="<fresh-disposable-owner-file>";d.pop("reset",None)
  result["agent"][name]=cfg
  refs[name]={"launch_sha256":sha(l),"domain_sha256":sha(domains[name]),"runtime_maps_sha256":sha(maps)}
 return result,refs
def audit(role,reference=None):
 b=role/"remote-full";assert (role/"remote-full.exit").read_text().strip()=="0"
 samples=[json.loads(x) for x in (b/"samples.jsonl").read_text().splitlines()]
 assert len(samples)==30 and all(s["passed"] for s in samples)
 assert Counter((s["context_tokens"],s["repeat"],s["warmup"]) for s in samples)==Counter((c,i,i==0) for c in [512,1024,2048,4096,8192] for i in range(6))
 normal=read(b/"request-diagnostics.json");remote=[x for x in normal["samples"] if x["tier"]=="remote_ssd"]
 assert len(remote)==30
 rows={(r["context_tokens"],r["repeat"],r["warmup"]):r for r in remote}
 config,refs=configuration(role);direct=read(role/"config.json")["agent"]["direct_receive"]
 reference_rows={}
 if reference:
  reference_normal=read(reference/"remote-full/remote-only-normalized.json");reference_config=json.loads(json.dumps(reference_normal["matched_configuration"]));reference_config["native_build"]=NATIVE;assert reference_config==config
  reference_rows={(r["context_tokens"],r["repeat"],r["warmup"]):r for r in reference_normal["samples"]}
 details=[]
 for s in samples:
  ctx=s["context_tokens"];pages=(ctx-1)//64;expected=pages*16777216;key=(ctx,s["repeat"],s["warmup"])
  p=b/s["artifact"];n=rows[key];proof=s["remote_proof"];obs=proof["observed"]
  assert proof["exact"] and proof["direct_receive"]==direct and proof["expected_bytes"]==expected
  assert s["cache"]["device"]==s["cache"]["host"]==0 and s["cache"]["storage"]==pages*64
  for field in ["requester_remote_read_bytes","owner_posix_read_bytes","owner_ucx_write_bytes"]:assert obs[field]==expected
  for field in ["requester_posix_read_bytes","requester_ucx_write_bytes","requester_posix_write_bytes","owner_posix_write_bytes","owner_staging_copy_bytes"]:assert obs[field]==0
  assert obs["requester_staging_copy_bytes"]==(0 if direct else expected)
  if direct:
   assert obs["requester_direct_receive_bytes"]==expected and obs["requester_direct_receive_segments"]==pages*2
  reqs=[read(p/f) for f in ["request.json","measured-request.json","cold-control-request.json","owner-seed-request.json"]]
  ids=reqs[0]["input_ids"];params=reqs[0]["sampling_params"]
  assert len(ids)==ctx and all(q["input_ids"]==ids and q["sampling_params"]==params for q in reqs)
  outputs=[read(p/f) for f in ["measured.json","cold-control.json","owner-seed.json"]]
  assert all(o["output_ids"]==outputs[0]["output_ids"] and o["text"]==outputs[0]["text"] for o in outputs)
  assert s["output_sha256"]==hashlib.sha256(json.dumps(outputs[0]["output_ids"]).encode()).hexdigest()
  assert n["source_proof"]["metadata_io"]==proof["metadata_io"] and n["receiver_mode"]==("direct" if direct else "staged")
  assert n["request_id"]==outputs[0]["request_id"]==reqs[1]["rid"]
  n.update(input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),sampling_params_sha256=sha(params),input_ids_artifact=str(p/"measured-request.json"),input_ids_artifact_sha256=file_sha(p/"measured-request.json"),sampling_params_artifact=str(p/"measured-request.json"))
  if reference:
   ref=reference_rows[key]
   assert all(n[k]==ref[k] for k in ["input_ids_sha256","sampling_params_sha256","output_sha256"])
  assert n["request_start_ns"]==outputs[0]["request_start_ns"] and n["first_token_ns"]==outputs[0]["first_token_ns"]
  timeline=n["trace_events"];assert timeline==read(p/"critical-path-raw.json")
  assert all(e["rid"]==n["request_id"] for e in timeline)
  assert [b["events"] for b in n["native_batches"]]==[e["events"] for e in timeline if e["stage"]=="native_batch"]
  flat=[dict(e,evidence="native:%s:%s"%(b["batch_handle"],i)) for b in n["native_batches"] for i,e in enumerate(b["events"])]
  assert flat==n["native_events"]
  forward=next(e["start_ns"] for e in timeline if e["stage"]=="first_forward_entry")
  assert all(e["build_marker"]==NATIVE[:12] for e in timeline if e["stage"]=="native_batch")
  events={}
  for batch in n["native_batches"]:
   assert batch["rid"]==n["request_id"]
   for i,e in enumerate(batch["events"]):
    ref="native:%s:%s"%(batch["batch_handle"],i);events[ref]=e
    assert n["request_start_ns"]<=e["start_ns"]<=e["end_ns"]<=forward
  rpc=[e for e in events.values() if e["stage"]=="remote_rpc"]
  copy=[e for e in events.values() if e["stage"]=="staging_copy"]
  assert sum(e["bytes"] for e in rpc)==expected and sum(e["owner_read_bytes"] for e in rpc)==expected
  assert sum(e["object_count"] for e in rpc)==pages
  assert sum(e["bytes"] for e in copy)==(0 if direct else expected)
  for e in rpc:
   assert e["owner_metadata_bytes"]==e["object_count"]*4096 and e["owner_metadata_ns"]>0
   assert e["owner_staging_copy_ns"]==e["owner_staging_copy_bytes"]==0
   assert e["owner_posix_ns"]>0 and e["owner_ucx_ns"]>0
   assert sum(e[k] for k in ["owner_posix_ns","owner_ucx_ns","owner_metadata_ns","owner_staging_copy_ns"])<=e["end_ns"]-e["start_ns"]
   if direct:assert e["direct_receive"] and e["destination_segments"]==e["object_count"]*2
  for who,pre,post in [("requester","before-stats.json","after-stats.json"),("owner","owner-before-stats.json","owner-after-stats.json")]:
   before=read(p/pre)["metrics"];after=read(p/post)["metrics"]
   for comp in ["metadata_read","metadata_write"]:
    delta=sum(v-before.get(k,0) for k,v in after.items() if k.startswith("sglang:nixlshard_component_bytes_total") and ('component="'+comp+'"') in k)
    assert delta==proof["metadata_io"][who]["observed"][comp+"_bytes"],(s["artifact"],who,comp,delta)
   for comp,field in [("posix_read","posix_read_bytes"),("posix_write","posix_write_bytes"),("ucx_write","ucx_write_bytes"),("staging_copy","staging_copy_bytes"),("remote_read","remote_read_bytes"),("direct_receive","direct_receive_bytes")]:
    dest=who+"_"+field
    if dest in obs:
     delta=sum(v-before.get(k,0) for k,v in after.items() if k.startswith("sglang:nixlshard_component_bytes_total") and ('component="'+comp+'"') in k)
     assert delta==obs[dest],(s["artifact"],dest,delta,obs[dest])
  metadata=proof["metadata_io"]
  for who in ["owner","requester"]:
   m=metadata[who];assert m["available"] and all(v>=0 for v in m["observed"].values())
  assert metadata["requester"]["observed"]["metadata_read_bytes"]==metadata["requester"]["observed"]["metadata_write_bytes"]==0
  om=metadata["owner"]["observed"];assert om["metadata_read_bytes"]>0 and sum(om.values())>=sum(e["owner_metadata_bytes"] for e in rpc)
  blocks=n["critical_blocks"];cursor=n["request_start_ns"];other=0
  for block in blocks:
   assert block["start_ns"]==cursor and block["end_ns"]>=cursor;cursor=block["end_ns"]
   if block["category"]=="other":other+=block["end_ns"]-block["start_ns"]
   if block["category"]=="remote_rpc":
    e=events[block["evidence"][0]]
    assert (block["start_ns"],block["end_ns"])==(e["start_ns"],e["end_ns"])
    assert block["composition_only"] and all(block[k]==e[k] for k in ["owner_metadata_ns","owner_metadata_bytes","owner_posix_ns","owner_ucx_ns","owner_staging_copy_ns","owner_staging_copy_bytes"])
  assert cursor==n["first_token_ns"] and not n["unresolved_overlaps"]
  assert len([x for x in blocks if x["category"]=="remote_rpc"])==len(rpc)
  assert abs((cursor-n["request_start_ns"])/1e6-s["ttft_ms"])<1e-8
  for w in n["service_windows"]:
   e=events[w["evidence"][0]];stage=w["stage"]
   expected_service={"ssd":(e["owner_read_bytes"],e["owner_posix_ns"]),"ucx":(e["bytes"],e["owner_ucx_ns"]),"metadata":(e["owner_metadata_bytes"],e["owner_metadata_ns"]),"staging_copy":(e["owner_staging_copy_bytes"],e["owner_staging_copy_ns"])}
   assert (w["bytes"],w["duration_ns"])==expected_service[stage]
  details.append({"context_tokens":ctx,"repeat":s["repeat"],"warmup":s["warmup"],"ttft_ms":s["ttft_ms"],"other_percent":100*other/(cursor-n["request_start_ns"]),"expected_payload_bytes":expected,"metadata_bytes":sum(e["owner_metadata_bytes"] for e in rpc),"owner_metadata_ms":sum(e["owner_metadata_ns"] for e in rpc)/1e6,"owner_payload_ms":sum(e["owner_posix_ns"] for e in rpc)/1e6,"owner_ucx_ms":sum(e["owner_ucx_ns"] for e in rpc)/1e6,"receiver_copy_ms":sum(e["end_ns"]-e["start_ns"] for e in copy)/1e6,"remote_groups":len(rpc),"input_ids_sha256":n["input_ids_sha256"],"sampling_params_sha256":n["sampling_params_sha256"],"output_sha256":n["output_sha256"]})
 normalized={k:v for k,v in normal.items() if k!="samples"};normalized.update(samples=remote,comparison_cohort="20261005-a10-b7-authoritative-g3",matched_configuration=config,source_study_id=normal["study_id"],source_artifact=str(b/"request-diagnostics.json"),source_artifact_sha256=file_sha(b/"request-diagnostics.json"),derivation="remote-only; extra cold controls excluded; exact archived input/sampling fingerprints added; producer raw unchanged")
 out=b/"remote-only-normalized.json";out.write_text(json.dumps(normalized,indent=2)+"\n")
 groups={}
 for c in [512,1024,2048,4096,8192]:
  d=[x for x in details if x["context_tokens"]==c and not x["warmup"]]
  groups[str(c)]={k:statistics.median(x[k] for x in d) for k in ["ttft_ms","owner_metadata_ms","owner_payload_ms","owner_ucx_ms","receiver_copy_ms","other_percent"]}
 audit={"passed":True,"samples":30,"measured":25,"receiver_mode":"direct" if direct else "staged","storage_contract":"authoritative_g3_v2","comparison_cohort":normalized["comparison_cohort"],"source_pins":{"native":NATIVE,"sglang":SG,"helper":TOOL},"runtime_domain_maps_verified":True,"saved_before_after_counter_deltas_verified":True,"runtime_witnesses":refs,"matched_configuration":config,"matched_configuration_sha256":sha(config),"strict_replay_verified":bool(reference),"directional_metadata_scope":"full-generation/export settled windows; not additive TTFT; per-RID metadata child is request trace","child_timing_scope":"owner durations nested serially in requester whole RPC; requester-clock wall partition conserved, no cross-host clocks","normalized_sha256":file_sha(out),"raw_samples_sha256":file_sha(b/"samples.jsonl"),"maximum_other_percent":max(x["other_percent"] for x in details),"median_measured":groups,"rows":details}
 ap=b/"independent-audit.json";ap.write_text(json.dumps(audit,indent=2)+"\n")
 print(json.dumps({"role":str(role),"audit_sha256":file_sha(ap),"normalized_sha256":file_sha(out),"max_other":audit["maximum_other_percent"],"medians":groups}))
if __name__=="__main__":audit(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None)
