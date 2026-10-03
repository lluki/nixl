#!/usr/bin/env python3
"""Run separate context epochs with idle-only model-preserving storage resets."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mock-file", type=Path, required=True)
    p.add_argument("--admin-key-file", type=Path, required=True)
    p.add_argument("--gcs-prefix", required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[512,1024,2048,4096,8192])
    args=p.parse_args()
    root=args.root.resolve()
    control=root/"epoch-supervisor"
    control.mkdir(exist_ok=False)
    config=json.loads(args.config.read_text())
    key=args.admin_key_file.read_text().strip()
    if not key: raise ValueError("empty admin key file")
    mock=args.mock_file.resolve()
    if args.mock_file.absolute()!=mock:
        raise ValueError("reset file path cannot contain symlink parents or traversal")
    assigned=config["agent"]["disks"]
    if len(assigned)!=1 or Path(assigned[0]["path"]).absolute()!=mock:
        raise ValueError("reset path must exactly match the sole assigned debug file")
    if root not in mock.parents or mock.is_symlink():
        raise ValueError("disposable file must be inside this run and not a symlink")
    if not assigned[0].get("create"):
        raise ValueError("fresh-file creation must be enabled")
    spec=importlib.util.spec_from_file_location("bench","/workspace/bench-ttft.py")
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    archives=bench.GcsArchives(args.gcs_prefix+"/supervisor",control)
    import hashlib
    (control/"source-manifest.json").write_text(json.dumps({"path":str(Path(__file__).resolve()),"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+"\n")
    def journal(phase,**fields):
        entry={"phase":phase,"monotonic_ns":time.monotonic_ns(),**fields}
        with (control/"events.jsonl").open("a") as f:
            f.write(json.dumps(entry)+"\n");f.flush()
    def snapshot(name):
        for path in control.iterdir():
            if not path.is_file() or path.suffix not in (".json", ".jsonl"):
                raise ValueError("control archive accepts only JSON data and journals")
            if key.encode() in path.read_bytes():
                raise ValueError("credential value in control artifact")
        archives.archive(control,name,recursive=False)
    def request(method,path,payload=None):
        journal("control_started",method=method,path=path)
        body=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request("http://127.0.0.1:31001"+path,data=body,method=method,
             headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=90) as response:
                text=response.read(1<<20).decode().replace(key,"<redacted>")
                journal("control_returned",method=method,path=path,status=response.status,response=text)
        except urllib.error.HTTPError as e:
            journal("control_failed",method=method,path=path,status=e.code)
            raise RuntimeError("control HTTP "+str(e.code)) from None
    def open_fds():
        found=[]
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit(): continue
            try: fds=list((proc/"fd").iterdir())
            except FileNotFoundError: continue
            for fd in fds:
                try: target=os.readlink(fd)
                except FileNotFoundError: continue
                if target in (str(mock),str(mock)+" (deleted)"):
                    found.append({"pid":int(proc.name),"fd":fd.name})
        return found
    command_template=[
       sys.executable,"/workspace/bench-ttft.py","--base-url","http://127.0.0.1:31001",
       "--admin-key-file",str(args.admin_key_file),"--tokenizer-path","/scratch/qwen32b",
       "--model-revision","c2d5a15ede2407bd2d2e6705851db3578777fed3",
       "--provenance-json",str(root/"launch.json"),"--scenarios","cold","gpu","host","ssd",
       "--warmups","1","--repeats","5","--gpu-capacity-tokens","16384",
       "--host-pressure-tokens","18432","--pressure-context","2048",
       "--verify-native-payload","--kv-bytes-per-page","16777216",
       "--request-trace-dir",str(root/"request-timeline"),"--request-trace-boot-id",
       Path("/proc/sys/kernel/random/boot_id").read_text().strip()]
    (control/"command-template.json").write_text(json.dumps(command_template,indent=2)+"\n")
    journal("supervisor_started",contexts=args.contexts)
    try:
        for context in args.contexts:
            request("POST","/flush_cache",{})
            request("DELETE","/hicache/storage-backend")
            remaining=open_fds()
            journal("closed_file_descriptors",context=context,remaining=remaining)
            if remaining: raise RuntimeError("backend still holds disposable file descriptors")
            if mock.exists():
                if not mock.is_file() or mock.is_symlink():
                    raise ValueError("reset target is no longer a regular debug file")
                journal("unlink_started",context=context,file=str(mock),allocated_bytes=mock.stat().st_blocks*512)
                mock.unlink()
                journal("unlink_completed",context=context)
            request("PUT","/hicache/storage-backend",
                    {"hicache_storage_backend":"nixlshard",
                     "hicache_storage_backend_extra_config_json":json.dumps(config),
                     "hicache_storage_prefetch_policy":"wait_complete",
                     "hicache_write_policy":"write_through"})
            request("GET","/hicache/storage-backend")
            journal("epoch_ready",context=context)
            snapshot("context-"+str(context)+"-ready.zip")
            out=root/("context-"+str(context))
            cmd=command_template+["--contexts",str(context),"--artifact-dir",str(out),
                                 "--gcs-prefix",args.gcs_prefix+"/epochs"]
            (control/("command-"+str(context)+".json")).write_text(json.dumps(cmd,indent=2)+"\n")
            with (root/("context-"+str(context)+".log")).open("w") as log:
                journal("benchmark_started",context=context)
                done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=700)
            summary=json.loads((out/"summary.json").read_text())
            rows=[json.loads(line) for line in (out/"samples.jsonl").read_text().splitlines()]
            if done.returncode or not summary["passed"] or len(rows)!=24 or not all(r["passed"] for r in rows):
                raise RuntimeError("context epoch failed strict sample checks")
            journal("epoch_completed",context=context,returncode=done.returncode,samples=len(rows))
            snapshot("context-"+str(context)+"-complete.zip")
        journal("supervisor_completed",passed=True)
        snapshot("complete.zip")
    except BaseException as error:
        journal("supervisor_failed",error_type=type(error).__name__,error=str(error).replace(key,"<redacted>"))
        snapshot("failed.zip")
        raise

if __name__=="__main__": main()
