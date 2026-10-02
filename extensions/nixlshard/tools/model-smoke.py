#!/usr/bin/env python3
"""Launch an SGLang GPU server and prove an SSD hit after upper-tier flush.

Run with the matching SGLang Python environment and source/native PYTHONPATH.
This is a correctness smoke, not a TTFT or throughput benchmark.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def request(base, path, payload=None, timeout=90):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def selected_metrics(metrics):
    return [line for line in metrics.splitlines()
            if not line.startswith("#") and any(name in line for name in (
                "backuped_tokens_total", "storage_prefetch_hit_tokens_total",
                'cache_source="storage"', 'mode="storage_hit"'))]


def wait_ready(process, base, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}; see server.log")
        try:
            request(base, "/health", timeout=2)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError("server readiness deadline exceeded; see server.log")


def wait_backup(base, artifact_dir, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metrics = request(base, "/metrics", timeout=5)
        artifact_dir.joinpath("warm-metrics.txt").write_text(metrics)
        backed_up = sum(float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines()
                        if line.startswith("sglang:backuped_tokens_total{"))
        if backed_up >= 64:
            return metrics
        time.sleep(0.5)
    raise TimeoutError("no completed storage backup reported before flush")


def stop_server(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # A terminated launcher can leave multiprocessing children in its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True,
                        help="immutable revision or frozen weights/tokenizer/config manifest digest")
    parser.add_argument("--artifact-dir", type=Path,
                        help="new, unique directory; defaults below /raid/nixlshard-v2/model-smoke")
    parser.add_argument("--port", type=int, default=31001)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir or Path("/raid/nixlshard-v2/model-smoke") / (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = artifact_dir.resolve()
    config = {
        "model_revision": args.model_revision,
        "prefetch_threshold": 64,
        "agent": {
            "name": "model-smoke-" + uuid.uuid4().hex[:12],
            "disks": [{"path": str(artifact_dir / "cache.bin"),
                       "capacity_bytes": 1 << 30, "unit_bytes": 1 << 16,
                       "metadata_bytes": 16 << 20, "create": True}],
            "listen_host": "127.0.0.1", "listen_port": 0,
            "staging_slot_bytes": 1 << 20, "staging_slots": 4,
            "workers": 2, "max_inflight": 64, "timeout_ms": 5000,
            "direct_io": False,
        },
    }
    artifact_dir.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", args.model_path, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "bfloat16", "--tp-size", "1", "--page-size", "64",
               "--max-total-tokens", "4096", "--context-length", "4096",
               "--mem-fraction-static", "0.2", "--disable-cuda-graph",
               "--attention-backend", "triton", "--enable-hierarchical-cache",
               "--hicache-size", "1", "--hicache-mem-layout", "page_first_direct",
               "--hicache-io-backend", "direct", "--hicache-write-policy", "write_through",
               "--hicache-storage-backend", "nixlshard",
               "--hicache-storage-backend-extra-config", "@" + str(artifact_dir / "config.json"),
               "--hicache-storage-prefetch-policy", "wait_complete", "--enable-metrics"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                       UCX_TLS="tcp,self,cuda_copy", TMPDIR=str(artifact_dir),
                       SGLANG_CACHE_DIR=str(artifact_dir / "sglang-cache"),
                       TRITON_CACHE_DIR=str(artifact_dir / "triton-cache"))
    import nixlshard
    import torch
    packages = ["torch", "torchvision", "torchaudio", "sglang-kernel", "flashinfer-python",
                "flashinfer-cubin", "flashinfer-jit-cache", "transformers", "tokenizers",
                "xgrammar", "compressed-tensors", "cuda-python", "cuda-bindings",
                "cuda-core", "cuda-tile", "apache-tvm-ffi", "nvidia-cutlass-dsl",
                "nvidia-cudnn-frontend", "nccl4py"]
    runtime = {"python": sys.version, "executable": sys.executable,
               "native_build": nixlshard.__build_marker__, "torch_path": torch.__file__,
               "packages": {name: importlib.metadata.version(name) for name in packages},
               "command": command, "artifact_dir": str(artifact_dir)}
    artifact_dir.joinpath("runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    base = "http://127.0.0.1:" + str(args.port)
    result = {}
    with artifact_dir.joinpath("server.log").open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        runtime["pid"] = process.pid
        artifact_dir.joinpath("runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        try:
            wait_ready(process, base, args.startup_timeout)
            prompt = ("NIXLShard integration validation: the cache holds key value tensors "
                      "for a deterministic prefix. " * 40) + "Write one short sentence about cache correctness."
            payload = {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 16}}
            artifact_dir.joinpath("request.json").write_text(json.dumps(payload, indent=2) + "\n")
            result["warm"] = request(base, "/generate", payload)
            warm_metrics = wait_backup(base, artifact_dir)
            result["warm_metrics"] = selected_metrics(warm_metrics)
            result["flush"] = request(base, "/flush_cache", {})
            if "Cache flushed" not in str(result["flush"]):
                raise RuntimeError("upper-tier flush did not complete: " + str(result["flush"]))
            result["replay"] = request(base, "/generate", payload)
            replay_metrics = request(base, "/metrics")
            artifact_dir.joinpath("replay-metrics.txt").write_text(replay_metrics)
            result["replay_metrics"] = selected_metrics(replay_metrics)
            details = result["replay"]["meta_info"].get("cached_tokens_details") or {}
            if details.get("storage", 0) <= 0 or details.get("device", 0) or details.get("host", 0):
                raise AssertionError("no isolated storage hit after upper-tier flush: " + str(details))
            if result["warm"]["output_ids"] != result["replay"]["output_ids"]:
                raise AssertionError("deterministic generated token IDs differ after storage reload")
            result["passed"] = True
            print(json.dumps({"passed": True, "storage_tokens": details["storage"],
                              "artifact_dir": str(artifact_dir), "native_build": runtime["native_build"]}))
        except BaseException as error:
            result["passed"] = False
            result["error"] = repr(error)
            raise
        finally:
            artifact_dir.joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
            stop_server(process)


if __name__ == "__main__":
    main()
