#!/usr/bin/env python3
"""Run the byte-frozen epoch supervisor with exact accepted local input replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

EPOCH = Path("/workspace/bench-tier-epochs.py")
WRAPPER = Path("/workspace/tools/replay-authoritative-g3.py")
REFERENCE = Path("/scratch/nixlshard-v2/authoritative-g3/20261005-a10-b7-local-direct/requester")
EPOCH_SHA = "4356200ae05432729030fd61faf32cc522e6d2f32d6f78703823847441728c1e"
WRAPPER_SHA = "e1855a942cca2122d33ebd7a5f4ffe835bb5d2e71d3612e6e5c77164dc312873"
PLAN_SHA = "883144a7d891c22b27a708e6cf13d93378730a0219eaad613470dac625469249"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if sha(EPOCH) != EPOCH_SHA or sha(WRAPPER) != WRAPPER_SHA:
        raise ValueError("frozen epoch or approved replay wrapper SHA mismatch")
    prefix = [sys.executable, str(WRAPPER), "--mode", "local",
              "--base-harness", "/workspace/bench-ttft.py",
              "--replay-helper", "/workspace/bench-ttft-replay.py",
              "--reference-artifact-dir", str(REFERENCE),
              "--expected-plan-sha256", PLAN_SHA]
    original_run = subprocess.run
    # Validate the entire accepted 120-sample plan before any supervisor API call.
    validated = original_run(prefix + ["--validate-only", "--", "--contexts",
                              "512", "1024", "2048", "4096", "8192",
                              "--scenarios", "cold", "gpu", "host", "ssd",
                              "--warmups", "1", "--repeats", "5"],
                             check=True, capture_output=True, text=True)
    if "--root" not in sys.argv:
        raise ValueError("explicit epoch artifact root required")
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    if not root.is_dir():
        raise ValueError("prepared epoch artifact root does not exist")
    provenance = dict(bootstrap_sha256=sha(Path(__file__)), epoch_sha256=EPOCH_SHA,
                      wrapper_sha256=WRAPPER_SHA, accepted_complete_plan_sha256=PLAN_SHA,
                      accepted_reference=str(REFERENCE),
                      validated_plan=json.loads(validated.stdout),
                      scope="Only the frozen epoch benchmark subprocess command is prefixed with exact-input replay; supervisor controls, FD proofs and tier preparation are unchanged.")
    (root / "replay-epoch-bootstrap.json").write_text(json.dumps(provenance, indent=2) + "\n")
    def replay_run(command, *args, **kwargs):
        if (isinstance(command, list) and len(command) > 1
                and command[1] == "/workspace/bench-ttft.py"):
            command = prefix + ["--"] + command[2:]
        return original_run(command, *args, **kwargs)
    spec = importlib.util.spec_from_file_location("frozen_epoch_supervisor", EPOCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run = replay_run
    try:
        module.main()
    finally:
        subprocess.run = original_run

if __name__ == "__main__":
    main()
