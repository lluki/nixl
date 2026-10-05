# Exact-input remote TTFT replay

For a new authoritative-G3 cohort, pass `--base-harness-sha256` with the approved
SHA256 of that cohort's `bench-ttft.py` source. The default retains the historical
`fcacef2` harness pin. Each invocation checks source bytes before importing or
sending HTTP and records the selected SHA separately from native/serving pins.
Keep the staged reference and direct replay within one matching implementation
cohort; historical results are not a control for the new SSD record format.

**Agent written**

The ordinary TTFT tool chooses a new UUID prefix for every candidate. Reusing a
run name does not reproduce its token sequence. This driver reuses all archived
staged candidate token IDs and sampling parameters through the existing
`RemoteExperiment.sample(ids=...)` hook. The receiver comparison can then use
identical input and output token sequences.

The driver requires the exact byte-pinned base tool from NIXL source
`fcacef2d331dec6a0f00457b5563849d9e7f007d` (SHA256
`db8263a1bcf463759b5ee4d83436adb80a6e005b2adf53708dbf3109210403f6`).
Native implementation `da3a22a` compiled with source marker `fcacef2d331d`, SGLang
`336fc956`, the base tool and sibling `ttft_trace.py` stay unchanged. Copy only
this new driver from its later Git commit into the serving sandbox; do not
rebuild or change serving runtime pins for this sampling fix.

Before traffic, the driver checks every expected context/repetition/warmup key,
candidate uniqueness, complete request lengths, exact sampling parameters,
staged pure-storage/byte proofs, and agreement of cold/owner/measured requests
and outputs. It records the full synthetic replay plan and input/source hashes.
The existing base harness still measures streaming TTFT and enforces native
bytes, positive direct scatter traces, cache masks, and output correctness. The
driver additionally compares each direct output to its archived staged output.

Start a fresh owner SSD and metadata service and flush the diskless requester.
Keep the owner's `direct_receive=false` in both modes; only the requester changes
to `true`. Reusing populated metadata could make the replay's cold controls hit
an earlier staged key. Both runs must use the same physical GPUs, model revision,
native/serving pins, pool geometry, UCX transport and serving flags.

```bash
python /workspace/bench-ttft-replay.py \
  --base-harness /workspace/bench-ttft.py \
  --reference-artifact-dir /scratch/.../remote-staged/requester/remote-full \
  --validate-only -- \
  --contexts 512 1024 2048 4096 8192 --warmups 1 --repeats 5 \
  --output-tokens 16 --page-size 64 --scenarios remote --direct-receive
```

For execution remove `--validate-only` and pass all the original base harness
arguments after `--`, changing only the fresh artifact/GCS/provenance/trace paths
and adding `--direct-receive`. Those arguments include both endpoints, frozen
model revision, local auth key **path**, boot ID, exact page bytes, and native
trace/payload proof options. Launch with `nohup` and an overall watchdog, as in
`bench-ttft.md`; a short MCP command must not own the long-running measurement.

Store source only in Git and the authorized persistent RAID backup. Approved GCS
archives contain synthetic requests, JSON/log data, launch configurations and
SHA manifests; they must never contain this reproduction script or key bytes.
The metadata `replay-plan.json` is allowed synthetic input data, not source.

