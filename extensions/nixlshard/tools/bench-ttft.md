**Agent written**

# Streaming cache-tier TTFT experiment

`bench-ttft.py` drives an already-running dedicated SGLang server, with one
generation request outstanding. It measures from HTTP submission to the first
received SSE event containing output token IDs. Empty metadata events do not
start the clock; final response time is recorded separately. Token events,
their client timestamps, generated IDs/text, cache-source details, metric
snapshots and every preparation step remain in raw artifacts.

Use the pinned source SGLang branch, `--enable-metrics`, `--stream-interval 1`,
TP1, BF16 KV, 64-token pages and write-through storage. For the configured
GB200 model, stage `Qwen/Qwen3-32B-FP8` revision
`c2d5a15ede2407bd2d2e6705851db3578777fed3` at `/scratch/qwen32b`; select
`--fp8-gemm-backend flashinfer_trtllm` and
`SGLANG_ENABLE_JIT_DEEPGEMM=0`. The server owner verifies the staged immutable
model snapshot and loaded native marker before measuring.

Supply a credential-free `provenance.json` with at least
`{"model_revision": "c2d5a15ede2407bd2d2e6705851db3578777fed3"}`.
Include the exact server command, resolved storage configuration, NIXL/SGLang
source hashes, loaded native marker/library paths, model manifest, hardware,
GPU/host cache budgets, and SSD path/device or debug-file details. The harness
also saves the live `/server_info` response and validates streaming interval,
page size, NIXLShard backend, metrics and GPU capacity.

For a server with `--max-total-tokens 16384`:

```bash
python extensions/nixlshard/tools/bench-ttft.py \
  --tokenizer-path /scratch/qwen32b \
  --model-revision c2d5a15ede2407bd2d2e6705851db3578777fed3 \
  --provenance-json /scratch/ttft-provenance.json \
  --artifact-dir /scratch/ttft-unique-run \
  --contexts 512 2048 8192 --warmups 2 --repeats 10 \
  --scenarios cold ssd gpu
```

Each cold request has a fresh nonce before its first complete cache page, and
flushes the device/host cache. GPU replay uses exactly the same prompt after
its backup counter settles, with no intervening generation. SSD measurement
seeds a fresh prompt, awaits the backup counter's increase and stability, then
flushes device/host cache while preserving storage. Cache-source fields must
show zero hits for cold, isolated storage hits for SSD, and isolated device hits
for GPU; positive hits must cover the eligible complete-page prefix. Cached
outputs must match their deterministic cold reference in both token IDs and
text. Warmup samples follow the same checks but are excluded from summaries.

The pinned HTTP API has no GPU-only eviction operation. To include a host
experiment, provide a pressure budget exceeding the configured GPU capacity,
while ensuring the server's host cache can retain the target. For example:

```text
--scenarios cold ssd host gpu --gpu-capacity-tokens 16384 \
--host-pressure-tokens 18432 --pressure-context 2048
```

Pressure prompts are unique, run sequentially, and are excluded from measured
TTFT. A sample only counts as host when the response reports sufficient host
hits and zero device/storage hits. Failed preparation/classification keeps
its raw response and fails the run; it is never silently renamed.

`summary.json` contains p50/p95 by context/cache source and the raw TTFT
samples; `samples.jsonl` and per-sample directories retain warmups and failures.
Percentiles use linear interpolation. Prometheus snapshots/deltas record the
available framework cache and staging histograms. The generic collector does
**not** expose adapter `native_*` component counters over HTTP; these are
explicitly marked unavailable unless a separately supplied diagnostic path
is passed with `--native-stats-url`.

The workload uses coherent repeated text truncated to exact token-ID context
lengths, plus a unique leading nonce. It has deterministic temperature-zero
fixed-length output and is a synthetic cache-hit workload. Counter stability
does not prove zero background I/O. End-to-end cache-tier differences do not
isolate attributable NIXLShard overhead, and component counters must not be
summed/subtracted into that claim. HiCache `direct` is GPU/host movement;
SSD O_DIRECT is the separate `agent.direct_io` storage setting.

Run the GPU-independent parser, timing and cache-provenance checks with:

```bash
python -m unittest discover -s extensions/nixlshard/tests -p test_bench_ttft.py -v
```
