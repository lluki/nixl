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
Percentiles use linear interpolation.

For ephemeral cluster pods, add `--gcs-prefix gs://dynamo-gcp-dev-02-nixl-object-perf-526392861238/runs/nixlshard-v2/UNIQUE-RUN`.
With `google-cloud-storage` installed and ADC credentials supplied by the runtime,
the harness uploads an initial provenance archive, one ZIP for each completed
sample (including failed cache checks), and a final summary archive. Each archive
is immutable (`if_generation_match=0`); a reused prefix fails rather than replacing
another run. Archives include raw SSE events and metrics, with hashes recorded in
`gcs-archives.json`. Uploads happen outside timed generation and failures fail the
run. In-flight samples can still be lost if the pod disappears before completion.

Prometheus snapshots/deltas record available framework cache and staging
histograms. SGLang source `d9115436cdda18801c9381f7ee6cc46edb8b1a95` additionally
provides a backend-owned native counter exporter. Add top-level
`"export_native_metrics": true` to storage extra configuration and launch with
`--enable-metrics`; both gates are required, and native export defaults off.

The fixed `sglang:nixlshard_component_seconds_total`,
`sglang:nixlshard_component_bytes_total`, and `sglang:nixlshard_events_total`
families expose seven known native timer components in seconds, five byte
components, and fixed status/resource events. They exclude current slot gauges
and arbitrary Agent fields. Timers aggregate concurrent workers/background
work and overlap, including owner I/O inside remote control latency. Exported
totals update when SGLang collects backend stats, rather than querying the
Agent from HTTP.

`runtime.json.native_component_counters` records the families actually
observed across the run, including whether all three fixed families appeared.
Per-sample snapshots record availability at that collection. The initial
metrics scrape is saved too, so counters appearing after initial collection
are reflected in the final runtime record. The separate optional
`--native-stats-url` records a caller-supplied diagnostic path and whether a
snapshot was received; it can provide full Agent fields if such an external
diagnostic facility exists. It is not needed for the fixed Prometheus families,
and this harness adds no server endpoint.

Record baseline and explicitly instrumented profiles separately, with their
exact SGLang source hash, loaded native build marker, resolved export setting
and hardware/storage provenance. Export and metric observations have their
own cost. These aggregate intervals exclude Python/controller/model work and
must not be summed or subtracted into TTFT or attributable NIXLShard overhead.

The workload uses coherent repeated text truncated to exact token-ID context
lengths, plus a unique leading nonce. It has deterministic temperature-zero
fixed-length output and is a synthetic cache-hit workload. Counter stability
does not prove zero background I/O. End-to-end cache-tier differences do not
isolate attributable NIXLShard overhead, and component counters must not be
summed/subtracted into that claim. HiCache `direct` is GPU/host movement;
SSD O_DIRECT is the separate `agent.direct_io` storage setting.

Long runs can exceed the execution tool's foreground timeout. Launch the same
command in the background, keeping its log outside the new artifact directory,
and save its PID. For example:

```bash
nohup python extensions/nixlshard/tools/bench-ttft.py \
  --tokenizer-path /scratch/qwen32b \
  --model-revision c2d5a15ede2407bd2d2e6705851db3578777fed3 \
  --provenance-json /scratch/ttft-provenance.json \
  --artifact-dir /scratch/ttft-UNIQUE-RUN \
  --contexts 512 2048 4096 --warmups 1 --repeats 5 \
  --scenarios cold ssd gpu \
  --gcs-prefix gs://dynamo-gcp-dev-02-nixl-object-perf-526392861238/runs/nixlshard-v2/UNIQUE-RUN \
  > /scratch/ttft-UNIQUE-RUN.log 2>&1 < /dev/null &
echo $! > /scratch/ttft-UNIQUE-RUN.pid
```

Poll the log and completed sample artifacts; avoid issuing concurrent generation
requests. Background execution does not protect against pod deletion, so retain
the incremental GCS archives.

Run the GPU-independent parser, timing and cache-provenance checks with:

```bash
python -m unittest discover -s extensions/nixlshard/tests -p test_bench_ttft.py -v
```
