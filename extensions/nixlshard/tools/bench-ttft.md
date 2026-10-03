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

## Actual remote storage-hit profile

Use `--scenarios remote` with two dedicated SGLang endpoints. Owner B has the
assigned SSD/debug file; requester A uses `agent.disks: []`, which prevents
requester writeback from turning later reads into local SSD hits. Both use the
same immutable model revision, model path, rank topology, BF16 KV/page layout,
native metadata server/peer records, and opt-in native metrics exporter. Record
transport selection and NIC/host placement in both launch provenance files;
UCX byte counters alone do not distinguish RC/RDMA from another UCX transport.
Both provenance files must state `kv_bytes_per_page` matching the configured
logical page size (16777216 for this Qwen TP1 profile).

For each fresh prefix, A first generates a matched cold control, with zero
cache hits and zero remote/SSD payload reads. B then generates the identical
token IDs and completes its full eligible backup. The two cold outputs must
match. The harness flushes A before bounded discovery preflights because native
MD key/peer discovery is asynchronous. These preparation generations are
journaled and excluded from measured TTFT; A's absent SSD prevents local-cache
admission. Once a complete remote hit has been proven, A is flushed again and
the measured streaming request replays exactly those IDs. Its output must match
both cold references, with zero GPU/host hits and the exact eligible storage
prefix.
Read-only observations and upper-tier cache flushes reconnect and retry once
when an idle HTTP keepalive connection has closed. Generation requests are
never retried, so a connection failure cannot silently duplicate model work.

```bash
python extensions/nixlshard/tools/bench-ttft.py \
  --base-url http://REQUESTER:31001 --owner-base-url http://OWNER:31001 \
  --tokenizer-path /scratch/qwen32b \
  --model-revision c2d5a15ede2407bd2d2e6705851db3578777fed3 \
  --provenance-json /scratch/requester-launch.json \
  --owner-provenance-json /scratch/owner-launch.json \
  --artifact-dir /scratch/remote-UNIQUE-RUN \
  --scenarios remote --contexts 512 1024 2048 4096 8192 \
  --warmups 1 --repeats 10 --kv-bytes-per-page 16777216 \
  --gcs-prefix gs://dynamo-gcp-dev-02-nixl-object-perf-526392861238/runs/nixlshard-v2/remote-UNIQUE-RUN
```

`remote_proof` requires exact owner POSIX-read/UCX-write bytes and exact
requester remote-read/staging bytes. Each equals
`((context_tokens-1)//page_size)*kv_bytes_per_page`. Requester POSIX reads,
requester UCX writes, either endpoint's POSIX writes, and owner staging copies
must be zero in that window. Missing counters, extra transfers, partial/mixed
hits, output mismatches, and unresolved discovery fail with raw evidence. The
harness polls only metrics after generation to allow scheduler exports to catch
up; it never sends owner generations during the measurement/counter window.
If an idle owner cannot expose current counters, the proof fails rather than
accepting stale zeros. No server diagnostics route is added by this harness.

Each remote sample preserves `before`/`after` and `owner_before`/`owner_after`
raw metrics, timer/byte/event deltas, discovery attempts, and SSE results.
`cold_control_ttft_ms`, `cold_control_cache`, and `cold_control_metric_deltas`
expose the matched control; `cold-control.json` preserves its complete stream.
Summary groups use `CONTEXT:remote` and `CONTEXT:cold-control`, with warmups
excluded from both. Intermediate immutable archives preserve the owner seed and
discovery-ready phases before the final sample archive. Uploads and discovery
change preparation pacing and must use identical flags across comparisons.
Native counter windows extend through full generation and export settling;
their overlapping timers do not isolate the first-token critical path or
additive NIXLShard overhead. A no-hit cold control still has the backend enabled;
it is not a backend-disabled overhead experiment.

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

## Five tiers with request timelines

Run the four-tier profile on a requester with a fresh assigned SSD file:
`--scenarios cold gpu host ssd --contexts 512 1024 2048 4096 8192`,
`--warmups 1 --repeats 5 --gpu-capacity-tokens 16384`,
`--host-pressure-tokens 18432 --pressure-context 2048`. Its 16 GB host cache
must retain the target while unique preparation prompts evict GPU pages.
Host preparation waits for durable seed backup; a diskless requester cannot
satisfy that condition. Then run the exclusive remote profile on the same
requester with `agent.disks=[]` and the same model, pools and other flags.
Disclose the assigned-disk versus diskless profile difference. Remote samples
also retain their matched diskless cold controls.

Add `--verify-native-payload --kv-bytes-per-page 16777216`. Local SSD reads
must equal the eligible prefix bytes, with no remote reads or UCX writes.
GPU/host/cold requests must have zero native payload reads. Aggregate staging
bytes combine reads and background stores, so exact get-copy coverage uses
the per-request trace. These counters establish route/amount, not additive
first-token time.

For diagnostics, launch trace-capable SGLang with
`SGLANG_REQUEST_TIMELINE_DIR=/scratch/UNIQUE-RUN/request-timeline` and native
`agent.enable_trace=true` (native `1a8ecc107a1e` or newer on both endpoints).
Run the client on the requester host, passing `--request-trace-dir` and
`--request-trace-boot-id` equal to that host's recorded kernel boot ID.
Each generation has an explicit unique serving request ID; first-token and
request-submission timestamps use CLOCK_MONOTONIC. Stream generations are
never automatically retried.

Each sample preserves raw selected serving events before validation and a
`critical-path.json` normalized record. `request-diagnostics.json` supplies
schema version, clock, study ID, contexts, samples and an initially empty
bandwidth-reference list for the report collector. Native batches retain
request IDs/handles/events; evidence references use `sg:<index>` and
`native:<handle>:<event-index>`. Successful remote RPCs require measured
owner POSIX and UCX durations, exact RPC/copy payload bytes, and correct output.

The wall partition preserves one framework/H2D/forward envelope. Additional
actual-anchor envelopes measure API-to-scheduler dispatch, terminal native
capture through cache/ACK/scheduler handoff, and prefill-result-to-API output
delivery. These include framework processing and waits, not CPU-exclusive
execution; evidence references identify both boundaries. Payload
intervals take precedence over metadata and queue occupancy; unknown gaps
remain other. Owner durations compose a whole RPC only, with no invented
cross-host timestamps. Intervals crossing first-forward remain explicit raw
overlaps inside the combined envelope and cannot supply a complete owner
stack. Means of per-request exclusive blocks sum to mean TTFT; independent
component medians generally do not sum to p50. Native service-window rates
are separate from first-token attribution and use descriptor-count patterns
for matching external fio/UCX references.

Timeline serialization/flushing adds instrumentation cost and is included in
the new measured curves. Record source/build/boot and both launch profiles;
do not graft older uninstrumented curves onto this campaign.
