# Native saturated-cache batch benchmark

**Agent written**

This benchmark fills a newly created disposable file, then stores and reads unique
immutable keys in batches that force reclamation. It measures one native batch
outstanding, independently of SGLang, TTFT, or the GB200 serving profile.

Defaults match the planned saturation comparison: 32 GiB total file capacity,
64 MiB metadata, 64 KiB allocation units, 16 MiB values, two workers, four
32 MiB staging slots, a 5 s native deadline, and five batches each of 32/64 values.
The exact usable capacity is calculated from these parameters: 2044 values.
Initial fill must succeed completely before trials are labeled saturated.

Activate the selected native installation and run from the NIXL source tree:

```sh
source /workspace/install/nixlshard/env.sh
python3 extensions/nixlshard/tools/bench-saturation.py \
  --directory /raid/nixlshard-v2/saturation
```

For long runs, launch in the background to avoid execution-tool timeouts:

```sh
nohup python3 extensions/nixlshard/tools/bench-saturation.py \
  --directory /raid/nixlshard-v2/saturation \
  > /workspace/logs/saturation.log 2>&1 &
```

Each invocation prints a unique artifact directory. It contains the compiled
native marker, loaded package path, tool SHA256, complete configuration, raw
per-key statuses, per-batch latencies and counter deltas, successful-read hashes,
explicit checkpoint evidence, and before/after-restart existence mappings.
Checkpoint contention reported as `busy` is retried within the operation
watchdog; each attempt and its elapsed time is preserved.
Only one aligned RAM block is registered, sized for the largest batch (1 GiB
by default). Its contents remain unchanged through terminal polling and release.
The same block is then overwritten for reads; every successful load must match
the SHA256 of that key's distinct content. Generation and hash checking occur
outside the timed native batch.
Submission, observed terminal statuses, and checkpoint attempts are journaled
before potentially blocking native cleanup, preserving active-phase evidence
when the hard watchdog fires. Batch `elapsed_ns` includes submission/polling
through terminal status plus the release call; the intervening flushed-terminal
journal gap is reported separately. `wall_elapsed_ns` includes that gap.

Non-success statuses are reported in the raw artifacts and summary. A completed
correctness check can coexist with failed cache stores or reads; inspect
`all_measured_stores_success`, `all_measured_loads_success`, and status counts.
A successful cache miss is never counted as a verified value. Initial fill
failures abort the experiment. Retained keys must remain retrievable after an
explicit checkpoint and orderly restart, while evicted old keys may miss.

The native deadline is separate from a 30 s operation watchdog and 1800 s hard
overall watchdog. A supervisor process terminates a stuck worker, including one
blocked in native release/close, and records an incomplete run. Debug files are
removed after orderly completion; `--keep-mock-file` preserves them. A hard
watchdog retains the unique debug file for inspection.

Optional `--gcs-prefix` uses the existing TTFT archive helper with ADC loaded
only when uploads are enabled. Use the explicitly approved destination, such as
`gs://dynamo-gcp-dev-02-nixl-object-perf-526392861238/runs/nixlshard-v2/saturation`.
The helper adds the unique local run name. After each fill chunk and each
measured store/read/verification iteration it uploads an immutable ZIP with
`if_generation_match=0`; the manifest records each archive's SHA256, size, and
URI. An explicit allowlist copies provenance, raw JSONL, summary, and archive
manifest only. SSD files and source/destination buffers are excluded. The final
summary, including hard-watchdog failure evidence, is uploaded by a separate
process bounded by the operation watchdog. Upload failures remain visible
locally and yield a nonzero exit status.

Each metadata-copy/ZIP/upload gap is logged outside native batch timing, and
the selected GCS flags are recorded in provenance. These gaps change workload
pacing and allow maintenance work between batches; compare identical upload
flags across native prefixes. The final upload's own duration is retained
locally after its immutable archive was formed.

Summary `percentile_method` identifies p95 as **nearest rank**; `median_ms` uses
Python's statistical median. The TTFT harness uses linear interpolation, so
small-sample p95 values from the two harnesses follow different definitions.

A small smoke test exercises serial fill, eager eviction, distinct-content
verification, checkpoint, and restart without allocating the full default size:

```sh
python3 extensions/nixlshard/tools/bench-saturation.py \
  --capacity-bytes 393216 --metadata-bytes 65536 --unit-bytes 4096 \
  --payload-bytes 65536 --staging-slot-bytes 131072 \
  --fill-batch-pages 1 --batch-pages 1,2 --iterations 3 \
  --overall-watchdog-seconds 30
```

This geometry has five value slots; nine measured stores force nine evictions.
The original native `baf888f17220` passed this smoke, verifying 16 successful
reads including two retained keys after restart, with both oldest keys missing.
That verifies the harness at small geometry and does not predict the default
large-metadata saturation outcome.

Use identical tool/configuration/filesystem parameters and separate artifact
directories to compare compiled `d9161cf` against a retry fix. Both versions must
first be available in isolated prefixes. Record their compiled markers rather
than inferring build provenance from a checkout. No A2/B2 benchmark has been
run by this tool yet.

Native elapsed-time counters sum worker/background intervals and may overlap.
They cannot be summed into batch wall time or attributed to model TTFT.
The default buffered debug-file route includes filesystem/page-cache effects;
`--direct-io` selects SSD payload O_DIRECT independently of SGLang's GPU/host
HiCache direct path. This benchmark establishes native saturation behavior,
not the root cause of the earlier serving run.
