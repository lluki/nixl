**Agent written**

The original supervisor at source commit `caa5d6ba02ca` preserves the exact
bytes executed in the
A6/C2 five-tier study: SHA256
`d5aaa4dbc40a28cb49528d769730dd01953aba0fc46977143a736e1dd1b169b1`.
That study used compiled native `1a8ecc107a1e`, SGLang `f43a9bafea3e`,
and benchmark harness `ce026d54dd73`. This later reproduction-tool commit does
not change those measured runtime pins. The current supervisor adds an
explicit direct-receive mode; its source-manifest SHA identifies the actual
executed bytes for new studies.

The supervisor runs one outstanding generation at a time for Qwen3-32B-FP8
revision `c2d5a15ede2407bd2d2e6705851db3578777fed3`, page64, 16MiB KV/page,
16384 GPU tokens, 18432 tokens of host pressure, and contexts512–8192.
Each context supplies one warmup plus five measured requests for each of
cold, GPU, host DRAM, and local SSD. The same model stays loaded throughout.
The separate diskless remote sweep uses `bench-ttft.py --scenarios remote`.

Start a dedicated requester at localhost:31001 with the pinned configuration,
metrics, tracing, `write_through`, `wait_complete`, and the same generated local
key for `--api-key` and `--admin-api-key`. Keep the key file outside artifacts.
Redact that key from startup logs and recorded launch arguments; the pinned
serving source exposes raw configuration. Deploy `bench-ttft.py` and
`ttft_trace.py` to `/workspace`, and use `/opt/sglang/bin/python` from the
verified serving environment. Run the supervisor in the background with an
outer watchdog; its per-context subprocess timeout is 700 seconds.

```sh
/opt/sglang/bin/python bench-tier-epochs.py \
  --root /scratch/UNIQUE-RUN/requester \
  --config /scratch/UNIQUE-RUN/requester/config.json \
  --mock-file /scratch/UNIQUE-RUN/requester/cache.bin \
  --admin-key-file /workspace/private/requester-admin-key \
  --gcs-prefix gs://APPROVED-BUCKET/UNIQUE-RUN
```

Before every context, including the first, it flushes upper tiers, detaches
storage through the existing idle-only admin API, verifies that no process
holds the exact debug file open, unlinks that file, and attaches the identical
configuration. Both detach and attach must return HTTP200. It rejects device
paths, symlinks, parent traversal, files outside the run, multiple assigned
paths, and configurations without fresh-file creation. The five actual study
resets passed descriptor-closure checks and all120 samples passed strict tier,
output, payload, and trace validation.

For a capable direct-receive native/serving pair, add `--direct-receive` and set
`agent.direct_receive: true` in the attached configuration. The supervisor
requires the explicit option to match that configuration, passes it to every
context's harness, and keeps the same idle-only reset/descriptor checks.
Eligible local SSD loads must prove exact direct bytes/segments and zero copies;
any counted safe fallback remains disclosed without a zero-copy claim.

Preparation runs outside measured requests. The flushed journal records each
control call and file-reset phase. Incremental control archives allow only
JSON/JSONL and reject actual credential bytes; source scripts are kept locally
and identified by SHA256 only. Per-sample data archives come from the harness.
Record storage-reset epochs, authentication, tracing cost, and exact GPU/NIC/
boot identity when combining context curves. The helper keeps source files
out of GCS; publish reproduction source through Git or retain it on RAID.
