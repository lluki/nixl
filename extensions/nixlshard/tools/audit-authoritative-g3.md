**Agent written.**

# Saved authoritative G3 benchmark audits

These analysis tools independently validate saved JSON artifacts from the accepted A10/B7 authoritative G3 baseline and metadata-batching candidate. They make no HTTP calls, import no model/native package, and send no benchmark traffic.

Run against scratch copies of the archived artifacts. Each audit writes derived normalized/audit JSON into its working role directory; augmentation updates those derived files and records their previous SHA256. Do not run against immutable frozen final artifacts. Preserve raw producer JSON, accepted baseline archives and original normalized derivations separately.

The accepted source pins are native baseline `3310a0cd924619f60f42978247c1817206a1f008`, native candidate `128a65a5066a35d774b86241c5f2668cdd816965`, SGLang `72726e814f09016d4ad077f2f3cd4ef7e7968f5e`, and frozen benchmark source `7125d6b4c5935b4067a690bacffae6096f232dd4`. The exact-input wrapper SHA is `e1855a942cca2122d33ebd7a5f4ffe835bb5d2e71d3612e6e5c77164dc312873`; its source checkpoint is separate from the compiled native marker. The native build is the only permitted implementation difference.

Use Python 3.12 or newer with assertions enabled. The scripts reject `python -O`. Dependencies are the standard library and the saved data; no package installation or live serving environment is required.

The remote audit requires these requester role files: launch/config/domain/runtime maps, sanitized owner launch, exit status and all raw `remote-full` requests, outputs, samples, counters, traces and diagnostics. Copy the owner role's config/domain/runtime maps into a separate data directory as `owner-config.json`, `owner-domain-witness.json` and `owner-runtime-maps.json`. Augmentation additionally reads actual saved POSIX/UCX plugin maps from requester `native-plugin-maps-final.json` and owner `owner-native-plugin-maps-final.json`.

For example, replace the working directory variables with paths to restored scratch copies:

```sh
export AUDIT_EXPECTED_NATIVE=128a65a5066a35d774b86241c5f2668cdd816965
export AUDIT_OWNER_ARTIFACTS=/workspace/audit-inputs/candidate-owner
python3 audit-authoritative-g3-remote.py "$CANDIDATE_REMOTE_ROLE" "$BASELINE_REMOTE_ROLE"
python3 augment-authoritative-g3-remote-audit.py "$CANDIDATE_REMOTE_ROLE" "$BASELINE_REMOTE_ROLE"
```

Run staged candidate against accepted staged baseline and direct candidate against accepted direct baseline. Both candidate modes reuse the original staged token plan. The scripts separately verify the replay plan's original reference artifacts and the comparison baseline's token/sampling/output equality.

Remote gates cover all 30 cases, exact model namespace/geometry/NUMA/source/hardware, role-aware serving configuration, native library/binding maps, raw trace joins, completed native work before first forward, conserving CLOCK_MONOTONIC partitions, exact payload/destination spans and zero direct receiver copy. Metadata remains 4 KiB per requested page and is separate from payload bytes. Directional metadata counters are independently recomputed over full-generation/export windows, including any background work; they are never stacked into first-token latency.

Local audit additionally requires all five `context-*` directories, the existing epoch supervisor journal/exit proof, and a matching remote requester role as a hardware/domain anchor:

```sh
export AUDIT_REMOTE_ROLE=/workspace/audit-inputs/candidate-direct-requester
python3 audit-authoritative-g3-local.py "$CANDIDATE_LOCAL_ROLE" "$BASELINE_LOCAL_ROLE"
python3 augment-authoritative-g3-local-audit.py "$CANDIDATE_LOCAL_ROLE" "$BASELINE_LOCAL_ROLE"
```

Local gates cover all 120 accepted input/sampling/text/output comparisons; canonical local 512 GiB disk/profile equality with only the native pin changed; exact tier masks for this fixed page64 experiment; flush/detach/FD-closure/unlink/attach ordering; actual local direct payload/2 K-V destination spans per page; metadata records, zero per-request receiver fallback/copy; and actual mapped plugins. Epoch replay manifests link the approved complete 120-case plan. The general frozen benchmark permits larger GPU/host hits in other workloads; these particular archived fixed-context cases all have exactly `floor((context-1)/64)*64` cached tokens, which this study-specific audit enforces.

Derived rows preserve accepted request/sample/output artifact SHA256, candidate full request artifact SHA256, raw trace references and direction-aware metadata proof. Local/remote profile differences remain explicit; a diskless remote requester is not represented as having the same configuration as the local SSD requester. Unknown wall spans and all warmups remain in raw data. Only the planned warmups are omitted from measured aggregates.

Timings are requester-clock intervals plus nested owner duration witnesses, not synchronized remote timestamps. Means are computed per request before aggregation. No sum of independent stage medians, pure CPU attribution, or claim that end-to-end differences establish the 5% overhead target is made. Baseline filtered runtime maps did not capture plugin paths; candidate explicit plugin maps are recorded without inventing missing baseline evidence.

Offline regressions use copies of the real archived baseline. Configure the artifact paths; without actual fixtures, they skip explicitly:

```sh
export G3_AUDIT_FIXTURE_ROOT=/raid/nixlshard-v2/gb200-observations/authoritative-g3-20261005-a10-b7
export G3_AUDIT_LOCAL_REFERENCE=/workspace/replay-reference/local
export NIXLSHARD_TEST_DIR=/raid/nixlshard-v2
python3 test-authoritative-g3-audit.py
python3 test-authoritative-g3-local-mask.py
```

The six remote regressions extract `a10-b7-staged-audited-requester.tar.gz` and `a10-b7-staged-owner-ready.tar.gz` into temporary scratch directories. They accept unchanged real data and reject altered directional counters, metadata bytes, post-forward DMA, sampling parameters and storage geometry. The four local regressions validate all 120 actual masks and reject overlong isolated hits, inconsistent totals and wrong-tier contamination. No frozen normalized/audit artifact is modified.

Keep these analysis scripts, tests and reproduction instructions in Git or RAID. Only approved data JSON/logs/config/SHA manifests belong in GCS; never upload source or credentials.
