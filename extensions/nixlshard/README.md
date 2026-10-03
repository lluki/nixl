**Agent written**

# NIXLShard v2

Prototype immutable-key SSD cache, built as an optional NIXL extension on the
`nixlshard-v2` branch. Design and progress:
`ssh://git@gitlab-master.nvidia.com:12051/adv-dev-team/nixlshard-design.git`,
branch `main`.

Build with `tools/build-dev.sh`; source its generated `env.sh` before running
Python or the standalone binaries. The helper uses isolated UCX/NIXL installations
and preserves the image's Torch installation. Compare `nixlshard.__build_marker__`,
the imported module path and loaded shared libraries with the current source.

## Minimal Python example

```python
import ctypes
import time
from nixlshard import Agent

# create=True initializes a new empty debug file only. Existing formatted files
# reopen with create=False. The client supplies and owns device assignment.
agent = Agent({
    "name": "worker-0",
    "disks": [{"path": "/tmp/new-cache.bin", "capacity_bytes": 64 * 1024 * 1024,
               "unit_bytes": 4096, "metadata_bytes": 1024 * 1024, "create": True}],
    "staging_slots": 4, "staging_slot_bytes": 1024 * 1024,
})
source = ctypes.create_string_buffer(b"immutable value")
destination = ctypes.create_string_buffer(len(source))
src = agent.register_memory(ctypes.addressof(source), len(source))
dst = agent.register_memory(ctypes.addressof(destination), len(destination))

def wait(handle):
    while (result := agent.poll(handle)) is None:
        time.sleep(0.0001)
    agent.release(handle)  # drains underlying work even after a timeout
    return result

assert wait(agent.batch_store([{"key": "namespace:key",
                               "segments": [(src, 0, len(source))]}])) == ["success"]
assert wait(agent.batch_load([{"key": "namespace:key",
                              "segments": [(dst, 0, len(destination))]}])) == ["success"]
assert destination.raw == source.raw
agent.deregister_memory(src)
agent.deregister_memory(dst)
agent.close()
```

Keys identify immutable, complete values. A successful load returns exactly the
stored length; mismatched destination length fails. Segments concatenate in caller
order. Keep registered memory valid until every referencing handle is released.
Raw-address registration does not retain a Python buffer object: the caller must
keep that object alive through release. The token prevents deregistration while
handles reference it. This first implementation copies caller segments through
an aligned owned staging pool registered with NIXL once.
Store packing and successful load copies are gated against cancellation, so
neither begins after polling has reported a terminal timeout.

Submission returns a handle without connecting peers or waiting for payload I/O.
`poll()` returns `None` while pending, then a stable per-object status list.
Logical deadlines can become visible before underlying I/O drains. `release()`
waits for worker quiescence before dropping caller references. Unreleased terminal
handles consume the admission budget. Invalid arguments and exhausted live-handle
budget raise exceptions; individual cache/resource/I/O failures return statuses.

For request-specific diagnostics, set `enable_trace=True` and fetch
`agent.trace(handle)` after successful polling and before `release(handle)`.
The bounded list belongs to that handle; releasing it also releases its trace.
Events contain `stage`, requester `start_ns`/`end_ns` in Linux CLOCK_MONOTONIC,
`bytes`, object ordinals and the remote control `request_id`. Admission/queue,
local POSIX reads, remote RPCs and staging copies are separate intervals.
Remote RPC events additionally carry measured `owner_posix_ns`, `owner_ucx_ns`
and physical `owner_read_bytes`. Missing/failed owner measurements omit these
Python fields rather than claiming zero cost. Owner read and UCX durations are
sequential children of the RPC: replace the enclosing RPC with its read,
network and control residual when constructing a stack; never add all four.
Tracing requires matching instrumented endpoints and uses the measured group
envelope even for a single remote object. It is off by default and introduces
small diagnostic overhead; it preserves owned staging and timeout quarantine.

## Remote loads and metadata

An owner uses NIXL POSIX to read its disk into owned DRAM, then initiates NIXL UCX
WRITE into requester-owned registered staging. The requester copies into caller
segments only after confirmed completion. Control TCP carries metadata and
descriptors, never payload bytes.

`remote_batch_limit` defaults to 1. With both endpoints upgraded, set it to 8
to group consecutive same-owner loads that fit one `staging_slot_bytes` slot.
The owner pins each value, submits batched POSIX reads and one scatter/gather
UCX WRITE; caller copies remain ordered and cancellation-gated. Missing keys
and invalid lengths retain individual statuses. A smaller owner slot or excess
allocation padding falls back to individual loads. An uncertain grouped RPC
quarantines the entire requester-owned slot until the existing cleanup fence
confirms quiescence. The pool remains bounded by `staging_slots` times
`staging_slot_bytes`; this option neither registers caller memory for UCX nor
changes caller-buffer lifetimes. For 16 MiB KV pages, a 128 MiB slot fits eight.

This prototype explicitly selects the POSIX Linux AIO queue. Other POSIX queue
implementations need separate error/quiescence validation before being enabled.

Configure `peers={"owner": {"host": "127.0.0.1", "port": 32001}}` and pass
`hint="owner"` per load, or configure
`metadata_endpoint={"host": "127.0.0.1", "port": 32000}` on both agents.
`MetadataServer({"host": "127.0.0.1", "port": 32000})` supplies bounded owner
registration, TTL, incarnation fencing, monotonic announcements and advisory
lookup. Discovery, connection establishment and checkpoints run in a maintenance
thread. Initial remote operations can return not_ready until that work completes.
The owner always verifies keys, including after positive exists/hint lookup.

The first wire protocol accepts numeric IPv4 addresses. A listener using 0.0.0.0
needs a routable advertised address before use across machines; use a concrete
interface address for this prototype.

## Persistence and resource limits

Each assigned path has an independent G3 index. A whole object occupies ordered
fixed-size slots on one path; placements may be scattered. Conservative FIFO
reclamation skips pinned entries. Double metadata snapshots and durable selectors
remove retired mappings before slot reuse; checkpoints sync payload before
publishing insertions. Uncheckpointed insertions may disappear after restart.
Corrupt committed metadata fails closed. The prototype initializes empty debug
files and reopens formatted devices; formatting a fresh raw SSD is not yet exposed.

Default unit size is 64 KiB. Direct I/O uses aligned private buffers and units
divisible by 4096. Maximum object size is the configured staging slot size.
Batches contain at most 128 objects; staging, live handles, connections, imported
peer identities, hint caches and remote cleanup records all have fixed bounds.
The owner allows 272 control connections for up to 128 peers, including separate
normal and cleanup channels plus a small control margin. An exists batch shares
one deadline across all contacted owners; failed calls contribute to its timer.

A timed-out remote operation quarantines its staging destination until the same
owner incarnation confirms quiescence. Repeated cleanup preserves cancellation
fences. A replacement incarnation alone cannot prove an earlier write drained.
If confirmation cannot be obtained, slots stay unavailable, and close raises
instead of freeing them. The destructor retains the bounded native pool in that
case. This conservative behavior protects caller data; recovery of such pools
and lease-based reclamation remain future work.

## Validation and measurement

Meson tests cover persistence, metadata, wire, local transfer and remote UCX paths.
The SGLang branch adds a distinct HiCache backend and adapter integration tests.
Use `stats()` to capture payload bytes and nanoseconds for POSIX, UCX, control,
staging copies and metadata checkpoints. Timings can overlap across workers and
are component counters rather than end-to-end TTFT attribution.

The development A100/TCP environment validates behavior. The target
Qwen3-32B-FP8/GB200/RDMA TTFT budget still requires its specified hardware and model
revision. No target overhead claim is made by local debug-file measurements.

Run `python tools/bench-native.py` from this directory for verified local and
two-process UCX TCP loads with one request outstanding. It creates disposable
mock SSD files beneath `/raid/nixlshard-v2` and records component counters.
Run `python tools/bench-baselines.py` for matching `fio` Linux AIO direct-file
reads and two-process UCX PUT baselines (`fio` and UCX `ucx_perftest` required).
Baseline UCX progresses actively, while the runtime uses its configured progress
thread and polling delay; this difference belongs in any comparison.
