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
    "namespace_id": "example-schema-v1", "numa_node": 0,
    "registration_mode": "EXPLICIT",
    "disks": [{"path": "/tmp/new-cache.bin", "capacity_bytes": 64 * 1024 * 1024,
               "unit_bytes": 4096, "min_object_bytes": 1,
               "max_object_bytes": 1024 * 1024, "numa_node": 0, "create": True}],
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

Keys are nonempty binary values up to the configured width (at most 32 bytes).
Python accepts bytes, including NUL/non-UTF8 octets, or UTF-8 strings. The complete
canonical model/layout schema binds a G3 instance and is persisted without hashing
or truncation. SGLang passes its unchanged 32-byte page digest. Remote/MD routing
length-frames the full namespace and key; instance names may differ across peers.
The generic API defaults to the exact namespace `nixlshard.generic.v2`.
Keys identify immutable, complete values. A successful load returns exactly the
stored length; mismatched destination length fails. Segments concatenate in caller
order. Keep registered memory valid until every referencing handle is released.
Raw-address registration does not retain a Python buffer object: the caller must
keep that object alive through release. The token prevents deregistration while
handles reference it. By default, loads copy caller segments through an aligned
owned staging pool registered with NIXL once. Store packing and staged load copies
are gated against cancellation.

Whole caller regions are registered with POSIX and UCX once. Set
`direct_receive=True` to use those regions as receive destinations. Remote
loads then write directly into the supplied scatter segments; the SSD owner
still stages the read in its own DRAM. Aligned local loads also read directly:
logical length must equal the allocation's physical length and all destination
addresses and lengths must be 4 KiB aligned. Other local loads use the counted
staging fallback, without writing allocation padding outside the requested spans.

Direct destinations require exclusive ownership through handle release.
Overlapping destinations, including those owned by another live handle, are
rejected. After a timeout the destination may still change; never publish or
reuse failed pages. `is_quiescent(handle)` becomes true only after the worker
finishes and every uncertain remote write has been fenced by its original owner
incarnation. Direct-mode `release(handle)` rejects while that proof is missing,
retaining the handle and registrations. A cache allocator must lease these pages,
defer frees and prohibit pool resets until release succeeds. Losing contact with
the original owner may retain the pages indefinitely within the admission bound.

Submission returns a handle without connecting peers or waiting for payload I/O.
`poll()` returns `None` while pending, then a stable per-object status list.
Logical deadlines can become visible before underlying I/O drains. In staged
mode, `release()` waits for worker quiescence before dropping caller references.
Unreleased terminal
handles consume the admission budget. Invalid arguments and exhausted live-handle
budget raise exceptions; individual cache/resource/I/O failures return statuses.

For request-specific diagnostics, set `enable_trace=True` and fetch
`agent.trace(handle)` after successful polling and before `release(handle)`.
The bounded list belongs to that handle; releasing it also releases its trace.
Events contain `stage`, requester `start_ns`/`end_ns` in Linux CLOCK_MONOTONIC,
`bytes`, object ordinals and the remote control `request_id`. Admission/queue,
local POSIX reads, remote RPCs and staging copies are separate intervals.
Remote RPC events carry payload-only `owner_posix_ns`, `owner_ucx_ns` and physical
`owner_read_bytes`, plus `owner_metadata_ns`/`owner_metadata_bytes` for SSD record
I/O and `owner_staging_copy_ns`/`owner_staging_copy_bytes` for sender fallback
copies. Local POSIX events carry the same metadata/payload/copy children; their
`bytes` count the logical returned value. Replace the enclosing local/RPC window
with its children and residual when constructing a stack. Remote owner copies
are distinct from requester fallback copies; never add parent and child times.
Tracing requires matching instrumented endpoints and uses the measured group
envelope even for a single remote object. It is off by default and introduces
diagnostic overhead; its cost requires a matched on/off measurement. It preserves
owned staging and timeout quarantine.

## Remote loads and metadata

An owner uses NIXL POSIX to read its disk into owned DRAM, then initiates NIXL UCX
WRITE into requester-owned registered staging, or directly into registered caller
segments when direct receive is enabled. Staged loads copy into caller segments
after confirmed completion. Control TCP carries metadata and descriptors, never
payload bytes. Direct receive uses distinct scatter operations; both endpoints
must be upgraded. Caller registration changes refresh the owner's complete
remote memory view, serialized against active writes to that requester.

`remote_batch_limit` defaults to 1. With both endpoints upgraded, set it to 8
to group consecutive same-owner loads that fit one `staging_slot_bytes` slot.
The owner pins each value, submits batched POSIX reads and one scatter/gather
UCX WRITE; caller copies remain ordered and cancellation-gated. Missing keys
and invalid lengths retain individual statuses. A smaller owner slot or excess
allocation padding falls back to individual loads. An uncertain grouped RPC
quarantines the entire requester-owned slot until the existing cleanup fence
confirms quiescence. The pool remains bounded by `staging_slots` times
`staging_slot_bytes` in staged mode. Grouping alone does not enable direct receive.
With direct receive, the owner retains that staging bound, while requester
destinations are protected by live handles and allocator leases. For 16 MiB KV
pages, a 128 MiB owner slot fits eight; page-first K/V layout uses two 8 MiB
destination segments per page.

This prototype explicitly selects the POSIX Linux AIO queue. Other POSIX queue
implementations need separate error/quiescence validation before being enabled.

Configure `peers={"owner": {"host": "127.0.0.1", "port": 32001}}` and pass
`hint="owner"` per load, or configure
`metadata_endpoint={"host": "127.0.0.1", "port": 32000}` on both agents.
`MetadataServer({"host": "127.0.0.1", "port": 32000})` supplies bounded owner
registration, TTL, incarnation fencing, monotonic announcements and advisory
lookup. Discovery, connection establishment and announcements run in a maintenance
thread. Initial remote operations can return not_ready until that work completes.
The owner always verifies keys, including after positive exists/hint lookup.

The first wire protocol accepts numeric IPv4 addresses. A listener using 0.0.0.0
needs a routable advertised address before use across machines; use a concrete
interface address for this prototype.

## Persistence and resource limits

The distributed Agent delegates placement, allocation, reclamation, payload I/O
and recovery to an authoritative G3TransferLayer. Each disk has two ownership
ledgers: fixed aligned metadata records and payload slots. Geometry derives from
`unit_bytes`, model min/max object bytes, key width and metadata alignment;
`metadata_bytes` is a deprecated compatibility field. RAM retains key/identity/
record-location/claims; every read fetches its ordered slots and actual length
from the SSD under a claim. FIFO policy reclaims complete unclaimed objects.

Batched reads fetch at most eight exact allocation records per NIXL metadata
request. Large record geometries split requests to keep metadata scratch within
128 MiB; claims remain held through payload completion. CLEAN close combines
contiguous FREE records into writes of at most 1 MiB, or one record when larger,
while preserving LIVE records and the persistence barriers below.

Recognized CLEAN media restores the exact live set. DIRTY or invalid recovery
state starts empty, and DIRTY is durably persisted before admission. Evictions
invalidate records before reuse without a per-victim durability barrier.
`close("CLEAN")` drains work, finalizes records and persists CLEAN last after
payload/metadata barriers; failures remain DIRTY and are reported. `checkpoint()`
flushes DIRTY state and does not enable recovery. `close("DISCARD")` and destruction
leave DIRTY without erasing payload. Unknown media is never implicitly formatted;
recognized incompatible media requires explicit `reset=True`.

The client assigns allowed SSD paths and intended NUMA affinity. G3 discovers
topology only for those paths; debug files require an explicit `numa_node`.
Writes use healthy assigned devices on that node only. Reads use their recorded
placement. The façade always supplies intended affinity, including for staging.
Advanced `g3_instances` configure distinct named namespaces with dedicated disk
sets; per-object `g3_instance` and `numa` select them. `batch_exists(...,
g3_instance="local")` is a RAM lookup without a claim or SSD read.

`include/nixlshard/g3.h` also provides the standalone C++ G3Session API: bounded
asynchronous open/read/write/register/deregister/close completions over the same
core. EXPLICIT mode rejects unregistered caller spans; AUTOMATIC temporarily
registers missing spans. General inferred-affinity overloads query buffer NUMA
placement and reject unknown/mixed placement; they never infer the calling CPU.
The initial POSIX payload backend supports DRAM, not direct VRAM access. Partial
startup warns for each excluded disk and succeeds if a usable device remains.

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
staging copies and directional metadata I/O. Timings can overlap across workers and
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
