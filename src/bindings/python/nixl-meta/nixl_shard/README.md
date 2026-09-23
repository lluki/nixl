# NIXLShard

NIXLShard is an in-process, rack-scale NVMe cache data path. `shardagent` and
`shardnclient` live in the application process so NIXL can register caller
buffers directly. One logical device is represented by one preallocated file;
objects use aligned offsets within it. A logical device is identified by its
device ID and generation; replacing a device requires a newer generation and
never inherits the old generation's data.

```text
application process A                      application process B
+-----------------------------+           +-----------------------------+
| caller buffers              |           | registered staging          |
| shardnclient + shardagent   |<- data -->| shardagent + shardnclient   |
+-----------------------------+           +--------------+--------------+
                                                        |
                                                   NIXL POSIX
                                                        |
                                                 fixed file / NVMe
```

The current vertical slice implements the batch-only local and remote
`shardagent` paths:

- safe create/open validation and exclusive ownership for a fixed-size file;
- direct registration of caller-owned host-memory regions;
- batched local load/store submission using one vector NIXL request per batch;
- per-item operation handles, terminal completions, polling/waits, and blocking
  batch wrappers;
- range, alignment, and bounded-inflight validation.
- mixed-device batch routing through `AgentConfig.remote_devices`;
- a loopback-only framed TCP fallback with bounded request, staging-byte, and worker limits,
  per-item completion results, deadlines, generation fencing, and orderly
  shutdown;
- an opt-in UCX/RDMA data path using registered, bounded staging allocations;
  TCP carries request metadata and terminal results with zero payload bytes;
- an in-memory naming service with batched allocation/lookup/finalization,
  leases, eviction, device heartbeats, and stop/drain/offline management;
- a batched client implementing reserve-store-commit and
  lookup-load-release, including bounded retries and backpressure.

For UCX, set `ucx_listen_host` on the device-owning agent, read its
`ucx_endpoint`, and add a matching `RemoteDevice(..., transport="ucx")` to
the routing agent. The endpoint is the control socket; the agents exchange
NIXL metadata there and transfer registered staging bytes through UCX. For
`bootstrap.create_client`, use `ucx_listen_host` and `ucx_listen_port` in the
local config and set `transport: "ucx"` on each UCX `remote_devices` entry.
The bootstrap registers a `ucx://` endpoint with the naming service. For
example, on a GB200 pair with RDMA device `mlx5_0:1`, set
`UCX_TLS=rc,cuda_copy` and `UCX_NET_DEVICES=mlx5_0:1` in both processes.
`ucx_transfer_bytes` counts successful UCX payload bytes on the initiating
agent. A server bound to `0.0.0.0` reports that wildcard in `ucx_endpoint`;
route clients to the server's reachable address.

The TCP fallback remains available with `tcp_listen_host`, `tcp_endpoint`,
and the default `RemoteDevice(..., transport="tcp")`. Its server accepts
loopback binds only. The batch API is identical for both transports.

UCX uses `tcp_max_request_bytes`, `tcp_max_staging_bytes`,
`tcp_max_workers`, and `tcp_request_timeout` for its control and staging
limits. Set `tcp_max_request_bytes` above the sum of the item lengths in the
largest batch, and keep `tcp_max_staging_bytes` at least as large on both
agents. For example, eight Qwen3-32B-FP8 KV pages of about 16 MiB each exceed
the 64 MiB default request limit; a 256 MiB request limit and a 512 MiB staging
limit passed the GB200 model-level L3 check. A rejected batch returns an error
for each item and does not publish its reserved extents. Each batch stages at
most its summed item length on each side. Each
UCX direction has its own NIXL agent and a lazy registered staging pool capped
by `tcp_max_staging_bytes`. Registrations stay live for reuse until agent
close, so metadata refreshes add stable descriptors without disconnecting
inbound transfers. A transfer holds its staging lease until NIXL reports
terminal completion. If control is lost while remote access may still be
active, the server retains that allocation against the pool cap until process
teardown. An uncertain NIXL release or failed deregistration similarly retains
client staging. UCX transfers to the same peer are serialized while metadata
is refreshed. Control connections and file operations can overlap, but this
peer serialization can limit scaling at higher queue depths. Transport timings
should account for control connection setup and initial staging registration.
If the initiating agent
loses a remote request before receiving its terminal reply, it suppresses its
naming-service terminal report. The data-owning agent reports after file I/O
and UCX transfer complete; a management fence remains the recovery path if
the data-owning agent is unreachable.

Bootstrap configuration calls the generation `logical_device_generation`;
`agent_epoch` remains accepted as a compatibility alias for the V1 wire/API
field. Peer routes are static, so install a replacement generation by rebuilding
the embedded client/agent configuration with the new generation and endpoint.
The bootstrap also binds each data-owning agent to the naming service's current
authority epoch. A restarted empty naming service therefore cannot reuse a
surviving old agent even if it allocates the same device generation. Both the
initiating client and the data-owning agent report terminal I/O; reports are
idempotent and allow reclaim-pending extents to become reusable promptly.

The matching core binding and POSIX plugin advertise complementary safe-release
capabilities (`HAVE_SAFE_POSIX_ERROR_RELEASE_CORE` and the runtime parameter
`safe_error_release=true`): POSIX errors become terminal only after backend
callbacks are quiesced, and active release retries preserve request state. If
either marker is absent, the agent deliberately keeps an ambiguous request and
its caller buffers alive until NIXL reports `DONE`; a bounded `wait` or `close`
can time out, but the extent is not exposed for unsafe reuse. A bounded
`load_batch`, `store_batch`, or `wait` raises `OperationTimeoutError`; its
`handles` remain valid and can be passed to `wait` again.

The runtime marker is currently enabled for Linux AIO and POSIX AIO. io_uring
terminal-submit recovery still needs an explicit kernel-owned request drain, so
that queue remains fail-closed on an ambiguous fatal submission.

Cancellation is best-effort: the public
result becomes `CANCELLED` only after the underlying local or remote operation
is terminal, so registered buffers are never released early.
