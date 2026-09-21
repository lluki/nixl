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
- an in-memory naming service with batched allocation/lookup/finalization,
  leases, eviction, device heartbeats, and stop/drain/offline management;
- a batched client implementing reserve-store-commit and
  lookup-load-release, including bounded retries and backpressure.

Set `tcp_listen_host` on the device-owning agent, read its `tcp_endpoint`,
and add a matching `RemoteDevice` to the routing agent. TCP is currently the
functional loopback transport and refuses non-loopback server binds; it stages
remote payloads and will be replaced by the UCX data path without changing the
public batch API.

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

The current remote path is the functional TCP fallback because UCX is not
available in this validation environment. UCX/RDMA integration and performance
validation remain follow-up work. Cancellation is best-effort: the public
result becomes `CANCELLED` only after the underlying local or remote operation
is terminal, so registered buffers are never released early.
