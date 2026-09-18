# NIXLShard

NIXLShard is an in-process, rack-scale NVMe cache data path. `shardagent` and
`shardnclient` live in the application process so NIXL can register caller
buffers directly. One logical device is represented by one preallocated file;
objects use aligned offsets within it.

```text
application process A                      application process B
+-----------------------------+           +-----------------------------+
| caller buffers              |           | registered staging          |
| shardnclient + shardagent   |<-- UCX -->| shardagent + shardnclient   |
+-----------------------------+           +--------------+--------------+
                                                        |
                                                   NIXL POSIX
                                                        |
                                                 fixed file / NVMe
```

The current vertical slice implements the batch-only local `shardagent` path:

- safe create/open validation and exclusive ownership for a fixed-size file;
- direct registration of caller-owned host-memory regions;
- batched local load/store submission using one vector NIXL request per batch;
- per-item operation handles, terminal completions, polling/waits, and blocking
  batch wrappers;
- range, alignment, and bounded-inflight validation.

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

Remote UCX staging, cancellation, naming, eviction, and SGLang integration are
planned follow-up milestones.
