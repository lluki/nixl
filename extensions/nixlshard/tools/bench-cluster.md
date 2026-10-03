**Agent written**

# Cross-host native benchmark

`bench-cluster.py` runs an SSD-owning agent and a requester on separate hosts.
HTTP carries only benchmark configuration, hashes, counters and shutdown; NIXL
POSIX/UCX carries the payload. Each successful read is hash-verified. Warmup,
connection establishment, checkpoints, statistics queries and hashing are outside
the measured submission-to-release interval. Retain both JSON results and logs.

Activate the same installed NIXLShard build on both hosts. In the GB200 dev pods,
the validated transport environment is:

```bash
. /workspace/install/nixlshard/env.sh
export UCX_TLS=rc,self,sm,cuda_copy UCX_NET_DEVICES=mlx5_0:1 UCX_LOG_LEVEL=info
# Owner pod B; disposable SSD files go under /scratch.
python extensions/nixlshard/tools/bench-cluster.py owner \
  --host 192.168.91.189 --output /scratch/nixlshard-v2/owner.json
# Requester pod A, after the owner starts.
python extensions/nixlshard/tools/bench-cluster.py requester \
  --host 192.168.98.221 --owner http://192.168.91.189:32080 \
  --iterations 20 --warmup 3 --stop-owner \
  --output /scratch/nixlshard-v2/requester.json
```

Supply current pod addresses rather than assuming these example addresses stay
valid. Add `--direct-io` on both sides for aligned regular-file O_DIRECT. Defaults
measure 4 KiB, 1 MiB and 16 MiB with one request outstanding. The owner exits after
600 seconds unless the requester stops it sooner. Its temporary files are removed
on orderly close. The metadata HTTP service is intended for an isolated experiment
and has no authentication; do not expose it beyond the experiment network.

Different host boot IDs are required and recorded. To establish RoCE, also retain
UCX logs showing the actual inter-node `rma(rc_mlx5/mlx5_0:1)` endpoint selection.
Successful transfers with default UCX transport discovery alone do not establish
RDMA. In this image, restricting transports to `rc_mlx5` without an auxiliary
transport failed endpoint creation; the validated `rc` configuration above selects
RC payload and UD keepalive/bootstrap. CUDA support is required by NIXL when a GPU
is present, even for these DRAM payloads.

Component counters overlap: control includes owner work, and multiple workers can
run concurrently. Do not add them or subtract standalone fio/UCP measurements to
claim attributable overhead. This benchmark measures native debug-file latency,
not model TTFT or raw-device durability. Record CPU affinity, storage settings,
UCX version and installed build markers with the results.
