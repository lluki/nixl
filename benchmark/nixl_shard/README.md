# NIXLShard two-pod data path check and microbenchmark

The client performs a deterministic, byte-for-byte SET then GET check for every
(size, queue depth, I/O batch size) condition. It exits on any non-OK completion,
short transfer, or mismatch. Timing starts only after that check and after buffers have been registered.
The agent may create a new control connection per operation; connection setup
may therefore be included in the measured latency. SET and GET are timed
separately; their samples and summary are JSON lines on stdout.

This is an **agent data-plane** benchmark. It does not include nserv metadata
RPCs, SGLang, cache hit ratio, or end-to-end TTFT. "I/O batch size" groups
`IoItem` requests in one agent submission; it is not the metadata batch size.
The TCP mode in the current branch binds to loopback only. UCX mode requires
the opt-in UCX NIXLShard interface; the harness fails if that API is absent.
For each UCX condition it also checks that the agent's UCX byte counter advanced
by at least the logical SET plus GET bytes.

## Cross-node RDMA

In both Pods, run from the NIXL repository root and point Python at the
source checkout. Reserve unique file names for each run, since file creation
is exclusive.

Pod B (server, replace the address if needed):

```sh
taskset -c 4-7 env UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_0:1 \
PYTHONPATH=src/bindings/python/nixl-meta \
python benchmark/nixl_shard/nixlshard_bench.py server \
  --transport ucx --direct-io --workers 32 --listen-host 0.0.0.0 --port 18741 \
  --file /workspace/bench-nixlshard/rdma-server-001.bin \
  --file-size 512m > /workspace/bench-nixlshard/rdma-server-001.log 2>&1
```

Pod A (client; use Pod B's current IP and the server's `server_ready` port):

```sh
taskset -c 0-3 env UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_0:1 \
PYTHONPATH=src/bindings/python/nixl-meta \
python benchmark/nixl_shard/nixlshard_bench.py client \
  --mode ucx --direct-io --workers 32 --remote-host 192.168.94.163 --remote-port 18741 \
  --file /workspace/bench-nixlshard/rdma-client-001.bin \
  --file-size 512m --sizes 4k 16k 64k 256k 1m 4m \
  --queue-depths 1 2 4 8 16 32 --io-batch-sizes 1 4 16 \
  --warmup 10 --iterations 100 \
  --output /workspace/bench-nixlshard/rdma-client-001.jsonl \
  > /workspace/bench-nixlshard/rdma-client-001.native.log 2>&1
```

Start with a small correctness matrix using `--iterations 0 --sizes 4k 64k 1m
--queue-depths 1 4 --io-batch-sizes 1 4`. Then scale duration and concurrency.
Use `--direct-io` on both ends for aligned file I/O. The script allocates
4096-byte aligned buffers. Pod storage is ephemeral; copy logs to the wiki
before deleting Pods.

Capture topology on each Pod before the run:

```sh
hostname; uname -a; taskset -pc $$
cat /sys/class/infiniband/mlx5_0/device/numa_node
cat /sys/devices/system/node/node0/cpulist
findmnt -T /workspace -o TARGET,FSTYPE
ucx_info -v | head -5
```

The example pins the client to CPUs 0–3 and the server to 4–7, both on
NUMA node 0 with `mlx5_0` in the current GB200 Pods. The JSONL records the
actual process affinity, UCX/NIXL versions, and source status. The Pod file
path is ephemeral; check its mount before describing it as an NVMe baseline.

For a TCP fallback check, server and client must run in the *same Pod* with
`--transport tcp --listen-host 127.0.0.1` and
`--mode tcp --remote-host 127.0.0.1` respectively. Local POSIX baseline
needs only `client --mode local`.

The JSONL includes source commit, environment, correctness pass events,
individual timing samples, logical bytes and throughput. It does not measure
wire bytes, CPU/NIC/NVMe counters, setup time, or confidence intervals. Do not
interpret three-sample smoke results as stable p95/p99 measurements.

## Client and naming service end-to-end check

After the UCX server above is ready, run this on Pod A. It creates a local
loopback naming service and RPC client, registers Pod B as the only storage
device, then performs batched reserve/store/commit and lookup/load/release
through `ShardClient`. Every result must route to Pod B, match the source
bytes, leave no active leases, and advance the UCX byte counter. The SET/GET
times here include metadata RPCs and the data path.

```sh
taskset -c 0-3 env UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_0:1 \
PYTHONPATH=src/bindings/python/nixl-meta \
python benchmark/nixl_shard/nixlshard_e2e.py \
  --transport ucx --remote-host 192.168.94.163 --remote-port 18741 \
  --file /workspace/bench-nixlshard/e2e-client-001.bin \
  --remote-file-size 512m --sizes 4k 16k 64k \
  --batch-sizes 1 4 16 --rounds 3 \
  --output /workspace/bench-nixlshard/e2e-client-001.jsonl \
  > /workspace/bench-nixlshard/e2e-client-001.native.log 2>&1
```

For a small first check use `--sizes 4k 16k --batch-sizes 1 4 --rounds 1`.
For a loopback baseline, run the server and client in one Pod using
`--transport tcp` and `--remote-host 127.0.0.1`.
