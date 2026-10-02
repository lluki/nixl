#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fio/Linux AIO and two-process UCX TCP baselines using a new mock SSD file."""
import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time

def run(args):
    fio = shutil.which("fio")
    ucx = pathlib.Path(args.ucx_prefix) / "bin" / "ucx_perftest"
    if not fio or not ucx.is_file():
        raise RuntimeError("fio and UCX ucx_perftest are required")
    directory = pathlib.Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = {"schema": "nixlshard-baselines-v1",
               "scope": "same-host /raid mock-file and two-process TCP; not raw disk or RDMA",
               "fio_version": subprocess.check_output([fio, "--version"], text=True).strip(),
               "ucx_prefix": str(args.ucx_prefix), "sizes": args.sizes,
               "one_request_outstanding": True, "fio": [], "ucx": []}
    with tempfile.TemporaryDirectory(prefix="baseline-", dir=directory) as temporary:
        ssd = str(pathlib.Path(temporary) / "mock.ssd")
        common = [fio, "--name=nixlshard-baseline", f"--filename={ssd}",
                  "--size=512m", "--ioengine=libaio", "--iodepth=1", "--numjobs=1",
                  "--direct=1", "--group_reporting", "--output-format=json"]
        subprocess.run(common + ["--rw=write", "--bs=1m", "--end_fsync=1",
                                  f"--output={directory / 'fio-prepare.json'}"],
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        for size in args.sizes:
            output = directory / f"fio-read-{size}.json"
            subprocess.run(common + ["--rw=read", f"--bs={size}", "--time_based",
                                      f"--runtime={args.runtime}", f"--output={output}"],
                           check=True, timeout=args.runtime + 30, stdout=subprocess.DEVNULL)
            data = json.loads(output.read_text())["jobs"][0]
            if data["error"]:
                raise RuntimeError(f"fio error: {data['error']}")
            read = data["read"]
            results["fio"].append({"bytes": size, "bandwidth_Bps": read["bw_bytes"],
                                   "iops": read["iops"], "latency_mean_us": read["lat_ns"]["mean"] / 1000,
                                   "completion_mean_us": read["clat_ns"]["mean"] / 1000,
                                   "raw": str(output)})
    environment = dict(os.environ)
    environment["UCX_TLS"] = "tcp,self,cuda_copy"
    environment["LD_LIBRARY_PATH"] = str(pathlib.Path(args.ucx_prefix) / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    for size in args.sizes:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        iterations = min(2000, max(64, 512 * 1024 * 1024 // size))
        common = [str(ucx), "-t", "ucp_put_bw", "-s", str(size), "-O", "1",
                  "-n", str(iterations), "-w", str(min(200, iterations // 4)),
                  "-p", str(port), "-v", "-f"]
        server_log = directory / f"ucx-server-{size}.log"
        client_log = directory / f"ucx-client-{size}.csv"
        with server_log.open("w") as output:
            server = subprocess.Popen(common, stdout=output, stderr=subprocess.STDOUT, env=environment)
            try:
                time.sleep(0.3)
                if server.poll() is not None:
                    raise RuntimeError(f"UCX server failed; see {server_log}")
                with client_log.open("w") as client_output:
                    subprocess.run([str(ucx), "127.0.0.1", *common[1:]], env=environment,
                                   stdout=client_output, stderr=subprocess.STDOUT, check=True, timeout=30)
                if server.wait(timeout=10):
                    raise RuntimeError(f"UCX server failed; see {server_log}")
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=5)
        results["ucx"].append({"bytes": size, "iterations": iterations,
                               "transport": environment["UCX_TLS"], "outstanding": 1,
                               "raw": str(client_log), "server_log": str(server_log)})
    output = directory / "summary.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/raid/nixlshard-v2/baselines")
    parser.add_argument("--ucx-prefix", default="/workspace/deps/ucx")
    parser.add_argument("--runtime", type=int, default=3)
    parser.add_argument("--sizes", type=lambda text: [int(v) for v in text.split(",")],
                        default=[4096, 1048576, 16777216])
    options = parser.parse_args()
    if options.runtime < 1 or any(size < 1 or size > 32 * 1024 * 1024 for size in options.sizes):
        parser.error("runtime must be positive and sizes must be in (0,32MiB]")
    run(options)
