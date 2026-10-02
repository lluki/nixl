# Development build

The optional Meson target builds `libnixlshard`, its Python bindings, the agent
and metadata executables, and native tests. It is disabled in ordinary NIXL
builds. Build inside a development container; the helper installs to a separate
prefix and preserves the image's SGLang, Torch, and production NIXL packages.

The initial environment is Ubuntu 24.04 / Python 3.12 / CUDA 12.9, from the
existing `lmsysorg/sglang:latest` image. Install build dependencies inside it:

```bash
apt-get update
apt-get install -y build-essential cmake pkg-config meson libaio-dev \
    liburing-dev libucx-dev libgtest-dev libgmock-dev autoconf automake libtool
python3 -m pip install --break-system-packages pybind11 meson-python patchelf pyyaml build
```

The current upstream NIXL base requires newer UCX headers than Ubuntu's UCX
1.16. Build a private UCX prefix. The initial validated source is
`24c9aa4184c59b1b1c6cc71e8d920300acf279cf` on `v1.23.x`:

```bash
git clone https://github.com/openucx/ucx.git /workspace/deps/src/ucx
cd /workspace/deps/src/ucx
git checkout 24c9aa4184c59b1b1c6cc71e8d920300acf279cf
./autogen.sh
./configure --prefix=/workspace/deps/ucx --enable-mt \
    --with-cuda=/usr/local/cuda \
    --with-nvcc-gencode='-gencode=arch=compute_80,code=sm_80' \
    --without-java --without-go --disable-numa
make -j16
make install
```

The SM80 target is for the development host's A100 GPUs. Change it for other
GPU architectures. UCX NUMA build support does not implement cache device
placement; clients remain responsible for assigning suitable device paths.

Run from the NIXL source checkout:

```bash
extensions/nixlshard/tools/build-dev.sh
source /workspace/install/nixlshard/env.sh
```

The helper selects only NIXL's UCX and POSIX plugins, installs into the private
prefix, runs the NIXLShard native test suite, and imports the installed Python
module. It builds the pinned Abseil 20250814.1 source into shared libraries at
`/workspace/deps/abseil` and selects that prefix through `PKG_CONFIG_PATH`.
Shared Abseil is required: NIXL's static Meson wrap duplicates synchronization
state across multiple shared objects, causing crashes when repeatedly creating
agents. System packages stay intact. Native builds default to
`/workspace/build/nixl-shared-absl` to avoid previously cached static dependencies.
Do not install the full upstream NIXL wheel with build isolation into this
environment: its build dependencies currently request a different Torch
version from the image's compatible SGLang environment.

Override paths with `NIXLSHARD_BUILD_DIR`, `NIXLSHARD_INSTALL_DIR`,
`NIXLSHARD_UCX_DIR`, and `NIXLSHARD_ABSEIL_DIR`; override parallelism with
`NIXLSHARD_BUILD_JOBS`. Tests honor `TMPDIR`; on the development host use
`TMPDIR=/raid/nixlshard-v2` for disposable SSD-backed debug files.
`env.sh` selects the installed bindings and native libraries explicitly.
Each configure embeds `NIXLSHARD_BUILD_GIT` in the Python binding so callers
can identify the source checkout that produced it. A Git marker identifies
the current commit; uncommitted changes still require a clean-source check
when publishing benchmark results.

# GPU serving correctness smoke

The source SGLang pin `20518d8518375f49be0d14ead7ea474dbc2721d0` requires
Torch 2.13 / CUDA 13 and sglang-kernel 0.4.7. The image's Torch 2.9 / CUDA 12.9
stack supports the native adapter tests but cannot load that newer kernel ABI.
Use an isolated Python environment for serving; preserve the image packages:

```bash
python3 -m venv --system-site-packages /raid/nixlshard-v2/runtime/torch213-cu130
runtime_python=/raid/nixlshard-v2/runtime/torch213-cu130/bin/python
export TMPDIR=/raid/nixlshard-v2/runtime/tmp
export PIP_CACHE_DIR=/raid/nixlshard-v2/runtime/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
"$runtime_python" -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
"$runtime_python" -m pip install --no-deps torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
"$runtime_python" -m pip install --no-deps sglang-kernel==0.4.7 \
    compressed-tensors==0.18.0 \
    transformers==5.12.1 tokenizers==0.22.2 xgrammar==0.2.1
"$runtime_python" -m pip install cuda-python==13.4.1 flashinfer-python==0.6.18 \
    apache-tvm-ffi==0.1.11 cuda-tile==1.6.0rc5 'nvidia-cutlass-dsl[cu13]==4.6.2'
"$runtime_python" -m pip install --no-deps flashinfer-cubin==0.6.18 \
    --index-url https://flashinfer.ai/whl
"$runtime_python" -m pip install --no-deps flashinfer-jit-cache==0.6.18 \
    --index-url https://flashinfer.ai/whl/cu130
source /workspace/install/nixlshard/env.sh
export PYTHONPATH=/workspace/src/sglang/python:$PYTHONPATH
"$runtime_python" extensions/nixlshard/tools/model-smoke.py \
    --model-path /raid/models/qwen2.5-0.5b \
    --model-revision local-qwen2.5-0.5b-sha256-9b54e1ff84127de01ba2b7f2ba86c0cec4bdc3e1c1edbd2d75566168ab5bfdc5
```

The revision above is the frozen local weights, tokenizer, and configuration
manifest digest used on the development host. Supply the corresponding immutable
revision for another model. This recipe inherits the image's other dependencies;
it is a development serving environment rather than a standalone deployment lock.

The helper starts an A100-compatible BF16 server with one outstanding request,
waits for a completed SSD backup, flushes device and host caches, and repeats the
same prompt. It requires storage-hit tokens with zero device/host hit tokens and
identical generated token IDs. Results, metrics, configuration, server logs, and
runtime versions go into a new directory under `/raid/nixlshard-v2/model-smoke`.
The server process group is stopped afterward. Stop serving and native benchmark
processes before reinstalling libraries into the activated prefix.

This smoke selects `page_first_direct` and the `direct` HiCache I/O backend.
The initial `page_first` kernel path crashed in upstream GPU-to-host staging.
A CUDA ABI mismatch is suspected: JIT compilation used the image's CUDA 12.9
toolkit against a CUDA 13 runtime, whose `cudaMemcpyBatchAsync` signature differs.
The direct path uses the matching prebuilt CUDA 13 kernel and passed the
end-to-end SSD replay. Validate a matching CUDA 13 compiler before enabling that
JIT path; the crash's root cause is not yet confirmed. This smoke establishes integration
correctness; it does not establish the target model's TTFT overhead or RDMA speed.
