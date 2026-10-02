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
