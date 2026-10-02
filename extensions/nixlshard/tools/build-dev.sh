#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nixl_source=$(cd -- "$script_dir/../../.." && pwd)
build_dir=${NIXLSHARD_BUILD_DIR:-/workspace/build/nixl-shared-absl}
install_dir=${NIXLSHARD_INSTALL_DIR:-/workspace/install/nixlshard}
ucx_dir=${NIXLSHARD_UCX_DIR:-/workspace/deps/ucx}
abseil_dir=${NIXLSHARD_ABSEIL_DIR:-/workspace/deps/abseil}
jobs=${NIXLSHARD_BUILD_JOBS:-16}

if [[ ! -f "$ucx_dir/include/ucp/api/ucp.h" ]]; then
    echo "Missing UCX development prefix: $ucx_dir" >&2
    exit 1
fi
if [[ ! -f "$abseil_dir/lib/libabsl_synchronization.so" ]]; then
    if [[ ! -f "$nixl_source/subprojects/abseil-cpp-20250814.1/CMakeLists.txt" ]]; then
        (cd "$nixl_source" && meson subprojects download abseil-cpp)
    fi
    cmake -S "$nixl_source/subprojects/abseil-cpp-20250814.1" \
        -B "$build_dir-abseil" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$abseil_dir" -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=ON -DABSL_BUILD_TESTING=OFF \
        -DABSL_PROPAGATE_CXX_STD=ON -DCMAKE_CXX_STANDARD=20
    cmake --build "$build_dir-abseil" -j "$jobs"
    cmake --install "$build_dir-abseil"
fi
configure_args=()
if [[ -f "$build_dir/meson-private/coredata.dat" ]]; then
    configure_args+=(--reconfigure --clearcache)
fi
# Reconfigure each time so the compiled Git marker follows milestone commits.
# Avoid pip building all of upstream NIXL: that would replace the image's
# compatible Torch dependency and its separately packaged production NIXL.
PKG_CONFIG=/usr/bin/pkg-config PKG_CONFIG_PATH="$abseil_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
    meson setup "${configure_args[@]}" \
    "$build_dir" "$nixl_source" \
    --prefix="$install_dir" --libdir=lib --buildtype=release \
    -Denable_plugins=UCX,POSIX -Ducx_path="$ucx_dir" \
    -Dbuild_nixlshard=true -Dbuild_tests=true -Dbuild_examples=false \
    -Dwith_trace=false -Dnixl_cuda_arch_list=80
ninja -C "$build_dir" -j "$jobs"
meson install -C "$build_dir" --no-rebuild

python_dir=$(meson introspect --installed "$build_dir" | python3 -c '
import json, pathlib, sys
paths = [p for p in json.load(sys.stdin).values() if p.endswith("/nixlshard/__init__.py")]
if len(paths) != 1:
    raise SystemExit("Expected exactly one installed nixlshard package")
print(pathlib.Path(paths[0]).parent.parent)
')
mkdir -p "$install_dir"
python3 - "$install_dir" "$python_dir" "$ucx_dir" "$abseil_dir" <<'PY'
import pathlib
import shlex
import sys
prefix, python_dir, ucx_dir, abseil_dir = sys.argv[1:]
lines = [
    "# Source this file to select the development library and bindings.",
    f"export PATH={shlex.quote(prefix + '/bin')}:\"$PATH\"",
    f"export PYTHONPATH={shlex.quote(python_dir)}${{PYTHONPATH:+:$PYTHONPATH}}",
    f"export LD_LIBRARY_PATH={shlex.quote(prefix + '/lib:' + ucx_dir + '/lib:' + abseil_dir + '/lib')}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}",
    f"export NIXL_PLUGIN_DIR={shlex.quote(prefix + '/lib/plugins')}",
    f"export PKG_CONFIG_PATH={shlex.quote(prefix + '/lib/pkgconfig')}${{PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}}",
]
pathlib.Path(prefix, 'env.sh').write_text('\n'.join(lines) + '\n')
PY
# shellcheck disable=SC1091
source "$install_dir/env.sh"
meson test -C "$build_dir" --no-rebuild --suite nixlshard --print-errorlogs
python3 -c 'import nixlshard; print("NIXLShard development module:", nixlshard.__file__); print("Compiled source commit:", nixlshard.__build_marker__)'
echo "Development environment: source $install_dir/env.sh"
