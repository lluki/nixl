<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL POSIX Plugin

This backend provides POSIX-compliant I/O operations using either Linux AIO (libaio) by default
Optionally POSIX plugin can also use liburing.

## File registration

`FILE_SEG` descriptors accept either fd-in-`devId` (fd-mode) or a
`"<modes>:<path>"` string in `metaInfo` (path-mode, backend owns the
open/close); see [`src/utils/file/README.md`](../../utils/file/README.md#path-mode-file-registration).

## Worker threads

The backend accepts two optional parameters:

- `thread_num` (default `1`): With the default, path-mode opens and I/O queue
  operations execute directly on the caller. Values greater than one create that
  many workers, each with an independent AIO/io_uring queue. Path-mode opens and
  transfer requests are assigned to workers in round-robin order.
- `thread_unpin` (default `false`, Linux only): Expand each worker's CPU
  affinity to all configured CPUs instead of inheriting the caller's affinity.
  Kernel cpuset/cgroup restrictions still apply. This is useful when the caller
  is CPU- or NUMA-pinned but file I/O should be scheduled freely.

For example:

```cpp
nixl_b_params_t params;
params["thread_num"] = "4";
params["thread_unpin"] = "true";
params["use_uring"] = "true";
```

## Dependencies
To enable Linux AIO support, you need to install the libaio package:

```bash
# Ubuntu/Debian
sudo apt-get install libaio-dev

# RHEL/CentOS/Fedora
sudo dnf install libaio-devel
```

### liburing

liburing support is enabled automatically via the Meson wrap under `subprojects/liburing.wrap` (pinned to WrapDB `liburing_2.14-1`). `meson setup` builds it from source when a system `liburing` is not found via pkg-config, so no manual install is required.

To use liburing with POSIX plugin use params["use_uring"] = "true"

# Running liburing with Docker
Docker by default blocks io_uring syscalls to the host system. These need to be explicitly enabled when running NIXL agents that use the posix plugin in Docker.

## Create a seccomp json file

```bash
$> wget https://github.com/moby/moby/blob/master/profiles/seccomp/default.json

# Add the following to the section, syscalls:names in default.json
# "io_uring_setup",
# "io_uring_enter",
# "io_uring_register",
# "io_uring_sync"

# Run docker with the new seccomp json file

$> docker run --security-opt seccomp=default.json -it --runtime=runc ... <imageid>
```
