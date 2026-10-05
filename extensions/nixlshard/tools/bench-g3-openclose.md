**Agent written**

# Synthetic authoritative G3 lifecycle and read benchmark

`bench-g3-openclose.cpp` is the exact source used for the matched GB200 G3 component comparison (SHA256 `171ca09d6c071e2624f03acecf584dc9c85d94e2ea3320f166ff99c86169e219`). It uses POSIX/Linux AIO and an explicitly registered aligned destination, independently of SGLang or network transfers.

Choose an unused debug-file path in an existing directory. The benchmark creates a 128 GiB cache, seeds 128 synthetic 16 MiB values, and performs five cycles. Each cycle reads 2 GiB in sixteen eight-page groups, with two 8 MiB destination spans per page. Full-value comparisons run outside read timing. The first open initializes new media; the next four opens recover the same CLEAN media. CLEAN close is measured after registration retirement. These component measurements are separate from model TTFT.

Build and run on the GB200 owner using the isolated prefix:

```sh
g3_prefix=/workspace/install/nixlshard-g3-opt-128a65a
source "$g3_prefix/env.sh"
g++ -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter \
  -I"$g3_prefix/include" bench-g3-openclose.cpp \
  -L"$g3_prefix/lib" -Wl,-rpath,"$g3_prefix/lib" \
  -Wl,-rpath-link,/workspace/deps/abseil/lib \
  -lnixlshard -lnixl -lnixl_build -laio -ldl -pthread \
  -o /workspace/tools/bench-g3-openclose-128a65a
unset UCX_NET_DEVICES
export UCX_TLS=tcp,self,cuda_copy
numactl --cpunodebind=0 --membind=0 \
  /workspace/tools/bench-g3-openclose-128a65a \
  /scratch/nixlshard-v2/g3-unused-debug-file \
  /scratch/nixlshard-v2/g3-openclose-results.json
```

The baseline used `/workspace/install/nixlshard-g3-3310a0c` with the identical source, compiler flags, CPU/memory binding, hardware and geometry, and its own fresh file. Record each compiled native marker and preserve all five raw rows. Compare five read/close samples; compare the four CLEAN reopens separately from fresh initialization. Expected successful replay totals are 2,147,483,648 payload bytes, 524,288 allocation-metadata bytes, and zero copy bytes per cycle. Source and build products are retained in Git/RAID; approved artifact exports contain results, logs and checksums.
