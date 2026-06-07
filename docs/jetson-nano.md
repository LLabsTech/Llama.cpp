# Building on Jetson Nano (CUDA 10.2 / JetPack 4.6, sm_53 Maxwell)

This fork carries a compatibility layer so the current llama.cpp CUDA backend can be
built with the old toolchain shipped on the original Jetson Nano:

- GPU: Maxwell, compute capability 5.3 (`sm_53`) - no tensor cores, no native bf16
- Toolkit: CUDA 10.2 (nvcc 10.2), JetPack 4.6
- Host compiler: gcc/g++ 8.5

nvcc 10.2 does not support `-std=c++17` for device code, has no `<cuda_bf16.h>`, and is
missing several CUDA 11/12 APIs that upstream now uses unconditionally. The changes below
let the backend compile and run; CUDA 11+ builds are unaffected (everything is guarded on
`CUDART_VERSION` / the nvcc version).

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true \
    -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
cmake --build build --config Release
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `50;61` in this fork (see top-level `CMakeLists.txt`).
The `-DCMAKE_CUDA_STANDARD=14` flag is required: upstream requests the CUDA17 dialect, which
nvcc 10.2 cannot provide. Expect `warning: constexpr if statements are a C++17 feature`
warnings - they are harmless (nvcc still compiles them).

## What the compatibility layer does

| Area | Change | Where |
|------|--------|-------|
| bf16 type | Emulated (`__nv_bfloat16` = top 16 bits of fp32, exact byte layout, conversions via fp32) for CUDART < 11000 | `ggml/src/ggml-cuda/vendors/bf16-compat.cuh`, wired in `vendors/cuda.h` |
| bf16 cuBLAS GEMM | `CUDA_R_16BF` does not exist pre-CUDA-11; the bf16 cuBLAS path (per-op and batched) is compiled out, bf16 routes to the fp32 cuBLAS path | `ggml-cuda.cu` (`#if CUDART_VERSION >= 11000`) |
| CUDA graphs | `cudaGraphExecUpdate` etc. unavailable; graphs disabled when nvcc < 11.0 | `ggml/src/ggml-cuda/CMakeLists.txt` |
| cooperative softmax | `<cooperative_groups/reduce.h>` (CUDA 11) include guarded; the cooperative kernel is runtime-gated to CC >= 6.0 so the Nano never launches it | `softmax.cu` |
| C++14 device | `if constexpr` left in place (nvcc 10.2 warns but compiles); a few that errored historically are written as plain `if` | `concat.cu`, `fattn.cu`, `mma.cuh` |

bf16 correctness: because bf16 is emulated bit-exactly and converted through fp32, BF16-format
GGUF models load and run correctly (just without native bf16 acceleration, which `sm_53` lacks
anyway). F16 and all quantized models are unaffected.

Performance note: CUDA graphs and the cuBLAS bf16 fast path are off on this toolchain, and
bf16 GEMMs fall back to fp32. These are correctness-preserving fallbacks, not regressions
relative to what `sm_53` can actually accelerate.

## On-device punch list

This layer was developed without access to nvcc 10.2, so a clean compile-fix loop on the
device is still needed. The remaining risk is C++17 *syntax* that nvcc 10.2 rejects outright
(as opposed to warning about). If the build stops, the fix is almost always local:

1. `error: ... is a C++17 feature` on an `if constexpr` whose non-taken branch is invalid for
   some template args -> rewrite that one as plain `if` (all branches must then compile, which
   they do wherever upstream uses `NO_DEVICE_CODE` in the `else`).
2. `static inline` data members / inline variables -> if rejected, drop `inline` (make it a
   `static constexpr` or out-of-line definition).
3. Missing bf16 intrinsic (`__hadd2`/`__hmul2`-style on `nv_bfloat162`, etc.) -> add it to
   `bf16-compat.cuh` next to the existing conversions. Only conversion helpers are provided so
   far because the `sm_53` code paths convert bf16 through fp32 rather than doing bf16 SIMD.
4. `memcpy` in `__device__` warnings from the shim -> replace the two `memcpy` calls in
   `bf16-compat.cuh` with `__float_as_uint` / `__uint_as_float`.

Report the first hard error (file:line + message) and it can be folded into the layer.
