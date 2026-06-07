#pragma once

// bfloat16 compatibility shim for CUDA toolkits that ship without <cuda_bf16.h>
// (CUDA < 11.0, e.g. CUDA 10.2 / JetPack 4.x on Jetson Nano, sm_53 Maxwell).
//
// bf16 is stored as the upper 16 bits of an IEEE-754 binary32, so reinterpreting
// raw bf16 GGUF tensor bytes through these types is bit-exact. All arithmetic is
// performed in float via the implicit conversion operators - Maxwell has no
// native bf16 path anyway, so this matches what newer hardware emulates.
//
// Only the subset of the CUDA bf16 API used by the ggml CUDA backend is provided.

#include <cstring>

struct __nv_bfloat16 {
    unsigned short __x;

    __host__ __device__ __nv_bfloat16() = default;

    __host__ __device__ __nv_bfloat16(float f) {
        unsigned int u;
        memcpy(&u, &f, sizeof(u));
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            __x = (unsigned short) ((u >> 16) | 0x0040u); // NaN -> quiet NaN
        } else {
            const unsigned int rounding_bias = 0x7fffu + ((u >> 16) & 1u); // round to nearest even
            __x = (unsigned short) ((u + rounding_bias) >> 16);
        }
    }

    __host__ __device__ operator float() const {
        const unsigned int u = ((unsigned int) __x) << 16;
        float f;
        memcpy(&f, &u, sizeof(f));
        return f;
    }
};

struct __align__(4) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

typedef __nv_bfloat16  nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16(float f)    { return nv_bfloat16(f); }
static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16_rn(float f) { return nv_bfloat16(f); }
static __host__ __device__ __forceinline__ float       __bfloat162float(nv_bfloat16 b) { return float(b); }
static __host__ __device__ __forceinline__ nv_bfloat16 __low2bfloat16 (nv_bfloat162 b) { return b.x; }
static __host__ __device__ __forceinline__ nv_bfloat16 __high2bfloat16(nv_bfloat162 b) { return b.y; }

static __host__ __device__ __forceinline__ nv_bfloat162 make_bfloat162(nv_bfloat16 x, nv_bfloat16 y) {
    nv_bfloat162 r;
    r.x = x;
    r.y = y;
    return r;
}

static __host__ __device__ __forceinline__ nv_bfloat162 __float22bfloat162_rn(float2 f) {
    return make_bfloat162(nv_bfloat16(f.x), nv_bfloat16(f.y));
}

static __host__ __device__ __forceinline__ float2 __bfloat1622float2(nv_bfloat162 b) {
    return make_float2(float(b.x), float(b.y));
}
