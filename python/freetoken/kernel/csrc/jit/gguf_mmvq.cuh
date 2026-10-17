#pragma once
#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <limits>
#include <stdexcept>
#include "../third_party/strata_mmvq/src/kernels/cuda/native_mmvq.cu"

namespace freetoken::gguf_native {
// Graph dependency edges belong to FreeToken; this module uses ordinary launches.
bool& pdl_scope() { static thread_local bool enabled = false; return enabled; }
bool pdl_supported() { return false; }
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }
size_t iq_row_bytes(int, int64_t) noexcept { return 0; }
void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) {
    throw std::invalid_argument("FreeToken native MMVQ excludes external IQ dispatch");
}
}

struct GGUFNativeMMVQ {
    static void independent(const void* a, size_t na, const void* b, size_t nb) {
        using namespace host;
        const auto pa = reinterpret_cast<uintptr_t>(a);
        const auto pb = reinterpret_cast<uintptr_t>(b);
        RuntimeCheck(pa + na <= pb || pb + nb <= pa, "GGUF buffers must not overlap");
    }

    static void quantize(tvm::ffi::TensorView x, tvm::ffi::TensorView q, int64_t stream) {
        using namespace host;
        auto tokens = SymbolicSize{"tokens"};
        auto width = SymbolicSize{"width"};
        auto device = SymbolicDevice{};
        TensorMatcher({tokens, width}).with_dtype<float>().with_device<kDLCUDA>(device).verify(x);
        const auto t = tokens.unwrap(), k = width.unwrap();
        RuntimeCheck(t > 0 && k > 0 && k % 32 == 0, "Q8_1 requires positive rows and width divisible by 32");
        RuntimeCheck(t <= std::numeric_limits<int>::max() / k, "Q8_1 flat input exceeds native index width");
        TensorMatcher({t, k / 32, 36}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(q);
        independent(x.data_ptr(), t * k * sizeof(float), q.data_ptr(), t * (k / 32) * 36);
        // Flattening keeps every 32-element block inside its original token.
        freetoken::gguf_native::native_quantize_q8_1(static_cast<const float*>(x.data_ptr()), q.data_ptr(),
            static_cast<int>(t * k), 1, reinterpret_cast<void*>(stream));
    }

    static void project(tvm::ffi::TensorView weights, tvm::ffi::TensorView q,
                        tvm::ffi::TensorView y, int64_t type, int64_t width, int64_t stream) {
        using namespace host;
        int64_t block = 0, bytes = 0;
        switch (type) {
            case 2: case 20: block = 32; bytes = 18; break;
            case 6: block = 32; bytes = 22; break;
            case 8: block = 32; bytes = 34; break;
            case 11: block = 256; bytes = 110; break;
            case 12: block = 256; bytes = 144; break;
            case 13: block = 256; bytes = 176; break;
            case 14: block = 256; bytes = 210; break;
            case 23: block = 256; bytes = 136; break;
            case 42: block = 64; bytes = 18; break;
            default: throw std::invalid_argument("Unsupported FreeToken native GGUF matrix type");
        }
        RuntimeCheck(width > 0 && width % block == 0 && width <= std::numeric_limits<int>::max() / 8,
            "Native GGUF input width differs from its format");
        auto tokens = SymbolicSize{"tokens"};
        auto rows = SymbolicSize{"rows"};
        auto device = SymbolicDevice{};
        TensorMatcher({tokens, rows}).with_dtype<float>().with_device<kDLCUDA>(device).verify(y);
        const auto t = tokens.unwrap(), n = rows.unwrap();
        RuntimeCheck(t > 0 && n > 0 && n <= std::numeric_limits<int>::max(), "Native GGUF output shape invalid");
        TensorMatcher({t, width / 32, 36}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(q);
        const auto weight_bytes = n * (width / block) * bytes;
        TensorMatcher({weight_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(weights);
        independent(weights.data_ptr(), weight_bytes, y.data_ptr(), t * n * sizeof(float));
        independent(q.data_ptr(), t * (width / 32) * 36, y.data_ptr(), t * n * sizeof(float));
        const auto* input = static_cast<const uint8_t*>(q.data_ptr());
        auto* output = static_cast<float*>(y.data_ptr());
        for (int64_t first = 0; first < t; first += 8) {
            freetoken::gguf_native::native_mmvq(static_cast<int>(type), weights.data_ptr(),
                input + first * (width / 32) * 36, output + first * n,
                static_cast<int>(width), static_cast<int>(n), static_cast<int>(std::min<int64_t>(8, t - first)),
                reinterpret_cast<void*>(stream));
        }
    }
};

// Small batches reuse the Strata arithmetic and each column reduction order.
struct GGUFInterleavedMMVQ {
    static void interleave(tvm::ffi::TensorView q, tvm::ffi::TensorView il, int64_t stream) {
        using namespace host;
        auto tokens = SymbolicSize{"tokens"}; auto blocks = SymbolicSize{"blocks"};
        auto device = SymbolicDevice{};
        TensorMatcher({tokens, blocks, 36}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(q);
        const auto t = tokens.unwrap(), k = blocks.unwrap() * 32;
        RuntimeCheck(t >= 2 && t <= 4 && k > 0, "Interleaved Q8_1 needs two to four columns");
        const auto bytes = static_cast<int64_t>(freetoken::gguf_native::native_q8_1_il_bytes(k, t));
        TensorMatcher({bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(il);
        GGUFNativeMMVQ::independent(q.data_ptr(), t * blocks.unwrap() * 36, il.data_ptr(), bytes);
        freetoken::gguf_native::native_q8_1_interleave(q.data_ptr(), il.data_ptr(), k, t,
            reinterpret_cast<void*>(stream));
    }
    static void project(tvm::ffi::TensorView weights, tvm::ffi::TensorView q,
                        tvm::ffi::TensorView il, tvm::ffi::TensorView y,
                        int64_t type, int64_t width, int64_t stream) {
        using namespace host;
        RuntimeCheck(type == 12 || type == 13 || type == 14 || type == 23, "Unsupported interleaved weight type");
        RuntimeCheck(width > 0 && width % 256 == 0 && width <= std::numeric_limits<int>::max()/8,
                     "Interleaved input geometry invalid");
        auto tokens=SymbolicSize{"tokens"};auto rows=SymbolicSize{"rows"};auto device=SymbolicDevice{};
        TensorMatcher({tokens, rows}).with_dtype<float>().with_device<kDLCUDA>(device).verify(y);
        const auto t=tokens.unwrap(), n=rows.unwrap();
        RuntimeCheck(t>=2 && t<=4 && n>0 && n<=std::numeric_limits<int>::max(), "Interleaved output geometry invalid");
        const int block_bytes=type==12?144:type==13?176:type==14?210:136;
        const auto weight_bytes=n*(width/256)*block_bytes;
        const auto il_bytes=static_cast<int64_t>(freetoken::gguf_native::native_q8_1_il_bytes(width,t));
        TensorMatcher({weight_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(weights);
        TensorMatcher({t,width/32,36}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(q);
        TensorMatcher({il_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(il);
        GGUFNativeMMVQ::independent(weights.data_ptr(),weight_bytes,y.data_ptr(),t*n*sizeof(float));
        GGUFNativeMMVQ::independent(q.data_ptr(),t*(width/32)*36,y.data_ptr(),t*n*sizeof(float));
        GGUFNativeMMVQ::independent(il.data_ptr(),il_bytes,y.data_ptr(),t*n*sizeof(float));
        freetoken::gguf_native::native_mmvq_il(type,weights.data_ptr(),q.data_ptr(),il.data_ptr(),
            static_cast<float*>(y.data_ptr()),width,n,t,reinterpret_cast<void*>(stream));
    }
};


// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "strata/kernels/dp4a.hpp"

namespace freetoken_draft_q5 {
constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = FREETOKEN_GGUF_DP4A(v0i, u[2 * i], FREETOKEN_GGUF_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = FREETOKEN_GGUF_DP4A(0x01010101, u[2 * i], FREETOKEN_GGUF_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void mapped_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out, const int64_t* __restrict__ row_ids) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row_ids[row0 + i]) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}


}

struct GGUFIndexedDraftQ5 {
    // This prototype accepts the validated full Flash Next Q5 head and a checked CPU-loaded row map.
    static void project(tvm::ffi::TensorView weights, tvm::ffi::TensorView q,
                        tvm::ffi::TensorView row_ids, tvm::ffi::TensorView y, int64_t stream) {
        using namespace host;
        auto rows = SymbolicSize{"rows"};
        auto device = SymbolicDevice{};
        TensorMatcher({1, rows}).with_dtype<float>().with_device<kDLCUDA>(device).verify(y);
        const auto n = rows.unwrap();
        RuntimeCheck(n > 0 && n <= 248320, "Draft subset output size invalid");
        TensorMatcher({248320 * 1760}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(weights);
        TensorMatcher({1, 80, 36}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(q);
        TensorMatcher({n}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(row_ids);
        freetoken_draft_q5::mapped_q5_k_mmvq_kernel<false><<<unsigned(n), dim3(32,4),0,reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const freetoken_draft_q5::Q5KBlock*>(weights.data_ptr()),
            static_cast<const freetoken_draft_q5::Q81Block*>(q.data_ptr()),
            static_cast<float*>(y.data_ptr()),2560,static_cast<int>(n),static_cast<const int64_t*>(row_ids.data_ptr()));
        const auto error = cudaGetLastError();
        RuntimeCheck(error == cudaSuccess, cudaGetErrorString(error));
    }
};
