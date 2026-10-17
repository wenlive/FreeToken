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
