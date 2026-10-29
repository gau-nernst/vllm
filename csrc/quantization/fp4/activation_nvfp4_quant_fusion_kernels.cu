/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/stable/tensor.h>
#include "torch_utils.h"
#include "dispatch_utils.h"
#include "../../cuda_vec_utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"
#include "launch_bounds_utils.h"

// Define before including nvfp4_utils.cuh so the header
// can use this macro during compilation.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"

namespace vllm {

#if CVT_FP4_PACK16
// silu(gate) * up rounded to Type, then NVFP4 (UE4M3 scale) quantization of one
// 16-element chunk: bit-identical to compute_silu_mul + cvt_warp_fp16_to_fp4,
// in ~35 % fewer instructions (the kernel is issue-bound with those):
// - __expf(-x) is ex2.approx(x * -log2(e)) plus a fixup for denormal
//   results. With .ftz those results become 0, and 1 + e rounds to 1 either
//   way. __fdividef(x, d) is x * rcp.approx(d).
// - Element pairs use packed f32x2 mul/add (same rounding as the scalar ops).
// - Elements stay in 32-bit words of two values, so bf16 unpacks with one
//   shift or mask and the rounded silu * up words feed the abs-max directly.
template <class Type>
__device__ __forceinline__ u32x2 silu_mul_cvt_16_to_fp4(
    PackedVec<Type, true> const& gate_vec, PackedVec<Type, true> const& up_vec,
    float SFScaleVal, uint8_t* SFout) {
  using packed_t = typename PackedTypeConverter<Type>::Type;
  static_assert(sizeof(packed_t) == sizeof(uint32_t));
  auto to_f2 = [](packed_t v) {
    if constexpr (std::is_same_v<Type, __nv_bfloat16>) {
      uint32_t const w = reinterpret_cast<uint32_t const&>(v);
      return make_float2(__uint_as_float(w << 16),
                         __uint_as_float(w & 0xffff0000u));
    } else {
      return cast_to_float2(v);
    }
  };
  auto mul2 = [](float2 a, float2 b) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __fmul2_rn(a, b);
  #else
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
  #endif
  };
  auto ex2_ftz = [](float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  };
  auto rcp_ftz = [](float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  };

  float const neg_log2e = -1.4426950216293334961f;  // __expf's log2(e)
  packed_t act[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 const x = to_f2(gate_vec.elts[i]);
    float2 const t = mul2(x, make_float2(neg_log2e, neg_log2e));
    float2 d = make_float2(ex2_ftz(t.x), ex2_ftz(t.y));
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    d = __fadd2_rn(d, make_float2(1.f, 1.f));
  #else
    d = make_float2(d.x + 1.f, d.y + 1.f);
  #endif
    float2 const silu = mul2(x, make_float2(rcp_ftz(d.x), rcp_ftz(d.y)));
    act[i] = cast_to_packed<packed_t>(mul2(silu, to_f2(up_vec.elts[i])));
  }

  // Scale factor and quantization as in cvt_warp_fp16_to_fp4.
  packed_t localMax = __habs2(act[0]);
  #pragma unroll
  for (int i = 1; i < 8; ++i) localMax = __hmax2(localMax, __habs2(act[i]));
  float const vecMax = float(__hmax(localMax.x, localMax.y));
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  __nv_fp8_e4m3 const sf = __nv_fp8_e4m3(SFValue);
  if (SFout) *SFout = reinterpret_cast<uint8_t const&>(sf);
  SFValue = float(sf);
  // reciprocal_approximate_ftz is asm volatile, so this stays after the input
  // loads; hoisting it to kernel start stalls every thread on the SFScale load
  // before it issues its first input load (+0.1-0.3 us at small M).
  float const outputScale =
      SFValue != 0.0f
          ? rcp_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
          : 0.0f;
  float2 scaled[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    scaled[i] = mul2(to_f2(act[i]), make_float2(outputScale, outputScale));
  return fp32_vec16_to_e2m1(scaled);
}
#endif

// Use UE4M3 by default.
//
// Work item i = row * num_packed_cols + col is one CVT_FP4_ELTS_PER_THREAD
// chunk of a row. Threads walk all numRows * num_packed_cols items with a flat
// grid-stride loop, so every launched thread has work for any shape (a rows x
// column-blocks grid idles the threads of a partial last column block). The
// host launches at most one full wave, so the initial index and the stride fit
// in 32 bits; the running index is 64-bit. A second loop zeroes the scale
// padding, as scaled_fp4_quant does.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(256, VLLM_BLOCKS_PER_SM(256))
    silu_mul_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols,
                             int32_t num_packed_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Programmatic dependent launch, as in cvt_fp16_to_fp4: wait for the
  // producer of `in`, then let the consumer (which synchronizes on its own)
  // launch while this grid is still running.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = (SFScale == nullptr) ? 1.0f : SFScale[0];

  int64_t const numItems = int64_t(numRows) * num_packed_cols;
  uint32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.x * blockDim.x;
  // Decompose once, then advance (row, col) by the stride with a carry.
  int32_t rowIdx = tid / num_packed_cols;
  int32_t colIdx = tid % num_packed_cols;
  int32_t const rowStep = stride / num_packed_cols;
  int32_t const colStep = stride % num_packed_cols;

  for (int64_t idx = tid; idx < numItems; idx += stride) {
    Type const* gate =
        in + int64_t(rowIdx) * 2 * numCols + colIdx * CVT_FP4_ELTS_PER_THREAD;
    PackedVec in_vec;
    PackedVec in_vec2;
    // Every loop item is in range: the predicates fold away, leaving plain
    // .cg loads.
    if constexpr (CVT_FP4_PACK16) {
      ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec), gate, true);
      ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec2), gate + numCols,
                       true);
    } else {
      ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec), gate, true);
      ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec2), gate + numCols, true);
    }

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx, colIdx, numKTiles, SFout);

    auto const out_val = [&] {
#if CVT_FP4_PACK16
      if constexpr (!UE8M0_SF) {
        return silu_mul_cvt_16_to_fp4<Type>(in_vec, in_vec2, SFScaleVal,
                                            sf_out);
      }
#endif
      PackedVec out_silu_mul = compute_silu_mul<Type>(in_vec, in_vec2);
      return cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
          out_silu_mul, SFScaleVal, sf_out);
    }();

    // The output is [numRows, num_packed_cols] chunks, row-major: item idx.
    if constexpr (CVT_FP4_PACK16) {
      reinterpret_cast<uint64_t*>(out)[idx] =
          (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
    } else {
      out[idx] = out_val;
    }

    rowIdx += rowStep;
    colIdx += colStep;
    if (colIdx >= num_packed_cols) {
      colIdx -= num_packed_cols;
      rowIdx++;
    }
  }

  // Zero the scale padding (the swizzled scale tensor may come from
  // torch.empty): scale columns numCols/16 .. 4 * numKTiles of the valid rows,
  // and rows numRows .. round_up(numRows, 128). Padding items are numbered from
  // the end of the grid, so at small M they go to threads without a data item.
  int32_t const numSfCols = numCols / CVT_FP4_SF_VEC_SIZE;
  int32_t const padCols = numKTiles * 4 - numSfCols;
  int64_t const numPadColItems = int64_t(numRows) * padCols;
  // One item per (k tile, row % 32) of the last, partial 128-row tile: 16
  // contiguous scale bytes, rows r + 32 * i (i = 0..3) x 4 scale columns.
  int32_t const firstPadRow = numRows % 128;  // within the last tile; 0: none
  int32_t const numPadRowItems = firstPadRow ? 32 * numKTiles : 0;
  uint8_t* const sf_bytes = reinterpret_cast<uint8_t*>(SFout);
  for (int64_t p = stride - 1 - tid; p < numPadColItems + numPadRowItems;
       p += stride) {
    if (p < numPadColItems) {
      int32_t const row = int32_t(p / padCols);
      int32_t const col = numSfCols + int32_t(p % padCols);
      *cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(row, col, numKTiles,
                                                       SFout) = 0;
    } else {
      int32_t const q = int32_t(p - numPadColItems);
      int32_t const outer = q & 31;
      uint8_t* const base =
          sf_bytes + ((int64_t(numRows / 128) * numKTiles + (q >> 5)) << 9) +
          outer * 16;
      if (outer >= firstPadRow) {
        *reinterpret_cast<uint4*>(base) = make_uint4(0, 0, 0, 0);
      } else {
#pragma unroll
        for (int i = 1; i < 4; ++i) {
          if (outer + 32 * i >= firstPadRow) {
            reinterpret_cast<uint32_t*>(base)[i] = 0;
          }
        }
      }
    }
  }
}

}  // namespace vllm

void silu_and_mul_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output,  // [..., d]
    torch::stable::Tensor& output_sf,
    torch::stable::Tensor& input,  // [..., 2 * d]
    torch::stable::Tensor& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1) / 2;

  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto stream = get_current_cuda_stream(input.get_device_index());

  int num_packed_cols = int(n / CVT_FP4_ELTS_PER_THREAD);
  int64_t const num_items = int64_t(m) * num_packed_cols;
  if (num_items == 0) return;
  constexpr int kBlockSize = 256;
  // Scale-padding items (see the kernel): padded scale columns of the valid
  // rows, and 16-byte groups of the padded rows.
  int const num_k_tiles = vllm::div_round_up(n, 64);
  int64_t const num_pad_items =
      int64_t(m) * (num_k_tiles * 4 - n / CVT_FP4_SF_VEC_SIZE) +
      (m % 128 ? 32 * num_k_tiles : 0);

  // PDL lets this grid launch while the producer GEMM is still running. That
  // pays for small inputs, but above ~4096 items per SM the early blocks slow
  // the chain down (measured in the gate_up -> act+quant -> down chain).
  bool const use_pdl = num_items <= int64_t(multiProcessorCount) * 4096;

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        auto kernel = vllm::silu_mul_cvt_fp16_to_fp4<cuda_type>;
        static int const blocks_per_sm = [&] {
          int blocks = 0;
          STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, kernel, kBlockSize, 0));
          return std::max(blocks, 1);
        }();

        // At most one full wave; larger inputs loop. (Splitting the items
        // evenly over fewer blocks was slower just above one wave.)
        int64_t const num_blocks = std::min(
            vllm::div_round_up<int64_t>(num_items + num_pad_items, kBlockSize),
            int64_t(multiProcessorCount) * blocks_per_sm);

        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(num_blocks);
        config.blockDim = dim3(kBlockSize);
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attrs;
        config.numAttrs = use_pdl ? 1 : 0;
        cudaLaunchKernelEx(&config, kernel, m, n, num_packed_cols, input_ptr,
                           input_sf_ptr,
                           reinterpret_cast<uint32_t*>(output_ptr),
                           reinterpret_cast<uint32_t*>(sf_out));
      });
}
