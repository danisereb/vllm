// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// Key optimizations:
// 1. Process multiple heads per block (heads sharing same B/C group)
// 2. Use shared memory for B/C tensors (shared across heads in group)
// 3. Vectorized loads for better memory coalescing
// 4. Larger tiles to hide memory latency

#include "selective_state_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

namespace vllm {
namespace mamba {

// Configuration for the optimized kernel
// These are tuned for B200 with typical Mamba2 configurations
constexpr int THREADS_PER_BLOCK = 256;
constexpr int HEADS_PER_GROUP = 8;  // nheads / ngroups for Nemotron-H

// Softplus function
__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : logf(expf(x) + 1.0f);
}

// Fast sigmoid approximation
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Vectorized load helpers
__device__ __forceinline__ float4 load_float4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
  *reinterpret_cast<float4*>(ptr) = val;
}

// Convert bf16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Convert float to bf16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
  return __float2bfloat16(x);
}

// Main kernel: processes one (batch, group) pair per block
// Each block handles HEADS_PER_GROUP heads that share the same B/C
template <typename input_t, int DIM, int DSTATE, bool HAS_D, bool HAS_Z,
          bool DT_SOFTPLUS>
__global__ void selective_state_update_kernel(
    float* __restrict__ state,            // [batch, nheads, dim, dstate]
    const input_t* __restrict__ x,        // [batch, nheads, dim]
    const input_t* __restrict__ dt,       // [batch, nheads, dim]
    const float* __restrict__ A,          // [nheads, dim, dstate]
    const input_t* __restrict__ B,        // [batch, ngroups, dstate]
    const input_t* __restrict__ C,        // [batch, ngroups, dstate]
    const input_t* __restrict__ D,        // [nheads, dim] or nullptr
    const input_t* __restrict__ z,        // [batch, nheads, dim] or nullptr
    const input_t* __restrict__ dt_bias,  // [nheads, dim] or nullptr
    input_t* __restrict__ out,            // [batch, nheads, dim]
    int batch_size, int nheads, int ngroups,
    // Strides (in elements, not bytes)
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group) {
  // Block indices
  const int batch_idx = blockIdx.x;
  const int group_idx = blockIdx.y;

  // Thread index within block
  const int tid = threadIdx.x;

  // Shared memory for B and C (shared across all heads in this group)
  __shared__ float smem_B[DSTATE];
  __shared__ float smem_C[DSTATE];

  // Load B and C into shared memory (all threads participate)
  const int b_offset = batch_idx * stride_B_batch + group_idx * stride_B_group;
  for (int i = tid; i < DSTATE; i += THREADS_PER_BLOCK) {
    smem_B[i] = static_cast<float>(B[b_offset + i]);
    smem_C[i] = static_cast<float>(C[b_offset + i]);
  }
  __syncthreads();

  // Each thread processes multiple (head, dim) pairs
  // Total work: HEADS_PER_GROUP * DIM elements
  const int total_elements = HEADS_PER_GROUP * DIM;
  const int elements_per_thread =
      (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Calculate head range for this group
  const int head_start = group_idx * HEADS_PER_GROUP;

  for (int elem_idx = 0; elem_idx < elements_per_thread; elem_idx++) {
    const int flat_idx = tid + elem_idx * THREADS_PER_BLOCK;
    if (flat_idx >= total_elements) break;

    // Decompose flat index into (local_head, dim)
    const int local_head = flat_idx / DIM;
    const int dim_idx = flat_idx % DIM;
    const int head_idx = head_start + local_head;

    if (head_idx >= nheads) continue;

    // Load x and dt for this element
    const int x_offset =
        batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;
    float x_val = static_cast<float>(x[x_offset]);
    float dt_val = static_cast<float>(dt[x_offset]);

    // Add dt_bias if present
    if (dt_bias != nullptr) {
      dt_val += static_cast<float>(dt_bias[head_idx * DIM + dim_idx]);
    }

    // Apply softplus to dt
    if constexpr (DT_SOFTPLUS) {
      dt_val = softplus(dt_val);
    }

    // Load z if present (for gating)
    float z_val = 0.0f;
    if constexpr (HAS_Z) {
      z_val = static_cast<float>(z[x_offset]);
    }

    // Load D if present (skip connection)
    float d_val = 0.0f;
    if constexpr (HAS_D) {
      d_val = static_cast<float>(D[head_idx * DIM + dim_idx]);
    }

    // State offset for this (batch, head, dim)
    const int state_base = batch_idx * stride_state_batch +
                           head_idx * stride_state_head + dim_idx * DSTATE;

    // A offset for this (head, dim)
    const int A_base = head_idx * DIM * DSTATE + dim_idx * DSTATE;

    // Accumulator for output: out = sum(state * C)
    float out_acc = 0.0f;

// Process state elements
// Unroll for better instruction-level parallelism
#pragma unroll 4
    for (int n = 0; n < DSTATE; n++) {
      // Load state and A
      float state_val = state[state_base + n];
      float A_val = A[A_base + n];

      // Compute dA = exp(A * dt)
      float dA = expf(A_val * dt_val);

      // Compute dB = B * dt
      float dB = smem_B[n] * dt_val;

      // Update state: state = state * dA + dB * x
      state_val = state_val * dA + dB * x_val;

      // Store updated state
      state[state_base + n] = state_val;

      // Accumulate output: out += state * C
      out_acc += state_val * smem_C[n];
    }

    // Add skip connection
    if constexpr (HAS_D) {
      out_acc += x_val * d_val;
    }

    // Apply gating: out *= z * sigmoid(z)
    if constexpr (HAS_Z) {
      out_acc *= z_val * fast_sigmoid(z_val);
    }

    // Store output
    out[x_offset] = static_cast<input_t>(out_acc);
  }
}

// Launcher function
void selective_state_update_cuda(torch::Tensor& state, const torch::Tensor& x,
                                 const torch::Tensor& dt,
                                 const torch::Tensor& A, const torch::Tensor& B,
                                 const torch::Tensor& C,
                                 const c10::optional<torch::Tensor>& D,
                                 const c10::optional<torch::Tensor>& z,
                                 const c10::optional<torch::Tensor>& dt_bias,
                                 torch::Tensor& out, bool dt_softplus) {
  // Get dimensions
  const int batch_size = state.size(0);
  const int nheads = state.size(1);
  const int dim = state.size(2);
  const int dstate = state.size(3);
  const int ngroups = B.size(1);

  // Validate dimensions
  TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
  TORCH_CHECK(nheads / ngroups == HEADS_PER_GROUP,
              "This kernel is optimized for nheads/ngroups == 8");

  // Get strides
  const int stride_state_batch = state.stride(0);
  const int stride_state_head = state.stride(1);
  const int stride_x_batch = x.stride(0);
  const int stride_x_head = x.stride(1);
  const int stride_B_batch = B.stride(0);
  const int stride_B_group = B.stride(1);

  // Grid: (batch, ngroups)
  dim3 grid(batch_size, ngroups);
  dim3 block(THREADS_PER_BLOCK);

  // Get CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Launch kernel based on template parameters
  const bool has_D = D.has_value();
  const bool has_z = z.has_value();

  // Get raw pointers
  float* state_ptr = state.data_ptr<float>();
  const float* A_ptr = A.data_ptr<float>();

  // Dispatch based on input type and flags
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "selective_state_update_cuda", [&] {
        const scalar_t* x_ptr = x.data_ptr<scalar_t>();
        const scalar_t* dt_ptr = dt.data_ptr<scalar_t>();
        const scalar_t* B_ptr = B.data_ptr<scalar_t>();
        const scalar_t* C_ptr = C.data_ptr<scalar_t>();
        const scalar_t* D_ptr =
            has_D ? D.value().data_ptr<scalar_t>() : nullptr;
        const scalar_t* z_ptr =
            has_z ? z.value().data_ptr<scalar_t>() : nullptr;
        const scalar_t* dt_bias_ptr = dt_bias.has_value()
                                          ? dt_bias.value().data_ptr<scalar_t>()
                                          : nullptr;
        scalar_t* out_ptr = out.data_ptr<scalar_t>();

// Dispatch based on dimensions and flags
// For now, support common configurations
#define LAUNCH_KERNEL(DIM_VAL, DSTATE_VAL, HAS_D_VAL, HAS_Z_VAL,          \
                      DT_SOFTPLUS_VAL)                                    \
  selective_state_update_kernel<scalar_t, DIM_VAL, DSTATE_VAL, HAS_D_VAL, \
                                HAS_Z_VAL, DT_SOFTPLUS_VAL>               \
      <<<grid, block, 0, stream>>>(                                       \
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,    \
          dt_bias_ptr, out_ptr, batch_size, nheads, ngroups,              \
          stride_state_batch, stride_state_head, stride_x_batch,          \
          stride_x_head, stride_B_batch, stride_B_group);

        // Common case: dim=64, dstate=128 (Nemotron-H)
        if (dim == 64 && dstate == 128) {
          if (has_D && has_z && dt_softplus) {
            LAUNCH_KERNEL(64, 128, true, true, true);
          } else if (has_D && has_z && !dt_softplus) {
            LAUNCH_KERNEL(64, 128, true, true, false);
          } else if (has_D && !has_z && dt_softplus) {
            LAUNCH_KERNEL(64, 128, true, false, true);
          } else if (!has_D && has_z && dt_softplus) {
            LAUNCH_KERNEL(64, 128, false, true, true);
          } else if (!has_D && !has_z && dt_softplus) {
            LAUNCH_KERNEL(64, 128, false, false, true);
          } else {
            LAUNCH_KERNEL(64, 128, false, false, false);
          }
        } else {
          TORCH_CHECK(false, "Unsupported dim/dstate combination: ", dim, "/",
                      dstate,
                      ". This kernel is optimized for dim=64, dstate=128.");
        }

#undef LAUNCH_KERNEL
      });

  // Check for errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mamba
}  // namespace vllm
