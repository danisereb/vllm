// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// V6 kernel: Thread-per-element approach (like Triton)
// Key optimizations:
// 1. One thread per (batch, head, dim) element - simpler, no warp reduction
// 2. Vectorized float4 loads for state and A tensors
// 3. Sequential loop over dstate dimension with unrolling

#include "selective_state_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace vllm {
namespace mamba {

// Softplus function
__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : logf(expf(x) + 1.0f);
}

// Fast sigmoid using intrinsic
__device__ __forceinline__ float fast_sigmoid(float x) {
  return __frcp_rn(1.0f + expf(-x));
}

// Convert input type to float
template <typename T>
__device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float to_float(at::Half val) {
  return __half2float(*reinterpret_cast<const __half*>(&val));
}

template <>
__device__ __forceinline__ float to_float(at::BFloat16 val) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
}

// V6 Kernel: Thread-per-element like Triton
// Each thread processes one (batch, head, dim) element, looping over dstate
// No warp shuffle needed - simpler and matches Triton's approach
// Grid: (num_elements / threads_per_block)
// This gives better occupancy and hides memory latency through more threads
template <typename input_t, int DSTATE, bool HAS_D, bool HAS_Z,
          bool DT_SOFTPLUS>
__global__ void selective_state_update_kernel_v6(
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
    int total_elements,                   // batch * nheads * dim
    int dim, int nheads, int ngroups, int heads_per_group,
    // Strides
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group) {
  // Each thread handles one (batch, head, dim) element
  const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= total_elements) return;

  // Decompose element index into (batch, head, dim)
  const int dim_idx = elem_idx % dim;
  const int head_idx = (elem_idx / dim) % nheads;
  const int batch_idx = elem_idx / (dim * nheads);
  const int group_idx = head_idx / heads_per_group;

  // Input offset for this (batch, head, dim)
  const int x_offset =
      batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;

  // Load scalar inputs
  float x_val = to_float(x[x_offset]);
  float dt_val = to_float(dt[x_offset]);

  // Add dt_bias if present
  if (dt_bias != nullptr) {
    dt_val += to_float(dt_bias[head_idx * dim + dim_idx]);
  }

  // Apply softplus to dt
  if constexpr (DT_SOFTPLUS) {
    dt_val = softplus(dt_val);
  }

  // Load z if present
  float z_val = 0.0f;
  if constexpr (HAS_Z) {
    z_val = to_float(z[x_offset]);
  }

  // Load D if present
  float d_val = 0.0f;
  if constexpr (HAS_D) {
    d_val = to_float(D[head_idx * dim + dim_idx]);
  }

  // State and A base offsets
  const int state_base = batch_idx * stride_state_batch +
                         head_idx * stride_state_head + dim_idx * DSTATE;
  const int A_base = head_idx * dim * DSTATE + dim_idx * DSTATE;
  const int bc_offset = batch_idx * stride_B_batch + group_idx * stride_B_group;

  // Process dstate elements sequentially using float4 loads
  float out_acc = 0.0f;
  float4* state_ptr = reinterpret_cast<float4*>(state + state_base);
  const float4* A_ptr = reinterpret_cast<const float4*>(A + A_base);

  // 128 elements / 4 per float4 = 32 iterations
#pragma unroll 8
  for (int n = 0; n < DSTATE / 4; n++) {
    float4 state_f4 = state_ptr[n];
    float4 A_f4 = A_ptr[n];

    // Load B and C for these 4 dstate positions
    float B0 = to_float(B[bc_offset + n * 4 + 0]);
    float B1 = to_float(B[bc_offset + n * 4 + 1]);
    float B2 = to_float(B[bc_offset + n * 4 + 2]);
    float B3 = to_float(B[bc_offset + n * 4 + 3]);

    float C0 = to_float(C[bc_offset + n * 4 + 0]);
    float C1 = to_float(C[bc_offset + n * 4 + 1]);
    float C2 = to_float(C[bc_offset + n * 4 + 2]);
    float C3 = to_float(C[bc_offset + n * 4 + 3]);

    // Compute dA and update state
    float dA0 = expf(A_f4.x * dt_val);
    float dA1 = expf(A_f4.y * dt_val);
    float dA2 = expf(A_f4.z * dt_val);
    float dA3 = expf(A_f4.w * dt_val);

    float new_state0 = state_f4.x * dA0 + B0 * dt_val * x_val;
    float new_state1 = state_f4.y * dA1 + B1 * dt_val * x_val;
    float new_state2 = state_f4.z * dA2 + B2 * dt_val * x_val;
    float new_state3 = state_f4.w * dA3 + B3 * dt_val * x_val;

    // Store updated state
    state_ptr[n] = make_float4(new_state0, new_state1, new_state2, new_state3);

    // Accumulate output
    out_acc +=
        new_state0 * C0 + new_state1 * C1 + new_state2 * C2 + new_state3 * C3;
  }

  // Add skip connection
  if constexpr (HAS_D) {
    out_acc += x_val * d_val;
  }

  // Apply gating
  if constexpr (HAS_Z) {
    out_acc *= z_val * fast_sigmoid(z_val);
  }

  // Store output
  out[x_offset] = static_cast<input_t>(out_acc);
}

// Macro for kernel launch (v6 uses 1D grid, one thread per element)
#define LAUNCH_KERNEL_V6(HAS_D_VAL, HAS_Z_VAL, DT_SOFTPLUS_VAL)         \
  selective_state_update_kernel_v6<scalar_t, 128, HAS_D_VAL, HAS_Z_VAL, \
                                   DT_SOFTPLUS_VAL>                     \
      <<<grid, block, 0, stream>>>(                                     \
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,  \
          dt_bias_ptr, out_ptr, total_elements, dim, nheads, ngroups,   \
          heads_per_group, stride_state_batch, stride_state_head,       \
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group)

// Helper template function for kernel dispatch
template <typename scalar_t>
void launch_selective_state_update_kernel(
    float* state_ptr, const float* A_ptr, const torch::Tensor& x,
    const torch::Tensor& dt, const torch::Tensor& B, const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& dt_bias, torch::Tensor& out, bool has_D,
    bool has_z, bool dt_softplus, int batch_size, int nheads, int dim,
    int ngroups, int heads_per_group, int total_elements,
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group, dim3 grid,
    dim3 block, cudaStream_t stream) {
  const scalar_t* x_ptr = x.data_ptr<scalar_t>();
  const scalar_t* dt_ptr = dt.data_ptr<scalar_t>();
  const scalar_t* B_ptr = B.data_ptr<scalar_t>();
  const scalar_t* C_ptr = C.data_ptr<scalar_t>();
  const scalar_t* D_ptr = has_D ? D.value().data_ptr<scalar_t>() : nullptr;
  const scalar_t* z_ptr = has_z ? z.value().data_ptr<scalar_t>() : nullptr;
  const scalar_t* dt_bias_ptr =
      dt_bias.has_value() ? dt_bias.value().data_ptr<scalar_t>() : nullptr;
  scalar_t* out_ptr = out.data_ptr<scalar_t>();

  // Dispatch on boolean flags (8 combinations)
  if (has_D && has_z && dt_softplus) {
    LAUNCH_KERNEL_V6(true, true, true);
  } else if (has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V6(true, true, false);
  } else if (has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V6(true, false, true);
  } else if (has_D && !has_z && !dt_softplus) {
    LAUNCH_KERNEL_V6(true, false, false);
  } else if (!has_D && has_z && dt_softplus) {
    LAUNCH_KERNEL_V6(false, true, true);
  } else if (!has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V6(false, true, false);
  } else if (!has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V6(false, false, true);
  } else {
    LAUNCH_KERNEL_V6(false, false, false);
  }
}

#undef LAUNCH_KERNEL_V6

// Launcher function with configurable parameters
void selective_state_update_cuda(
    torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& dt_bias, torch::Tensor& out,
    bool dt_softplus, int64_t block_size_m, int64_t threads_per_block) {
  // Get dimensions
  const int batch_size = state.size(0);
  const int nheads = state.size(1);
  const int dim = state.size(2);
  const int dstate = state.size(3);
  const int ngroups = B.size(1);
  const int heads_per_group = nheads / ngroups;

  // Validate dimensions
  TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
  TORCH_CHECK(dstate == 128, "This kernel requires dstate=128, got ", dstate);
  // V6 allows up to 1024 threads (no __syncthreads__ needed)
  TORCH_CHECK(threads_per_block >= 32 && threads_per_block <= 1024,
              "threads_per_block must be between 32 and 1024, got ",
              threads_per_block);
  TORCH_CHECK(threads_per_block % 32 == 0,
              "threads_per_block must be a multiple of 32, got ",
              threads_per_block);

  // Get strides
  const int stride_state_batch = state.stride(0);
  const int stride_state_head = state.stride(1);
  const int stride_x_batch = x.stride(0);
  const int stride_x_head = x.stride(1);
  const int stride_B_batch = B.stride(0);
  const int stride_B_group = B.stride(1);

  // V6: Simple 1D grid, one thread per element
  const int total_elements = batch_size * nheads * dim;
  const int threads = threads_per_block;
  const int num_blocks = (total_elements + threads - 1) / threads;

  dim3 grid(num_blocks);
  dim3 block(threads);

  // Get CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Launch kernel based on template parameters
  const bool has_D = D.has_value();
  const bool has_z = z.has_value();

  // Get raw pointers
  float* state_ptr = state.data_ptr<float>();
  const float* A_ptr = A.data_ptr<float>();

  // Dispatch based on input type
  if (x.scalar_type() == at::ScalarType::Half) {
    launch_selective_state_update_kernel<at::Half>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        total_elements, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block, stream);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_selective_state_update_kernel<at::BFloat16>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        total_elements, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block, stream);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    launch_selective_state_update_kernel<float>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        total_elements, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype: ", x.scalar_type());
  }

  // Check for errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mamba
}  // namespace vllm
