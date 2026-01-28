// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Configurable Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// Runtime configurable: block_size_m, threads_per_block
// No recompilation needed to tune performance.

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

// Fast sigmoid
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Main kernel with runtime-configurable parameters
// Uses dynamic shared memory for B/C
template <typename input_t, bool HAS_D, bool HAS_Z, bool DT_SOFTPLUS>
__global__ void selective_state_update_kernel_v2(
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
    int dim, int dstate, int nheads, int ngroups, int heads_per_group,
    int block_size_m,
    // Strides
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group) {
  // Dynamic shared memory for B and C
  extern __shared__ float smem[];
  float* smem_B = smem;
  float* smem_C = smem + dstate;

  // Block indices: (batch, head, dim_block)
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int dim_block = blockIdx.z;

  const int tid = threadIdx.x;
  const int dim_start = dim_block * block_size_m;

  // Which group does this head belong to?
  const int group_idx = head_idx / heads_per_group;

  // Load B and C into shared memory
  const int bc_offset = batch_idx * stride_B_batch + group_idx * stride_B_group;
  for (int i = tid; i < dstate; i += blockDim.x) {
    smem_B[i] = static_cast<float>(B[bc_offset + i]);
    smem_C[i] = static_cast<float>(C[bc_offset + i]);
  }
  __syncthreads();

  // Each thread processes one or more dim elements
  for (int local_dim = tid; local_dim < block_size_m; local_dim += blockDim.x) {
    const int dim_idx = dim_start + local_dim;
    if (dim_idx >= dim) continue;

    // Input offset for this (batch, head, dim)
    const int x_offset =
        batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;

    // Load inputs
    float x_val = static_cast<float>(x[x_offset]);
    float dt_val = static_cast<float>(dt[x_offset]);

    // Add dt_bias if present
    if (dt_bias != nullptr) {
      dt_val += static_cast<float>(dt_bias[head_idx * dim + dim_idx]);
    }

    // Apply softplus to dt
    if constexpr (DT_SOFTPLUS) {
      dt_val = softplus(dt_val);
    }

    // Load z if present
    float z_val = 0.0f;
    if constexpr (HAS_Z) {
      z_val = static_cast<float>(z[x_offset]);
    }

    // Load D if present
    float d_val = 0.0f;
    if constexpr (HAS_D) {
      d_val = static_cast<float>(D[head_idx * dim + dim_idx]);
    }

    // State and A base offsets
    const int state_base = batch_idx * stride_state_batch +
                           head_idx * stride_state_head + dim_idx * dstate;
    const int A_base = head_idx * dim * dstate + dim_idx * dstate;

    // Accumulator for output
    float out_acc = 0.0f;

    // Process state elements
    for (int n = 0; n < dstate; n++) {
      float state_val = state[state_base + n];
      float A_val = A[A_base + n];

      // dA = exp(A * dt)
      float dA = expf(A_val * dt_val);

      // dB = B * dt
      float dB = smem_B[n] * dt_val;

      // Update state
      state_val = state_val * dA + dB * x_val;
      state[state_base + n] = state_val;

      // Accumulate output
      out_acc += state_val * smem_C[n];
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
}

// Macro for kernel launch with dynamic shared memory
#define LAUNCH_KERNEL_V2(HAS_D_VAL, HAS_Z_VAL, DT_SOFTPLUS_VAL)                \
  selective_state_update_kernel_v2<scalar_t, HAS_D_VAL, HAS_Z_VAL,             \
                                   DT_SOFTPLUS_VAL>                            \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,         \
          dt_bias_ptr, out_ptr, dim, dstate, nheads, ngroups, heads_per_group, \
          block_size_m, stride_state_batch, stride_state_head, stride_x_batch, \
          stride_x_head, stride_B_batch, stride_B_group)

// Helper template function for kernel dispatch
template <typename scalar_t>
void launch_selective_state_update_kernel(
    float* state_ptr, const float* A_ptr, const torch::Tensor& x,
    const torch::Tensor& dt, const torch::Tensor& B, const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& dt_bias, torch::Tensor& out, bool has_D,
    bool has_z, bool dt_softplus, int batch_size, int nheads, int dim,
    int dstate, int ngroups, int heads_per_group, int block_size_m,
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group, dim3 grid,
    dim3 block, size_t shared_mem_size, cudaStream_t stream) {
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
    LAUNCH_KERNEL_V2(true, true, true);
  } else if (has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V2(true, true, false);
  } else if (has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V2(true, false, true);
  } else if (has_D && !has_z && !dt_softplus) {
    LAUNCH_KERNEL_V2(true, false, false);
  } else if (!has_D && has_z && dt_softplus) {
    LAUNCH_KERNEL_V2(false, true, true);
  } else if (!has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V2(false, true, false);
  } else if (!has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V2(false, false, true);
  } else {
    LAUNCH_KERNEL_V2(false, false, false);
  }
}

#undef LAUNCH_KERNEL_V2

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
  TORCH_CHECK(block_size_m > 0 && block_size_m <= 256,
              "block_size_m must be in range [1, 256]");
  TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024,
              "threads_per_block must be in range [1, 1024]");

  // Get strides
  const int stride_state_batch = state.stride(0);
  const int stride_state_head = state.stride(1);
  const int stride_x_batch = x.stride(0);
  const int stride_x_head = x.stride(1);
  const int stride_B_batch = B.stride(0);
  const int stride_B_group = B.stride(1);

  // Grid: (batch, nheads, dim_blocks) for fine-grained parallelism
  const int dim_blocks = (dim + block_size_m - 1) / block_size_m;
  dim3 grid(batch_size, nheads, dim_blocks);
  dim3 block(threads_per_block);

  // Dynamic shared memory: 2 * dstate floats for B and C
  const size_t shared_mem_size = 2 * dstate * sizeof(float);

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
        dt_softplus, batch_size, nheads, dim, dstate, ngroups, heads_per_group,
        block_size_m, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block,
        shared_mem_size, stream);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_selective_state_update_kernel<at::BFloat16>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, dstate, ngroups, heads_per_group,
        block_size_m, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block,
        shared_mem_size, stream);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    launch_selective_state_update_kernel<float>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, dstate, ngroups, heads_per_group,
        block_size_m, stride_state_batch, stride_state_head, stride_x_batch,
        stride_x_head, stride_B_batch, stride_B_group, grid, block,
        shared_mem_size, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype: ", x.scalar_type());
  }

  // Check for errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mamba
}  // namespace vllm
