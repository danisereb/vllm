// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// Key optimization: Parallelize over dstate using warp-level cooperation.
// Each warp processes one (batch, head, dim) element, with threads
// cooperating to process the 128 dstate values in parallel.

#include "selective_state_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace vllm {
namespace mamba {

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Softplus function
__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : logf(expf(x) + 1.0f);
}

// Fast sigmoid
__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Main kernel: Each warp processes one (batch, head, dim) element
// Threads within the warp cooperate on the dstate dimension
// Grid: (batch * nheads * dim / WARPS_PER_BLOCK, 1, 1)
// Block: (32 * WARPS_PER_BLOCK, 1, 1)
template <typename input_t, int DSTATE, bool HAS_D, bool HAS_Z,
          bool DT_SOFTPLUS>
__global__ void selective_state_update_kernel_v3(
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
  // Which warp am I in (globally)?
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  // Lane within warp (0-31)
  const int lane_id = threadIdx.x % 32;

  if (warp_id >= total_elements) return;

  // Decompose warp_id into (batch, head, dim)
  const int dim_idx = warp_id % dim;
  const int head_idx = (warp_id / dim) % nheads;
  const int batch_idx = warp_id / (dim * nheads);

  // Which group does this head belong to?
  const int group_idx = head_idx / heads_per_group;

  // Input offset for this (batch, head, dim)
  const int x_offset =
      batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;

  // Load scalar inputs (same for all lanes in warp)
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
                         head_idx * stride_state_head + dim_idx * DSTATE;
  const int A_base = head_idx * dim * DSTATE + dim_idx * DSTATE;

  // B and C offset for this batch and group
  const int bc_offset = batch_idx * stride_B_batch + group_idx * stride_B_group;

  // Each lane processes DSTATE/32 = 4 elements (for DSTATE=128)
  // We process elements: lane_id, lane_id+32, lane_id+64, lane_id+96
  float out_acc = 0.0f;

#pragma unroll
  for (int i = 0; i < DSTATE / 32; i++) {
    const int n = lane_id + i * 32;

    // Load state, A, B, C for this dstate index
    float state_val = state[state_base + n];
    float A_val = A[A_base + n];
    float B_val = static_cast<float>(B[bc_offset + n]);
    float C_val = static_cast<float>(C[bc_offset + n]);

    // dA = exp(A * dt)
    float dA = expf(A_val * dt_val);

    // dB = B * dt
    float dB = B_val * dt_val;

    // Update state: state = state * dA + dB * x
    state_val = state_val * dA + dB * x_val;

    // Store updated state
    state[state_base + n] = state_val;

    // Accumulate output: out += state * C
    out_acc += state_val * C_val;
  }

  // Warp-level reduction to sum out_acc across all lanes
  out_acc = warp_reduce_sum(out_acc);

  // Lane 0 writes the final output
  if (lane_id == 0) {
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

// Macro for kernel launch
#define LAUNCH_KERNEL_V3(HAS_D_VAL, HAS_Z_VAL, DT_SOFTPLUS_VAL)         \
  selective_state_update_kernel_v3<scalar_t, 128, HAS_D_VAL, HAS_Z_VAL, \
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
    LAUNCH_KERNEL_V3(true, true, true);
  } else if (has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V3(true, true, false);
  } else if (has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V3(true, false, true);
  } else if (has_D && !has_z && !dt_softplus) {
    LAUNCH_KERNEL_V3(true, false, false);
  } else if (!has_D && has_z && dt_softplus) {
    LAUNCH_KERNEL_V3(false, true, true);
  } else if (!has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V3(false, true, false);
  } else if (!has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V3(false, false, true);
  } else {
    LAUNCH_KERNEL_V3(false, false, false);
  }
}

#undef LAUNCH_KERNEL_V3

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

  // Get strides
  const int stride_state_batch = state.stride(0);
  const int stride_state_head = state.stride(1);
  const int stride_x_batch = x.stride(0);
  const int stride_x_head = x.stride(1);
  const int stride_B_batch = B.stride(0);
  const int stride_B_group = B.stride(1);

  // Total elements: each warp handles one (batch, head, dim)
  const int total_elements = batch_size * nheads * dim;

  // Use warps_per_block from threads_per_block (must be multiple of 32)
  const int warps_per_block = threads_per_block / 32;
  const int threads = warps_per_block * 32;
  const int num_blocks =
      (total_elements + warps_per_block - 1) / warps_per_block;

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
