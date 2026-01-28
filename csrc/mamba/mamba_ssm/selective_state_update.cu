// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// V9 kernel: Software pipelining for improved memory latency hiding
//
// Key optimizations:
// 1. Each warp processes MULTIPLE dim elements sequentially (DIMS_PER_WARP)
// 2. Software pipelining: prefetch next element while computing current
// 3. Shared B/C loaded cooperatively with proper coverage for all thread counts
// 4. Vectorized float4 loads/stores for state and A tensors
// 5. Warp-level reduction using shuffle intrinsics

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

// V9 Kernel: Software pipelining for large batch performance
// Grid: (num_warps_total / warps_per_block, nheads, batch_size)
// Each warp processes multiple dim elements with pipelining
template <typename input_t, int DSTATE, int DIMS_PER_WARP, bool HAS_D,
          bool HAS_Z, bool DT_SOFTPLUS>
__global__ void __launch_bounds__(256, 4) selective_state_update_kernel_v9(
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
    int dim, int nheads, int ngroups, int heads_per_group,
    // Strides
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group) {
  // Shared memory for B/C (shared across all warps in block)
  __shared__ __align__(16) float shared_B[DSTATE];
  __shared__ __align__(16) float shared_C[DSTATE];

  // Grid: (num_warp_groups, nheads, batch_size)
  const int warp_group_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int group_idx = head_idx / heads_per_group;

  const int warps_per_block = blockDim.x / 32;
  const int local_warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // Global warp ID determines which dim elements this warp processes
  const int global_warp_id = warp_group_idx * warps_per_block + local_warp_id;
  const int dim_start = global_warp_id * DIMS_PER_WARP;

  // B/C pointers for this block
  const input_t* B_ptr =
      B + batch_idx * stride_B_batch + group_idx * stride_B_group;
  const input_t* C_ptr =
      C + batch_idx * stride_B_batch + group_idx * stride_B_group;

  // Cooperative load of B/C into shared memory
  {
    const int elems_per_thread = (DSTATE + blockDim.x - 1) / blockDim.x;
#pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      const int idx = threadIdx.x + i * blockDim.x;
      if (idx < DSTATE) {
        shared_B[idx] = to_float(__ldg(&B_ptr[idx]));
        shared_C[idx] = to_float(__ldg(&C_ptr[idx]));
      }
    }
  }

  __syncthreads();

  // Early exit if this warp has no work
  if (dim_start >= dim) return;

  // Process multiple dim elements sequentially with pipelining
#if __CUDA_ARCH__ >= 800 && DIMS_PER_WARP > 1
  // Pipelined version: prefetch next while computing current

  // Stage 0: Load first element
  int dim_idx = dim_start;
  if (dim_idx >= dim) return;

  int x_offset =
      batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;
  int state_base = batch_idx * stride_state_batch +
                   head_idx * stride_state_head + dim_idx * DSTATE;
  int A_base = head_idx * dim * DSTATE + dim_idx * DSTATE;

  // Load current state/A
  const float4* state_vec = reinterpret_cast<const float4*>(state + state_base);
  const float4* A_vec = reinterpret_cast<const float4*>(A + A_base);
  float4 state_f4 = __ldg(&state_vec[lane_id]);
  float4 A_f4 = __ldg(&A_vec[lane_id]);

  // Prefetch scalars
  float x_val = to_float(__ldg(&x[x_offset]));
  float dt_val = to_float(__ldg(&dt[x_offset]));
  if (dt_bias != nullptr) {
    dt_val += to_float(__ldg(&dt_bias[head_idx * dim + dim_idx]));
  }
  if constexpr (DT_SOFTPLUS) {
    dt_val = softplus(dt_val);
  }

  float z_val = 0.0f, d_val = 0.0f;
  if constexpr (HAS_Z) z_val = to_float(__ldg(&z[x_offset]));
  if constexpr (HAS_D) d_val = to_float(__ldg(&D[head_idx * dim + dim_idx]));

  #pragma unroll
  for (int d = 0; d < DIMS_PER_WARP; d++) {
    dim_idx = dim_start + d;
    if (dim_idx >= dim) break;

    // Prefetch next element's data (if not last)
    float4 next_state_f4, next_A_f4;
    float next_x_val = 0, next_dt_val = 0, next_z_val = 0, next_d_val = 0;

    if (d + 1 < DIMS_PER_WARP && dim_start + d + 1 < dim) {
      int next_dim_idx = dim_start + d + 1;
      int next_x_offset =
          batch_idx * stride_x_batch + head_idx * stride_x_head + next_dim_idx;
      int next_state_base = batch_idx * stride_state_batch +
                            head_idx * stride_state_head +
                            next_dim_idx * DSTATE;
      int next_A_base = head_idx * dim * DSTATE + next_dim_idx * DSTATE;

      const float4* next_state_vec =
          reinterpret_cast<const float4*>(state + next_state_base);
      const float4* next_A_vec =
          reinterpret_cast<const float4*>(A + next_A_base);

      next_state_f4 = __ldg(&next_state_vec[lane_id]);
      next_A_f4 = __ldg(&next_A_vec[lane_id]);

      next_x_val = to_float(__ldg(&x[next_x_offset]));
      next_dt_val = to_float(__ldg(&dt[next_x_offset]));
      if (dt_bias != nullptr) {
        next_dt_val += to_float(__ldg(&dt_bias[head_idx * dim + next_dim_idx]));
      }
      if constexpr (DT_SOFTPLUS) {
        next_dt_val = softplus(next_dt_val);
      }
      if constexpr (HAS_Z) next_z_val = to_float(__ldg(&z[next_x_offset]));
      if constexpr (HAS_D)
        next_d_val = to_float(__ldg(&D[head_idx * dim + next_dim_idx]));
    }

    // Compute current element
    float B_vals[4], C_vals[4];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int n = lane_id * 4 + i;
      B_vals[i] = shared_B[n];
      C_vals[i] = shared_C[n];
    }

    float state_vals[4] = {state_f4.x, state_f4.y, state_f4.z, state_f4.w};
    float A_vals[4] = {A_f4.x, A_f4.y, A_f4.z, A_f4.w};

    float out_acc = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float dA = expf(A_vals[i] * dt_val);
      float dB = B_vals[i] * dt_val;
      state_vals[i] = state_vals[i] * dA + dB * x_val;
      out_acc += state_vals[i] * C_vals[i];
    }

    // Store updated state
    x_offset = batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;
    state_base = batch_idx * stride_state_batch + head_idx * stride_state_head +
                 dim_idx * DSTATE;
    float4* state_out_vec = reinterpret_cast<float4*>(state + state_base);
    float4 state_out =
        make_float4(state_vals[0], state_vals[1], state_vals[2], state_vals[3]);
    state_out_vec[lane_id] = state_out;

    // Warp reduction and output
    out_acc = warp_reduce_sum(out_acc);
    if (lane_id == 0) {
      if constexpr (HAS_D) out_acc += x_val * d_val;
      if constexpr (HAS_Z) out_acc *= z_val * fast_sigmoid(z_val);
      out[x_offset] = static_cast<input_t>(out_acc);
    }

    // Move prefetched to current for next iteration
    state_f4 = next_state_f4;
    A_f4 = next_A_f4;
    x_val = next_x_val;
    dt_val = next_dt_val;
    z_val = next_z_val;
    d_val = next_d_val;
  }

#else
  // Non-pipelined version for SM < 8.0 or DIMS_PER_WARP == 1
  #pragma unroll
  for (int d = 0; d < DIMS_PER_WARP; d++) {
    const int dim_idx = dim_start + d;
    if (dim_idx >= dim) break;

    const int x_offset =
        batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;
    const int state_base = batch_idx * stride_state_batch +
                           head_idx * stride_state_head + dim_idx * DSTATE;
    const int A_base = head_idx * dim * DSTATE + dim_idx * DSTATE;

    float x_val = to_float(__ldg(&x[x_offset]));
    float dt_val = to_float(__ldg(&dt[x_offset]));
    if (dt_bias != nullptr) {
      dt_val += to_float(__ldg(&dt_bias[head_idx * dim + dim_idx]));
    }
    if constexpr (DT_SOFTPLUS) {
      dt_val = softplus(dt_val);
    }

    float z_val = 0.0f, d_val = 0.0f;
    if constexpr (HAS_Z) z_val = to_float(__ldg(&z[x_offset]));
    if constexpr (HAS_D) d_val = to_float(__ldg(&D[head_idx * dim + dim_idx]));

    const float4* state_vec =
        reinterpret_cast<const float4*>(state + state_base);
    const float4* A_vec = reinterpret_cast<const float4*>(A + A_base);
    float4* state_out_vec = reinterpret_cast<float4*>(state + state_base);

    float4 state_f4 = __ldg(&state_vec[lane_id]);
    float4 A_f4 = __ldg(&A_vec[lane_id]);

    float B_vals[4], C_vals[4];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int n = lane_id * 4 + i;
      B_vals[i] = shared_B[n];
      C_vals[i] = shared_C[n];
    }

    float state_vals[4] = {state_f4.x, state_f4.y, state_f4.z, state_f4.w};
    float A_vals[4] = {A_f4.x, A_f4.y, A_f4.z, A_f4.w};

    float out_acc = 0.0f;
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float dA = expf(A_vals[i] * dt_val);
      float dB = B_vals[i] * dt_val;
      state_vals[i] = state_vals[i] * dA + dB * x_val;
      out_acc += state_vals[i] * C_vals[i];
    }

    float4 state_out =
        make_float4(state_vals[0], state_vals[1], state_vals[2], state_vals[3]);
    state_out_vec[lane_id] = state_out;

    out_acc = warp_reduce_sum(out_acc);
    if (lane_id == 0) {
      if constexpr (HAS_D) out_acc += x_val * d_val;
      if constexpr (HAS_Z) out_acc *= z_val * fast_sigmoid(z_val);
      out[x_offset] = static_cast<input_t>(out_acc);
    }
  }
#endif
}

// Launch helper with DIMS_PER_WARP template
template <typename scalar_t, int DIMS_PER_WARP>
void launch_kernel_with_dims_per_warp(
    float* state_ptr, const scalar_t* x_ptr, const scalar_t* dt_ptr,
    const float* A_ptr, const scalar_t* B_ptr, const scalar_t* C_ptr,
    const scalar_t* D_ptr, const scalar_t* z_ptr, const scalar_t* dt_bias_ptr,
    scalar_t* out_ptr, bool has_D, bool has_z, bool dt_softplus, int batch_size,
    int nheads, int dim, int ngroups, int heads_per_group,
    int stride_state_batch, int stride_state_head, int stride_x_batch,
    int stride_x_head, int stride_B_batch, int stride_B_group,
    int threads_per_block, cudaStream_t stream) {
  const int warps_per_block = threads_per_block / 32;
  const int dims_per_block = warps_per_block * DIMS_PER_WARP;
  const int num_warp_groups = (dim + dims_per_block - 1) / dims_per_block;

  dim3 grid(num_warp_groups, nheads, batch_size);
  dim3 block(threads_per_block);

#define LAUNCH_V9(HAS_D_VAL, HAS_Z_VAL, DT_SOFTPLUS_VAL)                    \
  selective_state_update_kernel_v9<scalar_t, 128, DIMS_PER_WARP, HAS_D_VAL, \
                                   HAS_Z_VAL, DT_SOFTPLUS_VAL>              \
      <<<grid, block, 0, stream>>>(                                         \
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,      \
          dt_bias_ptr, out_ptr, dim, nheads, ngroups, heads_per_group,      \
          stride_state_batch, stride_state_head, stride_x_batch,            \
          stride_x_head, stride_B_batch, stride_B_group)

  if (has_D && has_z && dt_softplus) {
    LAUNCH_V9(true, true, true);
  } else if (has_D && has_z && !dt_softplus) {
    LAUNCH_V9(true, true, false);
  } else if (has_D && !has_z && dt_softplus) {
    LAUNCH_V9(true, false, true);
  } else if (has_D && !has_z && !dt_softplus) {
    LAUNCH_V9(true, false, false);
  } else if (!has_D && has_z && dt_softplus) {
    LAUNCH_V9(false, true, true);
  } else if (!has_D && has_z && !dt_softplus) {
    LAUNCH_V9(false, true, false);
  } else if (!has_D && !has_z && dt_softplus) {
    LAUNCH_V9(false, false, true);
  } else {
    LAUNCH_V9(false, false, false);
  }
#undef LAUNCH_V9
}

// Helper template function for kernel dispatch
template <typename scalar_t>
void launch_selective_state_update_kernel(
    float* state_ptr, const float* A_ptr, const torch::Tensor& x,
    const torch::Tensor& dt, const torch::Tensor& B, const torch::Tensor& C,
    const c10::optional<torch::Tensor>& D,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& dt_bias, torch::Tensor& out, bool has_D,
    bool has_z, bool dt_softplus, int batch_size, int nheads, int dim,
    int ngroups, int heads_per_group, int stride_state_batch,
    int stride_state_head, int stride_x_batch, int stride_x_head,
    int stride_B_batch, int stride_B_group, int threads_per_block,
    int dims_per_warp, cudaStream_t stream) {
  const scalar_t* x_ptr = x.data_ptr<scalar_t>();
  const scalar_t* dt_ptr = dt.data_ptr<scalar_t>();
  const scalar_t* B_ptr = B.data_ptr<scalar_t>();
  const scalar_t* C_ptr = C.data_ptr<scalar_t>();
  const scalar_t* D_ptr = has_D ? D.value().data_ptr<scalar_t>() : nullptr;
  const scalar_t* z_ptr = has_z ? z.value().data_ptr<scalar_t>() : nullptr;
  const scalar_t* dt_bias_ptr =
      dt_bias.has_value() ? dt_bias.value().data_ptr<scalar_t>() : nullptr;
  scalar_t* out_ptr = out.data_ptr<scalar_t>();

  // Dispatch based on dims_per_warp (must be compile-time for pipelining)
  switch (dims_per_warp) {
    case 1:
      launch_kernel_with_dims_per_warp<scalar_t, 1>(
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
          dt_bias_ptr, out_ptr, has_D, has_z, dt_softplus, batch_size, nheads,
          dim, ngroups, heads_per_group, stride_state_batch, stride_state_head,
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group,
          threads_per_block, stream);
      break;
    case 2:
      launch_kernel_with_dims_per_warp<scalar_t, 2>(
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
          dt_bias_ptr, out_ptr, has_D, has_z, dt_softplus, batch_size, nheads,
          dim, ngroups, heads_per_group, stride_state_batch, stride_state_head,
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group,
          threads_per_block, stream);
      break;
    case 4:
      launch_kernel_with_dims_per_warp<scalar_t, 4>(
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
          dt_bias_ptr, out_ptr, has_D, has_z, dt_softplus, batch_size, nheads,
          dim, ngroups, heads_per_group, stride_state_batch, stride_state_head,
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group,
          threads_per_block, stream);
      break;
    case 8:
      launch_kernel_with_dims_per_warp<scalar_t, 8>(
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
          dt_bias_ptr, out_ptr, has_D, has_z, dt_softplus, batch_size, nheads,
          dim, ngroups, heads_per_group, stride_state_batch, stride_state_head,
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group,
          threads_per_block, stream);
      break;
    default:
      // Fallback to dims_per_warp=1
      launch_kernel_with_dims_per_warp<scalar_t, 1>(
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
          dt_bias_ptr, out_ptr, has_D, has_z, dt_softplus, batch_size, nheads,
          dim, ngroups, heads_per_group, stride_state_batch, stride_state_head,
          stride_x_batch, stride_x_head, stride_B_batch, stride_B_group,
          threads_per_block, stream);
      break;
  }
}

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
  TORCH_CHECK(threads_per_block >= 32 && threads_per_block <= 256,
              "threads_per_block must be between 32 and 256, got ",
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

  // Use block_size_m as dims_per_warp (how many dim elements each warp
  // processes) Valid values: 1, 2, 4, 8
  int dims_per_warp = static_cast<int>(block_size_m);
  if (dims_per_warp < 1) dims_per_warp = 1;
  if (dims_per_warp > 8) dims_per_warp = 8;
  // Round to nearest power of 2
  if (dims_per_warp >= 6)
    dims_per_warp = 8;
  else if (dims_per_warp >= 3)
    dims_per_warp = 4;
  else if (dims_per_warp >= 2)
    dims_per_warp = 2;
  else
    dims_per_warp = 1;

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
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, static_cast<int>(threads_per_block),
        dims_per_warp, stream);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_selective_state_update_kernel<at::BFloat16>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, static_cast<int>(threads_per_block),
        dims_per_warp, stream);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    launch_selective_state_update_kernel<float>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, static_cast<int>(threads_per_block),
        dims_per_warp, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype: ", x.scalar_type());
  }

  // Check for errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mamba
}  // namespace vllm
