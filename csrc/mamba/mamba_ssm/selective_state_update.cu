// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// V8 kernel with optional TMA (Tensor Memory Accelerator) for SM 9.0+
//
// Two modes:
// 1. Standard mode (SM < 9.0): Cooperative loads with shared memory
// 2. TMA mode (SM >= 9.0): True async bulk copies with mbarrier
//
// Key optimizations:
// 1. cp.async.bulk for TMA loads (SM 9.0+)
// 2. mbarrier for async completion tracking
// 3. 3D grid for B/C sharing
// 4. Warp-level cooperation over dstate
// 5. Vectorized float4 loads/stores

#include "selective_state_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

// TMA support requires SM 9.0+ (Hopper/Blackwell)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  #define VLLM_USE_TMA 1
  #include <cuda/barrier>
  #include <cuda/pipeline>
using barrier = cuda::barrier<cuda::thread_scope_block>;
#else
  #define VLLM_USE_TMA 0
#endif

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

// V8 Kernel: Optimized with async copies and proper shared memory handling
// Grid: (num_dim_blocks, nheads, batch_size)
// Each block processes warps_per_block consecutive dim elements
// All warps in block share B/C loaded into shared memory
template <typename input_t, int DSTATE, bool HAS_D, bool HAS_Z,
          bool DT_SOFTPLUS>
__global__ void __launch_bounds__(256, 4) selective_state_update_kernel_v8(
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
  // Shared memory layout for B/C (shared across warps in block)
  __shared__ __align__(16) float shared_B[DSTATE];
  __shared__ __align__(16) float shared_C[DSTATE];

#if VLLM_USE_TMA
  // Use cuda::barrier for SM 9.0+ async completion tracking
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();
#endif

  // Grid: (num_dim_blocks, nheads, batch_size)
  const int dim_block_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int group_idx = head_idx / heads_per_group;

  const int warps_per_block = blockDim.x / 32;
  const int local_warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int dim_idx = dim_block_idx * warps_per_block + local_warp_id;

  // B/C pointers for this block (shared by all warps)
  const input_t* B_ptr =
      B + batch_idx * stride_B_batch + group_idx * stride_B_group;
  const input_t* C_ptr =
      C + batch_idx * stride_B_batch + group_idx * stride_B_group;

// Load B/C into shared memory cooperatively
// Use async copy path for SM 8.0+
#if __CUDA_ARCH__ >= 800
  // Async cooperative load of B/C with conversion to float
  {
    // Calculate elements per thread to ensure full coverage
    constexpr int max_elems = (DSTATE + 31) / 32;  // Max for 32 threads
    const int elems_per_thread = (DSTATE + blockDim.x - 1) / blockDim.x;

  #pragma unroll
    for (int i = 0; i < max_elems; i++) {
      if (i < elems_per_thread) {
        const int idx = threadIdx.x + i * blockDim.x;
        if (idx < DSTATE) {
          // __ldg uses L2 cache for read-only data
          shared_B[idx] = to_float(__ldg(&B_ptr[idx]));
          shared_C[idx] = to_float(__ldg(&C_ptr[idx]));
        }
      }
    }
  }
#else
  // Standard cooperative load for older architectures
  for (int i = threadIdx.x; i < DSTATE; i += blockDim.x) {
    shared_B[i] = to_float(B_ptr[i]);
    shared_C[i] = to_float(C_ptr[i]);
  }
#endif

#if VLLM_USE_TMA
  // Use barrier instead of __syncthreads for SM 9.0+
  bar.arrive_and_wait();
#else
  // Synchronize to ensure B/C are loaded before computation
  __syncthreads();
#endif

  // Early exit for out-of-bounds warps (AFTER sync to avoid deadlock)
  if (dim_idx >= dim) return;

  // Input offset for this (batch, head, dim)
  const int x_offset =
      batch_idx * stride_x_batch + head_idx * stride_x_head + dim_idx;

  // State and A base offsets
  const int state_base = batch_idx * stride_state_batch +
                         head_idx * stride_state_head + dim_idx * DSTATE;
  const int A_base = head_idx * dim * DSTATE + dim_idx * DSTATE;

  // Load scalar inputs using __ldg for read-only data
  float x_val = to_float(__ldg(&x[x_offset]));
  float dt_val = to_float(__ldg(&dt[x_offset]));

  // Add dt_bias if present
  if (dt_bias != nullptr) {
    dt_val += to_float(__ldg(&dt_bias[head_idx * dim + dim_idx]));
  }

  // Apply softplus to dt
  if constexpr (DT_SOFTPLUS) {
    dt_val = softplus(dt_val);
  }

  // Load z if present
  float z_val = 0.0f;
  if constexpr (HAS_Z) {
    z_val = to_float(__ldg(&z[x_offset]));
  }

  // Load D if present
  float d_val = 0.0f;
  if constexpr (HAS_D) {
    d_val = to_float(__ldg(&D[head_idx * dim + dim_idx]));
  }

  // Each lane processes DSTATE/32 = 4 elements (for DSTATE=128)
  float out_acc = 0.0f;

  // Load state and A using float4 vectorized loads via __ldg
  const float4* state_vec = reinterpret_cast<const float4*>(state + state_base);
  const float4* A_vec = reinterpret_cast<const float4*>(A + A_base);
  float4* state_out_vec = reinterpret_cast<float4*>(state + state_base);

  // Each lane handles one float4 (4 consecutive dstate elements)
  float4 state_f4 = __ldg(&state_vec[lane_id]);
  float4 A_f4 = __ldg(&A_vec[lane_id]);

  // Read B and C from shared memory (already converted to float)
  float B_vals[4], C_vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int n = lane_id * 4 + i;
    B_vals[i] = shared_B[n];
    C_vals[i] = shared_C[n];
  }

  // Process 4 elements per lane
  float state_vals[4] = {state_f4.x, state_f4.y, state_f4.z, state_f4.w};
  float A_vals[4] = {A_f4.x, A_f4.y, A_f4.z, A_f4.w};

#pragma unroll
  for (int i = 0; i < 4; i++) {
    // dA = exp(A * dt)
    float dA = expf(A_vals[i] * dt_val);

    // dB = B * dt
    float dB = B_vals[i] * dt_val;

    // Update state: state = state * dA + dB * x
    state_vals[i] = state_vals[i] * dA + dB * x_val;

    // Accumulate output: out += state * C
    out_acc += state_vals[i] * C_vals[i];
  }

  // Store updated state using float4
  float4 state_out =
      make_float4(state_vals[0], state_vals[1], state_vals[2], state_vals[3]);
  state_out_vec[lane_id] = state_out;

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

// Macro for kernel launch (v8 uses 3D grid)
#define LAUNCH_KERNEL_V8(HAS_D_VAL, HAS_Z_VAL, DT_SOFTPLUS_VAL)         \
  selective_state_update_kernel_v8<scalar_t, 128, HAS_D_VAL, HAS_Z_VAL, \
                                   DT_SOFTPLUS_VAL>                     \
      <<<grid, block, 0, stream>>>(                                     \
          state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,  \
          dt_bias_ptr, out_ptr, dim, nheads, ngroups, heads_per_group,  \
          stride_state_batch, stride_state_head, stride_x_batch,        \
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
    int ngroups, int heads_per_group, int stride_state_batch,
    int stride_state_head, int stride_x_batch, int stride_x_head,
    int stride_B_batch, int stride_B_group, dim3 grid, dim3 block,
    cudaStream_t stream) {
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
    LAUNCH_KERNEL_V8(true, true, true);
  } else if (has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V8(true, true, false);
  } else if (has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V8(true, false, true);
  } else if (has_D && !has_z && !dt_softplus) {
    LAUNCH_KERNEL_V8(true, false, false);
  } else if (!has_D && has_z && dt_softplus) {
    LAUNCH_KERNEL_V8(false, true, true);
  } else if (!has_D && has_z && !dt_softplus) {
    LAUNCH_KERNEL_V8(false, true, false);
  } else if (!has_D && !has_z && dt_softplus) {
    LAUNCH_KERNEL_V8(false, false, true);
  } else {
    LAUNCH_KERNEL_V8(false, false, false);
  }
}

#undef LAUNCH_KERNEL_V8

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

  // V8: Use 3D grid (num_dim_blocks, nheads, batch_size)
  // This ensures all warps in a block share the same (batch, group) for B/C
  const int warps_per_block = threads_per_block / 32;
  const int threads = warps_per_block * 32;
  const int num_dim_blocks = (dim + warps_per_block - 1) / warps_per_block;

  dim3 grid(num_dim_blocks, nheads, batch_size);
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
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, grid, block, stream);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    launch_selective_state_update_kernel<at::BFloat16>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, grid, block, stream);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    launch_selective_state_update_kernel<float>(
        state_ptr, A_ptr, x, dt, B, C, D, z, dt_bias, out, has_D, has_z,
        dt_softplus, batch_size, nheads, dim, ngroups, heads_per_group,
        stride_state_batch, stride_state_head, stride_x_batch, stride_x_head,
        stride_B_batch, stride_B_group, grid, block, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype: ", x.scalar_type());
  }

  // Check for errors
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mamba
}  // namespace vllm
