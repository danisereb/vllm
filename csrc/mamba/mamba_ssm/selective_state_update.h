// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Configurable Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// Runtime configurable: block_size_m, threads_per_block

#pragma once

#include <torch/all.h>

namespace vllm {
namespace mamba {

// Configurable selective state update for decode phase.
// Processes state updates with configurable parallelization parameters.
//
// Args:
//   state: (batch, nheads, dim, dstate) - SSM state tensor, updated in-place
//   x: (batch, nheads, dim) - input tensor
//   dt: (batch, nheads, dim) - delta time tensor
//   A: (nheads, dim, dstate) - state transition matrix (negative values)
//   B: (batch, ngroups, dstate) - input projection
//   C: (batch, ngroups, dstate) - output projection
//   D: (nheads, dim) or None - skip connection
//   z: (batch, nheads, dim) or None - gate tensor
//   dt_bias: (nheads, dim) or None - bias for dt
//   out: (batch, nheads, dim) - output tensor
//   dt_softplus: whether to apply softplus to dt
//   block_size_m: dim elements per block (default: 32)
//   threads_per_block: threads per block (default: 128)
//
void selective_state_update_cuda(
    torch::Tensor& state,                   // [batch, nheads, dim, dstate]
    const torch::Tensor& x,                 // [batch, nheads, dim]
    const torch::Tensor& dt,                // [batch, nheads, dim]
    const torch::Tensor& A,                 // [nheads, dim, dstate]
    const torch::Tensor& B,                 // [batch, ngroups, dstate]
    const torch::Tensor& C,                 // [batch, ngroups, dstate]
    const c10::optional<torch::Tensor>& D,  // [nheads, dim]
    const c10::optional<torch::Tensor>& z,  // [batch, nheads, dim]
    const c10::optional<torch::Tensor>& dt_bias,  // [nheads, dim]
    torch::Tensor& out,                           // [batch, nheads, dim]
    bool dt_softplus,
    int64_t block_size_m,      // dim elements per block
    int64_t threads_per_block  // threads per block
);

}  // namespace mamba
}  // namespace vllm
