// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Optimized Mamba SSM decode kernel for Blackwell (B200) and newer GPUs.
// Uses TMA-style memory access patterns for better bandwidth utilization.

#pragma once

#include <torch/all.h>

namespace vllm {
namespace mamba {

// Optimized selective state update for decode phase.
// Processes state updates with improved memory access patterns for B200.
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
    bool dt_softplus);

}  // namespace mamba
}  // namespace vllm
