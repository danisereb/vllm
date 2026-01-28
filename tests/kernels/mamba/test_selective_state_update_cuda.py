# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the optimized CUDA selective_state_update kernel for Blackwell GPUs.

This kernel is designed for dstate=128 (Mamba2/Nemotron-H configuration) and
uses warp-level parallelism with vectorized loads for better performance.
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Reference implementation for selective_state_update.

    Argument:
        state: (batch, nheads, dim, dstate)
        x: (batch, nheads, dim)
        dt: (batch, nheads, dim)
        A: (nheads, dim, dstate)
        B: (batch, ngroups, dstate)
        C: (batch, ngroups, dstate)
        D: (nheads, dim)
        z: (batch, nheads, dim)
        dt_bias: (nheads, dim)
    Return:
        out: (batch, nheads, dim)
    """
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(
        rearrange(dt, "b h d -> b h d 1") * A
    )  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
        B, "b h n -> b h 1 n"
    )  # (batch, nheads, dim, dstate)
    state.copy_(
        state * dA + dB * rearrange(x, "b h d -> b h d 1")
    )  # (batch, nheads, dim, dstate)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    return out


def selective_state_update_cuda_opcheck_fn(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    out=None,
    dt_softplus=True,
    block_size_m=32,
    threads_per_block=128,
):
    """Run opcheck on the CUDA kernel."""
    opcheck(
        torch.ops._C.selective_state_update_cuda,
        (
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            out,
            dt_softplus,
            block_size_m,
            threads_per_block,
        ),
        test_utils=["test_schema", "test_faketensor"],
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("has_D", [False, True])
@pytest.mark.parametrize("batch_size", [1, 32, 128, 512])
@pytest.mark.parametrize(
    "nheads,dim,dstate,ngroups",
    [
        (64, 64, 128, 8),  # Nemotron-H configuration
    ],
)
def test_selective_state_update_cuda_correctness(
    nheads, dim, dstate, ngroups, batch_size, has_D, has_z, itype
):
    """Test correctness of the CUDA kernel against reference implementation."""
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1

    set_random_seed(0)

    # Create input tensors matching the expected shapes
    state = torch.randn(
        batch_size, nheads, dim, dstate, dtype=torch.float32, device=device
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32) - 0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    D = torch.randn(nheads, dim, device=device, dtype=itype) if has_D else None
    z = (
        torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
        if has_z
        else None
    )
    dt_bias = torch.randn(nheads, dim, device=device, dtype=itype)
    out = torch.empty_like(x)

    # Clone state for reference computation
    state_ref = state.clone()

    # Run CUDA kernel
    ops.selective_state_update_cuda(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        out,
        dt_softplus=True,
        block_size_m=32,
        threads_per_block=128,
    )

    # Run reference implementation
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    # Compare results
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol), (
        f"State mismatch: max diff = {(state - state_ref).abs().max().item()}"
    )
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch: max diff = {(out - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("threads_per_block", [64, 128, 256])
@pytest.mark.parametrize("batch_size", [32, 128])
def test_selective_state_update_cuda_configs(batch_size, threads_per_block):
    """Test different kernel configurations."""
    device = "cuda"
    itype = torch.bfloat16
    rtol, atol = 1e-1, 1e-1

    set_random_seed(0)

    nheads, dim, dstate, ngroups = 64, 64, 128, 8

    state = torch.randn(
        batch_size, nheads, dim, dstate, dtype=torch.float32, device=device
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32) - 0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    D = torch.randn(nheads, dim, device=device, dtype=itype)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=itype)
    out = torch.empty_like(x)

    state_ref = state.clone()

    # Run with specified config
    ops.selective_state_update_cuda(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        out,
        dt_softplus=True,
        block_size_m=32,
        threads_per_block=threads_per_block,
    )

    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol), (
        f"State mismatch with threads={threads_per_block}: "
        f"max diff = {(state - state_ref).abs().max().item()}"
    )
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch with threads={threads_per_block}: "
        f"max diff = {(out - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
def test_selective_state_update_cuda_opcheck(itype):
    """Test that the kernel passes torch opcheck validation."""
    device = "cuda"
    set_random_seed(0)

    batch_size = 32
    nheads, dim, dstate, ngroups = 64, 64, 128, 8

    state = torch.randn(
        batch_size, nheads, dim, dstate, dtype=torch.float32, device=device
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32) - 0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    D = torch.randn(nheads, dim, device=device, dtype=itype)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=itype)
    out = torch.empty_like(x)

    selective_state_update_cuda_opcheck_fn(
        state, x, dt, A, B, C, D, z, dt_bias, out, True, 32, 128
    )


@pytest.mark.parametrize("dt_softplus", [True, False])
def test_selective_state_update_cuda_dt_softplus(dt_softplus):
    """Test with and without dt_softplus."""
    device = "cuda"
    itype = torch.bfloat16
    rtol, atol = 1e-1, 1e-1

    set_random_seed(0)

    batch_size = 64
    nheads, dim, dstate, ngroups = 64, 64, 128, 8

    state = torch.randn(
        batch_size, nheads, dim, dstate, dtype=torch.float32, device=device
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32) - 0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    D = torch.randn(nheads, dim, device=device, dtype=itype)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=itype)
    out = torch.empty_like(x)

    state_ref = state.clone()

    ops.selective_state_update_cuda(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        out,
        dt_softplus=dt_softplus,
        block_size_m=32,
        threads_per_block=128,
    )

    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol), (
        f"State mismatch with dt_softplus={dt_softplus}: "
        f"max diff = {(state - state_ref).abs().max().item()}"
    )
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch with dt_softplus={dt_softplus}: "
        f"max diff = {(out - out_ref).abs().max().item()}"
    )


def test_selective_state_update_cuda_no_optional_tensors():
    """Test without optional tensors (D, z, dt_bias)."""
    device = "cuda"
    itype = torch.bfloat16
    rtol, atol = 1e-1, 1e-1

    set_random_seed(0)

    batch_size = 64
    nheads, dim, dstate, ngroups = 64, 64, 128, 8

    state = torch.randn(
        batch_size, nheads, dim, dstate, dtype=torch.float32, device=device
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=torch.float32) - 0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=itype)
    out = torch.empty_like(x)

    state_ref = state.clone()

    # No D, z, or dt_bias
    ops.selective_state_update_cuda(
        state,
        x,
        dt,
        A,
        B,
        C,
        None,  # D
        None,  # z
        None,  # dt_bias
        out,
        dt_softplus=True,
        block_size_m=32,
        threads_per_block=128,
    )

    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=True
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol), (
        f"State mismatch: max diff = {(state - state_ref).abs().max().item()}"
    )
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch: max diff = {(out - out_ref).abs().max().item()}"
    )
