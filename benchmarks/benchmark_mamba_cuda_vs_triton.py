# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing Triton vs CUDA Mamba decode kernels.

Usage:
    # First rebuild vLLM with the new CUDA kernel:
    pip install -e . --no-build-isolation

    # Then run the benchmark:
    python benchmarks/benchmark_mamba_cuda_vs_triton.py
"""

import torch


def benchmark_triton_kernel(
    batch_size: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    num_iterations: int = 100,
) -> float:
    """Benchmark the Triton kernel."""
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        _selective_scan_update_kernel,
    )
    from vllm.triton_utils import triton

    device = torch.device("cuda:0")

    # Create tensors
    state = torch.randn(
        batch_size, nheads, dim, dstate, device=device, dtype=torch.float32
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    A = torch.randn(nheads, dim, dstate, device=device, dtype=torch.float32) * -0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=torch.bfloat16)
    D = torch.randn(nheads, dim, device=device, dtype=torch.bfloat16)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)

    # Block size tuning (Blackwell settings)
    BLOCK_SIZE_M, num_warps = 32, 8
    _ = triton.next_power_of_2(dstate)  # Verify triton is available

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch_size, nheads)

    # Warmup
    for _ in range(10):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            None,
            None,
            -1,
            None,
            None,
            batch_size,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0),
            D.stride(1),
            z.stride(0),
            z.stride(1),
            z.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            0,
            0,
            0,
            0,
            True,
            False,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            None,
            None,
            -1,
            None,
            None,
            batch_size,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0),
            D.stride(1),
            z.stride(0),
            z.stride(1),
            z.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            0,
            0,
            0,
            0,
            True,
            False,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iterations


def benchmark_cuda_kernel(
    batch_size: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    num_iterations: int = 100,
) -> float:
    """Benchmark the CUDA kernel."""
    from vllm import _custom_ops as ops

    device = torch.device("cuda:0")

    # Create tensors
    state = torch.randn(
        batch_size, nheads, dim, dstate, device=device, dtype=torch.float32
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    A = torch.randn(nheads, dim, dstate, device=device, dtype=torch.float32) * -0.1
    B = torch.randn(batch_size, ngroups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn(batch_size, ngroups, dstate, device=device, dtype=torch.bfloat16)
    D = torch.randn(nheads, dim, device=device, dtype=torch.bfloat16)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)

    # Warmup
    for _ in range(10):
        ops.selective_state_update_cuda(state, x, dt, A, B, C, D, z, dt_bias, out, True)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        ops.selective_state_update_cuda(state, x, dt, A, B, C, D, z, dt_bias, out, True)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iterations


def main():
    # Configuration matching Nemotron-H
    nheads = 64
    dim = 64
    dstate = 128
    ngroups = 8

    batch_sizes = [32, 64, 128, 256, 512, 1024]

    print("=" * 70)
    print("Mamba Decode Kernel Benchmark: Triton vs CUDA")
    print("=" * 70)
    print(
        f"Configuration: nheads={nheads}, dim={dim}, dstate={dstate}, ngroups={ngroups}"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"{'Batch':>8} | {'Triton (ms)':>12} | {'CUDA (ms)':>12} | {'Speedup':>10}")
    print("-" * 50)

    for batch_size in batch_sizes:
        try:
            triton_time = benchmark_triton_kernel(
                batch_size, nheads, dim, dstate, ngroups
            )
        except Exception:
            print(f"{batch_size:>8} | {'ERROR':>12} | ", end="")
            triton_time = None

        try:
            cuda_time = benchmark_cuda_kernel(batch_size, nheads, dim, dstate, ngroups)
        except Exception:
            print(f"{'ERROR':>12} | {'N/A':>10}")
            cuda_time = None
            continue

        if triton_time and cuda_time:
            speedup = triton_time / cuda_time
            print(
                f"{batch_size:>8} | {triton_time:>12.3f}"
                f" | {cuda_time:>12.3f} | {speedup:>9.2f}x"
            )
        elif cuda_time:
            print(f"{cuda_time:>12.3f} | {'N/A':>10}")

    print()
    print("Note: Speedup > 1.0 means CUDA kernel is faster than Triton.")


if __name__ == "__main__":
    main()
