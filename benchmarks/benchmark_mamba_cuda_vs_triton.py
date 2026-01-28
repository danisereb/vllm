# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing Triton vs CUDA Mamba decode kernels.

Usage:
    # First rebuild vLLM with the new CUDA kernel:
    pip install -e . --no-build-isolation

    # Then run the benchmark:
    python benchmarks/benchmark_mamba_cuda_vs_triton.py

    # Sweep configurations:
    python benchmarks/benchmark_mamba_cuda_vs_triton.py --sweep
"""

import argparse

import torch


def benchmark_triton_kernel(
    batch_size: int,
    nheads: int,
    dim: int,
    dstate: int,
    ngroups: int,
    state_dtype: torch.dtype = torch.float32,
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
        batch_size, nheads, dim, dstate, device=device, dtype=state_dtype
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
    block_size_m: int = 32,
    threads_per_block: int = 128,
    state_dtype: torch.dtype = torch.float32,
    num_iterations: int = 100,
) -> float:
    """Benchmark the CUDA kernel with configurable parameters."""
    from vllm import _custom_ops as ops

    device = torch.device("cuda:0")

    # Create tensors
    state = torch.randn(
        batch_size, nheads, dim, dstate, device=device, dtype=state_dtype
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
            True,
            block_size_m,
            threads_per_block,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
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
            True,
            block_size_m,
            threads_per_block,
        )
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iterations


def run_comparison(
    batch_sizes,
    nheads,
    dim,
    dstate,
    ngroups,
    block_size_m=32,
    threads_per_block=128,
    state_dtype=torch.float32,
):
    """Run comparison between Triton and CUDA kernels."""
    dtype_name = "float32" if state_dtype == torch.float32 else "float16"
    print("=" * 70)
    print("Mamba Decode Kernel Benchmark: Triton vs CUDA")
    print("=" * 70)
    print(f"Config: nheads={nheads}, dim={dim}, dstate={dstate}, ngroups={ngroups}")
    print(f"State dtype: {dtype_name}")
    print(
        f"CUDA params: block_size_m={block_size_m}, "
        f"threads_per_block={threads_per_block}"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"{'Batch':>8} | {'Triton (ms)':>12} | {'CUDA (ms)':>12} | {'Speedup':>10}")
    print("-" * 55)

    for batch_size in batch_sizes:
        triton_time = None
        cuda_time = None

        try:
            triton_time = benchmark_triton_kernel(
                batch_size, nheads, dim, dstate, ngroups, state_dtype
            )
        except Exception as e:
            print(f"{batch_size:>8} | Triton ERROR: {e}")
            continue

        try:
            cuda_time = benchmark_cuda_kernel(
                batch_size,
                nheads,
                dim,
                dstate,
                ngroups,
                block_size_m,
                threads_per_block,
                state_dtype,
            )
        except Exception as e:
            print(f"{batch_size:>8} | {triton_time:>12.3f} | CUDA ERROR: {e}")
            continue

        if triton_time and cuda_time:
            speedup = triton_time / cuda_time
            print(
                f"{batch_size:>8} | {triton_time:>12.3f}"
                f" | {cuda_time:>12.3f} | {speedup:>9.2f}x"
            )
        elif cuda_time:
            print(f"{batch_size:>8} | {'N/A':>12} | {cuda_time:>12.3f} | {'N/A':>10}")

    print()


def run_config_sweep(
    batch_size, nheads, dim, dstate, ngroups, state_dtype=torch.float32
):
    """Sweep different CUDA kernel configurations."""
    dtype_name = "float32" if state_dtype == torch.float32 else "float16"
    print("=" * 70)
    print("CUDA Kernel Configuration Sweep")
    print("=" * 70)
    print(
        f"Config: batch={batch_size}, nheads={nheads}, dim={dim}, "
        f"dstate={dstate}, ngroups={ngroups}"
    )
    print(f"State dtype: {dtype_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Get Triton baseline
    try:
        triton_time = benchmark_triton_kernel(
            batch_size, nheads, dim, dstate, ngroups, state_dtype
        )
        print(f"Triton baseline: {triton_time:.3f} ms")
    except Exception as e:
        print(f"Triton ERROR: {e}")
        triton_time = None
    print()

    # Configuration grid
    # Note: v4 kernel limited to 256 threads by __launch_bounds__(256, 4)
    block_sizes = [8, 16, 32, 64, 128]
    thread_counts = [32, 64, 128, 256]

    print(
        f"{'block_size_m':>12} | {'threads':>8} | {'CUDA (ms)':>10} | {'vs Triton':>10}"
    )
    print("-" * 50)

    best_time = float("inf")
    best_config = None

    for block_size_m in block_sizes:
        for threads in thread_counts:
            try:
                cuda_time = benchmark_cuda_kernel(
                    batch_size,
                    nheads,
                    dim,
                    dstate,
                    ngroups,
                    block_size_m,
                    threads,
                    state_dtype,
                )
                if triton_time:
                    speedup = triton_time / cuda_time
                    speedup_str = f"{speedup:>9.2f}x"
                else:
                    speedup_str = "N/A"

                marker = " *" if cuda_time < best_time else ""
                print(
                    f"{block_size_m:>12} | {threads:>8} | {cuda_time:>10.3f} | "
                    f"{speedup_str}{marker}"
                )

                if cuda_time < best_time:
                    best_time = cuda_time
                    best_config = (block_size_m, threads)

            except Exception as e:
                print(f"{block_size_m:>12} | {threads:>8} | ERROR: {e}")

    print()
    if best_config:
        print(
            f"Best config: block_size_m={best_config[0]}, "
            f"threads_per_block={best_config[1]}"
        )
        print(f"Best time: {best_time:.3f} ms")
        if triton_time:
            print(f"Best vs Triton: {triton_time / best_time:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Mamba CUDA vs Triton kernels"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep different CUDA kernel configurations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for sweep mode (default: 512)",
    )
    parser.add_argument(
        "--block-size-m",
        type=int,
        default=32,
        help="Dim elements per block (default: 32)",
    )
    parser.add_argument(
        "--threads", type=int, default=128, help="Threads per block (default: 128)"
    )
    parser.add_argument(
        "--state-dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="State/cache dtype (default: float32)",
    )
    args = parser.parse_args()

    # Configuration matching Nemotron-H
    nheads = 64
    dim = 64
    dstate = 128
    ngroups = 8

    # Parse state dtype
    state_dtype = torch.float32 if args.state_dtype == "float32" else torch.float16

    if args.sweep:
        run_config_sweep(args.batch_size, nheads, dim, dstate, ngroups, state_dtype)
    else:
        batch_sizes = [32, 64, 128, 256, 512, 1024]
        run_comparison(
            batch_sizes,
            nheads,
            dim,
            dstate,
            ngroups,
            args.block_size_m,
            args.threads,
            state_dtype,
        )

    print("Note: Speedup > 1.0 means CUDA kernel is faster than Triton.")


if __name__ == "__main__":
    main()
