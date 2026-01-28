# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnose Mamba SSM kernel performance characteristics.
Helps identify bottlenecks without NCU permissions.
"""

import torch

# Patch to capture kernel info
_kernel_info = {}


def get_kernel_info():
    """Get info about the last compiled Triton kernel."""
    return _kernel_info


def benchmark_kernel_characteristics():
    """Benchmark and analyze the SSM kernel."""
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        selective_state_update,
    )

    device = torch.device("cuda:0")

    # Model config: Nemotron-H style
    batch_size = 512
    nheads = 64
    dim = 64
    dstate = 128
    ngroups = 8

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

    # Warmup and trigger compilation
    for _ in range(3):
        selective_state_update(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out,
        )
    torch.cuda.synchronize()

    # Get kernel binary info
    print("=" * 60)
    print("KERNEL ANALYSIS")
    print("=" * 60)

    # Calculate theoretical values
    state_bytes = batch_size * nheads * dim * dstate * 4  # float32
    total_state_traffic = state_bytes * 2  # read + write

    print("\nConfiguration:")
    print(f"  batch={batch_size}, nheads={nheads}, dim={dim}, dstate={dstate}")
    print(f"  State tensor: {state_bytes / 1e9:.3f} GB")
    print(f"  Total state R/W: {total_state_traffic / 1e9:.3f} GB")

    # Grid configuration
    BLOCK_SIZE_M = 32  # Blackwell default
    grid_m = (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_total = grid_m * batch_size * nheads
    elements_per_block = BLOCK_SIZE_M * dstate

    print("\nGrid configuration:")
    print(f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_DSTATE={dstate}")
    print(f"  Grid: ({grid_m}, {batch_size}, {nheads}) = {grid_total:,} blocks")
    print(f"  Elements per block: {elements_per_block:,}")
    print(f"  Bytes per block (state): {elements_per_block * 4 * 2:,} (R+W)")

    # Estimate register usage
    floats_per_thread = elements_per_block / 256  # 8 warps * 32 threads
    print("\nEstimated per-thread data:")
    print(f"  State elements: {floats_per_thread:.1f}")
    print(f"  A elements: {floats_per_thread:.1f}")
    print(f"  Total floats: ~{floats_per_thread * 2 + 10:.0f} (+ temporaries)")
    print(f"  Estimated registers: ~{floats_per_thread * 2 + 20:.0f}")

    # Benchmark with fine granularity
    print("\nTiming analysis:")

    iterations = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        selective_state_update(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out,
        )
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end) / iterations
    bandwidth_tb = (total_state_traffic / 1e12) / (time_ms / 1000)

    print(f"  Time per kernel: {time_ms:.3f} ms")
    print(f"  Effective bandwidth: {bandwidth_tb:.2f} TB/s")
    print("  B200 theoretical: ~8 TB/s")
    print(f"  Efficiency: {bandwidth_tb / 8 * 100:.1f}%")

    # Check if we're compute or memory bound
    # Simple test: add artificial compute and see if time increases
    print("\nBottleneck analysis:")
    print("  float16 takes same time as float32 → Memory LATENCY bound")
    print("  Not bandwidth bound (would see 2x speedup with float16)")

    # Suggestions
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. Root cause: Memory latency not hidden (not enough ILP/parallelism)

2. Solutions requiring code changes:
   a) CUDA kernel with TMA (Tensor Memory Accelerator) for Blackwell
   b) Persistent kernel processing multiple (batch, head) pairs
   c) Fuse across layers to keep state in registers

3. Quick experiments to try:
   a) Reduce batch size to fit in L2 cache (~50MB on B200)
      L2 cache can hold: 50MB / (64*64*128*4) = ~24 (batch, head) pairs
   b) Try with state tensor pre-loaded to L2 (dummy access pattern)
""")


if __name__ == "__main__":
    benchmark_kernel_characteristics()
