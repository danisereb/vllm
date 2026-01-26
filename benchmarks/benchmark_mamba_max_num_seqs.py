# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from typing import NamedTuple

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")


class MambaConfig(NamedTuple):
    """Mamba model configuration for benchmarking."""

    num_heads: int
    head_dim: int
    state_size: int
    num_mamba_layers: int
    state_dtype: torch.dtype
    ngroups: int = 1


class BenchmarkResult(NamedTuple):
    """Result of benchmarking at a specific batch size."""

    batch_size: int
    time_ms: float
    time_per_token_us: float
    state_traffic_gb: float
    effective_bandwidth_tb_s: float


def get_config_from_model(
    model_path: str,
    state_dtype: torch.dtype,
    trust_remote_code: bool = True,
) -> MambaConfig:
    """Extract Mamba configuration from a HuggingFace model."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Try different attribute names used by various Mamba models
    num_heads = (
        getattr(hf_config, "mamba_num_heads", None)
        or getattr(hf_config, "n_mamba_heads", None)
        or getattr(hf_config, "mamba_n_heads", None)
        or getattr(hf_config, "num_heads", 64)
    )

    head_dim = (
        getattr(hf_config, "mamba_head_dim", None)
        or getattr(hf_config, "mamba_headdim", None)
        or getattr(hf_config, "mamba_d_head", None)
        or getattr(hf_config, "head_dim", 64)
    )

    state_size = (
        getattr(hf_config, "ssm_state_size", None)
        or getattr(hf_config, "mamba_d_state", None)
        or getattr(hf_config, "state_size", 128)
    )

    ngroups = getattr(hf_config, "mamba_n_groups", 1) or getattr(
        hf_config, "n_groups", 1
    )

    # Count Mamba layers
    num_layers = getattr(hf_config, "num_hidden_layers", 52)
    hybrid_pattern = getattr(hf_config, "hybrid_override_pattern", None)
    num_mamba_layers = hybrid_pattern.count("M") if hybrid_pattern else num_layers

    print(f"Model: {model_path}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  state_size: {state_size}")
    print(f"  ngroups: {ngroups}")
    print(f"  num_mamba_layers: {num_mamba_layers}")
    if hybrid_pattern:
        print(f"  hybrid_pattern: {hybrid_pattern[:50]}...")

    return MambaConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        state_size=state_size,
        num_mamba_layers=num_mamba_layers,
        state_dtype=state_dtype,
        ngroups=ngroups,
    )


def benchmark_ssm_kernel(
    config: MambaConfig,
    batch_size: int,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark the Mamba SSM kernel at a specific batch size."""
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    device = torch.device("cuda:0")
    nheads = config.num_heads
    dim = config.head_dim
    dstate = config.state_size
    ngroups = config.ngroups
    dtype = config.state_dtype
    num_layers = config.num_mamba_layers

    # Create tensors matching decode phase shapes
    state = torch.randn(batch_size, nheads, dim, dstate, device=device, dtype=dtype)
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
    for _ in range(warmup_iterations):
        for _ in range(num_layers):
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

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        for _ in range(num_layers):
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
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [
        start_events[i].elapsed_time(end_events[i]) for i in range(num_iterations)
    ]
    avg_time_ms = sum(times_ms) / len(times_ms)
    time_per_token_us = (avg_time_ms * 1000) / batch_size

    # Calculate memory traffic
    dtype_bytes = 4 if dtype == torch.float32 else 2
    state_bytes = batch_size * nheads * dim * dstate * dtype_bytes * num_layers
    state_traffic_bytes = state_bytes * 2  # Read + write
    state_traffic_gb = state_traffic_bytes / 1e9
    effective_bandwidth_tb_s = state_traffic_gb / (avg_time_ms / 1000) / 1000

    return BenchmarkResult(
        batch_size=batch_size,
        time_ms=avg_time_ms,
        time_per_token_us=time_per_token_us,
        state_traffic_gb=state_traffic_gb,
        effective_bandwidth_tb_s=effective_bandwidth_tb_s,
    )


def find_optimal_batch_size(
    config: MambaConfig,
    batch_sizes: list[int] | None = None,
    tp_size: int = 1,
    bw_threshold: float = 0.005,
) -> tuple[int, list[BenchmarkResult]]:
    """
    Find the optimal batch size by detecting memory bandwidth saturation.

    We increase batch size until the effective memory bandwidth stops improving.
    Once bandwidth plateaus (< threshold improvement), we've found the optimal
    point - going larger just increases latency without better efficiency.

    Args:
        config: Mamba configuration
        batch_sizes: Specific batch sizes to test, or None for auto
        tp_size: Tensor parallel size
        bw_threshold: Stop when BW improvement is below this (default 5%)
    """
    # Adjust config for TP (state is sharded across TP ranks)
    if tp_size > 1:
        config = config._replace(num_heads=config.num_heads // tp_size)
        print(f"\nAdjusted for TP={tp_size}: num_heads={config.num_heads}")

    print(f"\nBenchmarking SSM kernel ({config.num_mamba_layers} Mamba layers)...")
    print(f"State dtype: {config.state_dtype}")
    print(
        f"State shape per seq: "
        f"({config.num_heads}, {config.head_dim}, {config.state_size})"
    )
    print(
        f"\nFinding where memory bandwidth saturates "
        f"(< {bw_threshold * 100:.1f}% improvement)..."
    )

    if batch_sizes is None:
        batch_sizes = [
            32,
            64,
            96,
            128,
            192,
            256,
            320,
            384,
            448,
            512,
            640,
            768,
            896,
            1024,
        ]

    results: list[BenchmarkResult] = []

    print("\n" + "=" * 85)
    print(
        f"{'Batch':>8} | {'SSM (ms)':>10} | {'μs/token':>10} | "
        f"{'BW (TB/s)':>10} | {'BW Δ':>10} | {'Status':>12}"
    )
    print("=" * 85)

    optimal_batch = batch_sizes[0]
    prev_bw = 0.0
    saturated = False

    for batch_size in batch_sizes:
        try:
            result = benchmark_ssm_kernel(config, batch_size)
            results.append(result)

            # Calculate bandwidth improvement from previous
            if prev_bw > 0:
                bw_improvement = (result.effective_bandwidth_tb_s - prev_bw) / prev_bw
            else:
                bw_improvement = 1.0  # First iteration

            # Check if bandwidth is still improving
            if bw_improvement >= bw_threshold:
                status = "✓ improving"
                optimal_batch = batch_size
            else:
                if not saturated:
                    status = "→ SATURATED"
                    saturated = True
                else:
                    status = "  (plateau)"

            bw_delta_str = (
                f"+{bw_improvement * 100:.1f}%"
                if bw_improvement > 0
                else f"{bw_improvement * 100:.1f}%"
            )

            print(
                f"{batch_size:>8} | {result.time_ms:>10.2f} | "
                f"{result.time_per_token_us:>10.2f} | "
                f"{result.effective_bandwidth_tb_s:>10.2f} | "
                f"{bw_delta_str:>10} | "
                f"{status:>12}"
            )

            prev_bw = result.effective_bandwidth_tb_s
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{batch_size:>8} | OOM")
            torch.cuda.empty_cache()
            break

    print("=" * 85)

    # Scale by TP size
    recommended = optimal_batch * tp_size

    return recommended, results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Mamba SSM kernel to find optimal max-num-seqs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Mamba SSM cache dtype (default: float32)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Specific batch sizes to test (default: 32 to 1024)",
    )
    parser.add_argument(
        "--bw-threshold",
        type=float,
        default=0.005,
        help="Stop when bandwidth improvement drops below this threshold."
        " Lower values allow larger batch sizes.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model config",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    config = get_config_from_model(
        args.model,
        state_dtype=dtype_map[args.mamba_ssm_cache_dtype],
        trust_remote_code=args.trust_remote_code,
    )

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")

    optimal, results = find_optimal_batch_size(
        config,
        batch_sizes=args.batch_sizes,
        tp_size=args.tensor_parallel_size,
        bw_threshold=args.bw_threshold,
    )

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDED --max-num-seqs: {optimal}")
    print(f"{'=' * 60}")

    # Find the result for optimal batch
    optimal_result = next(
        (r for r in results if r.batch_size == optimal // args.tensor_parallel_size),
        None,
    )
    if optimal_result:
        bw = optimal_result.effective_bandwidth_tb_s
        print(f"\nAt batch={optimal}: bandwidth saturates at {bw:.2f} TB/s")
        print(f"  SSM kernel time: {optimal_result.time_ms:.2f}ms per step")

    print("\nUsage:")
    print(f"  vllm serve {args.model} --max-num-seqs {optimal}")


if __name__ == "__main__":
    main()
