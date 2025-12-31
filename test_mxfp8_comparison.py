#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test to compare ModelOpt MXFP8 quantization vs FlashInfer MXFP8 quantization
and verify mm_mxfp8 works correctly with both.
"""

import time

import torch
import torch.nn.functional as F
from flashinfer import mm_mxfp8, mxfp8_quantize


def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)


def test_mxfp8_quantization_comparison():
    """Compare quantization algorithms and verify mm_mxfp8 accuracy."""
    torch.manual_seed(42)
    device = "cuda"

    # Use typical LLM dimensions
    M, N, K = 256, 4096, 4096

    log(f"\n{'=' * 60}")
    log(f"Testing MXFP8 mm with dimensions: M={M}, N={N}, K={K}")
    log(f"{'=' * 60}")

    # Create test data with typical model weight statistics (small std)
    input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02

    # Reference output (BF16 matmul)
    reference = torch.mm(input_bf16, weight_bf16.T)
    log(
        f"\nReference (BF16) output: min={reference.min():.4f}, "
        f"max={reference.max():.4f}, std={reference.std():.4f}"
    )

    # ==================================================
    # Test 1: FlashInfer quantization + mm_mxfp8 (swizzled)
    # ==================================================
    log(f"\n{'=' * 60}")
    log("Test 1: FlashInfer quantization (swizzled)")
    log(f"{'=' * 60}")

    input_mxfp8_s, input_scale_s = mxfp8_quantize(
        input_bf16, is_sf_swizzled_layout=True
    )
    weight_mxfp8_s, weight_scale_s = mxfp8_quantize(
        weight_bf16, is_sf_swizzled_layout=True
    )

    log(f"Input MXFP8: shape={input_mxfp8_s.shape}, dtype={input_mxfp8_s.dtype}")
    log(f"Input scale: shape={input_scale_s.shape}, dtype={input_scale_s.dtype}")
    log(f"Weight MXFP8: shape={weight_mxfp8_s.shape}, dtype={weight_mxfp8_s.dtype}")
    log(f"Weight scale: shape={weight_scale_s.shape}, dtype={weight_scale_s.dtype}")

    log("Running mm_mxfp8 (first call may trigger autotuning, please wait ~30s)...")
    start = time.time()
    output_swizzled = mm_mxfp8(
        input_mxfp8_s,
        weight_mxfp8_s.T,  # [N, K] -> [K, N]
        input_scale_s,
        weight_scale_s,  # 1D swizzled
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    torch.cuda.synchronize()
    log(f"mm_mxfp8 completed in {time.time() - start:.2f}s")

    cos_sim_s = F.cosine_similarity(
        reference.float().view(-1), output_swizzled.float().view(-1), dim=0
    )
    rel_err_s = (reference - output_swizzled).abs().mean() / reference.abs().mean()
    log(
        f"Swizzled output: min={output_swizzled.min():.4f}, "
        f"max={output_swizzled.max():.4f}"
    )
    log(f"Cosine similarity: {cos_sim_s:.6f}")
    log(f"Relative error: {rel_err_s:.6f}")

    # ==================================================
    # Test 2: FlashInfer quantization + mm_mxfp8 (non-swizzled)
    # ==================================================
    log(f"\n{'=' * 60}")
    log("Test 2: FlashInfer quantization (non-swizzled)")
    log(f"{'=' * 60}")

    input_mxfp8_ns, input_scale_ns = mxfp8_quantize(
        input_bf16, is_sf_swizzled_layout=False
    )
    weight_mxfp8_ns, weight_scale_ns = mxfp8_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    # Reshape 1D scales to 2D for non-swizzled
    input_scale_2d = input_scale_ns.view(M, K // 32)
    weight_scale_2d = weight_scale_ns.view(N, K // 32)

    log(f"Input MXFP8: shape={input_mxfp8_ns.shape}, dtype={input_mxfp8_ns.dtype}")
    log(f"Input scale 2D: shape={input_scale_2d.shape}, dtype={input_scale_2d.dtype}")
    log(f"Weight MXFP8: shape={weight_mxfp8_ns.shape}, dtype={weight_mxfp8_ns.dtype}")
    log(
        f"Weight scale 2D: shape={weight_scale_2d.shape}, dtype={weight_scale_2d.dtype}"
    )

    output_non_swizzled = mm_mxfp8(
        input_mxfp8_ns,
        weight_mxfp8_ns.T,  # [N, K] -> [K, N]
        input_scale_2d,  # 2D [M, K/32]
        weight_scale_2d.t(),  # 2D [N, K/32] -> [K/32, N]
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )

    cos_sim_ns = F.cosine_similarity(
        reference.float().view(-1), output_non_swizzled.float().view(-1), dim=0
    )
    rel_err_ns = (reference - output_non_swizzled).abs().mean() / reference.abs().mean()
    log(
        f"Non-swizzled output: min={output_non_swizzled.min():.4f}, "
        f"max={output_non_swizzled.max():.4f}"
    )
    log(f"Cosine similarity: {cos_sim_ns:.6f}")
    log(f"Relative error: {rel_err_ns:.6f}")

    # ==================================================
    # Test 3: ModelOpt-style quantization + mm_mxfp8
    # ==================================================
    log(f"\n{'=' * 60}")
    log("Test 3: ModelOpt-style quantization (simulated)")
    log(f"{'=' * 60}")

    # Simulate ModelOpt quantization:
    # 1. Compute per-block amax
    # 2. Compute E8M0 scale as ceil(log2(amax / 448)) + 127
    # 3. Quantize values to FP8 E4M3

    def modelopt_quantize(x, block_size=32):
        """Simulate ModelOpt MXFP8 quantization."""
        E4M3_MAX = 448.0
        BIAS = 127

        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        num_rows, num_cols = x_flat.shape

        # Pad K to multiple of block_size
        pad_cols = (block_size - num_cols % block_size) % block_size
        if pad_cols > 0:
            x_flat = F.pad(x_flat, (0, pad_cols))

        # Reshape to blocks
        num_blocks = x_flat.shape[1] // block_size
        x_blocked = x_flat.view(num_rows, num_blocks, block_size)

        # Compute per-block amax
        amax = x_blocked.abs().max(dim=-1).values.float()  # [num_rows, num_blocks]

        # Compute E8M0 exponent: ceil(log2(amax / 448))
        descale = amax / E4M3_MAX
        log2_descale = torch.where(
            descale > 0,
            torch.log2(descale),
            torch.tensor(-127.0, device=x.device),
        )
        e8m0_exponent = torch.ceil(log2_descale)
        e8m0_exponent = torch.clamp(e8m0_exponent, min=-127, max=127)

        # Biased uint8 scale
        scale_uint8 = (e8m0_exponent + BIAS).to(torch.uint8)

        # Compute scale for quantization: 2^e8m0_exponent
        scale_float = torch.pow(2.0, e8m0_exponent).unsqueeze(-1)

        # Quantize: x_fp8 = x / scale, then clamp to FP8 range
        x_scaled = x_blocked / scale_float
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)

        # Reshape back
        x_fp8 = x_fp8.view(num_rows, -1)[:, :num_cols].view(shape)
        scale_uint8 = scale_uint8.view(num_rows, -1)

        # Restore original batch dimensions if needed
        if x.ndim > 2:
            scale_shape = list(x.shape[:-1]) + [num_blocks]
            scale_uint8 = scale_uint8.view(scale_shape)

        return x_fp8, scale_uint8

    # Quantize input with FlashInfer (for fair comparison - input is dynamic)
    input_mxfp8_mo, input_scale_mo_1d = mxfp8_quantize(
        input_bf16, is_sf_swizzled_layout=False
    )
    input_scale_mo = input_scale_mo_1d.view(M, K // 32)

    # Quantize weight with ModelOpt-style
    weight_mxfp8_mo, weight_scale_mo = modelopt_quantize(weight_bf16)

    log(f"Input MXFP8 (FlashInfer): shape={input_mxfp8_mo.shape}")
    log(f"Input scale: shape={input_scale_mo.shape}")
    log(f"Weight MXFP8 (ModelOpt): shape={weight_mxfp8_mo.shape}")
    log(f"Weight scale (ModelOpt): shape={weight_scale_mo.shape}")

    # Compare ModelOpt vs FlashInfer weight quantization
    log("\nWeight FP8 comparison (FlashInfer vs ModelOpt):")
    fp8_diff = (weight_mxfp8_ns.float() - weight_mxfp8_mo.float()).abs()
    log(f"  FP8 max diff: {fp8_diff.max():.4f}")
    log(f"  FP8 mean diff: {fp8_diff.mean():.4f}")
    log(f"  % identical: {(fp8_diff == 0).float().mean() * 100:.1f}%")

    # Reshape FlashInfer weight scale for comparison
    weight_scale_ns_2d = weight_scale_ns.view(N, K // 32)
    scale_diff = (weight_scale_ns_2d.float() - weight_scale_mo.float()).abs()
    log("\nScale comparison (FlashInfer vs ModelOpt):")
    log(f"  Scale max diff: {scale_diff.max():.0f}")
    log(f"  Scale mean diff: {scale_diff.mean():.4f}")
    log(f"  % identical: {(scale_diff == 0).float().mean() * 100:.1f}%")

    try:
        output_modelopt = mm_mxfp8(
            input_mxfp8_mo,
            weight_mxfp8_mo.T,  # [N, K] -> [K, N]
            input_scale_mo,  # 2D [M, K/32]
            weight_scale_mo.t().contiguous(),  # 2D [N, K/32] -> [K/32, N]
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )

        cos_sim_mo = F.cosine_similarity(
            reference.float().view(-1), output_modelopt.float().view(-1), dim=0
        )
        rel_err_mo = (reference - output_modelopt).abs().mean() / reference.abs().mean()
        log(
            f"\nModelOpt-style output: min={output_modelopt.min():.4f}, "
            f"max={output_modelopt.max():.4f}"
        )
        log(f"Cosine similarity: {cos_sim_mo:.6f}")
        log(f"Relative error: {rel_err_mo:.6f}")
    except Exception as e:
        log(f"\nModelOpt mm_mxfp8 FAILED: {e}")
        cos_sim_mo = 0.0

    # ==================================================
    # Test 4: Mixed quantization (FlashInfer input + ModelOpt weight)
    # ==================================================
    log(f"\n{'=' * 60}")
    log("Test 4: Mixed quantization (FlashInfer input + ModelOpt-style weight)")
    log(f"{'=' * 60}")

    try:
        output_mixed = mm_mxfp8(
            input_mxfp8_ns,
            weight_mxfp8_mo.T,  # ModelOpt weight
            input_scale_2d,  # FlashInfer input scale
            weight_scale_mo.t().contiguous(),  # ModelOpt weight scale
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )

        cos_sim_mixed = F.cosine_similarity(
            reference.float().view(-1), output_mixed.float().view(-1), dim=0
        )
        rel_err_mixed = (reference - output_mixed).abs().mean() / reference.abs().mean()
        log(f"Mixed output: min={output_mixed.min():.4f}, max={output_mixed.max():.4f}")
        log(f"Cosine similarity: {cos_sim_mixed:.6f}")
        log(f"Relative error: {rel_err_mixed:.6f}")
    except Exception as e:
        log(f"Mixed mm_mxfp8 FAILED: {e}")
        cos_sim_mixed = 0.0

    # ==================================================
    # Summary
    # ==================================================
    log(f"\n{'=' * 60}")
    log("SUMMARY")
    log(f"{'=' * 60}")
    log(f"Test 1 (FlashInfer swizzled):     cos_sim = {cos_sim_s:.6f}")
    log(f"Test 2 (FlashInfer non-swizzled): cos_sim = {cos_sim_ns:.6f}")
    log(f"Test 3 (ModelOpt-style):          cos_sim = {cos_sim_mo:.6f}")
    log(f"Test 4 (Mixed):                   cos_sim = {cos_sim_mixed:.6f}")

    if cos_sim_s > 0.9 and cos_sim_ns > 0.9:
        log("\n✓ FlashInfer quantization works correctly with mm_mxfp8")

    if cos_sim_mo < 0.9:
        log("\n✗ ModelOpt-style quantization has issues with mm_mxfp8!")
        log("  This suggests a format incompatibility between ModelOpt and CUTLASS.")

    if cos_sim_mixed < 0.9:
        log("\n✗ Mixed quantization has issues!")
        log("  This confirms weight quantization format is incompatible.")


if __name__ == "__main__":
    test_mxfp8_quantization_comparison()
