#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic script to verify MXFP8 scale tensor formats match FlashInfer's
expectations. This tests whether our manual swizzling produces the same format
as FlashInfer's mxfp8_quantize.
"""

import sys

import torch


def log(msg):
    print(msg, flush=True)


def compare_scale_formats():
    """Compare our swizzle function output with FlashInfer's native swizzled output."""
    from flashinfer import mm_mxfp8, mxfp8_quantize

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        swizzle_mxfp8_scale,
    )

    torch.manual_seed(42)
    device = "cuda"

    # Test dimensions matching the model (LLaMA 8B)
    test_cases = [
        (256, 4096, 4096),  # input @ qkv
        (256, 4096, 6144),  # input @ down_proj
        (256, 14336, 4096),  # input @ up/gate proj
        (256, 4096, 14336),  # input @ down_proj
    ]

    log(f"\n{'=' * 70}")
    log("MXFP8 Scale Format Comparison: Manual Swizzle vs FlashInfer")
    log(f"{'=' * 70}")

    all_passed = True

    for M, N, K in test_cases:
        log(f"\n{'=' * 70}")
        log(f"Testing M={M}, N={N}, K={K}")
        log(f"{'=' * 70}")

        # Create test tensors
        input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02

        # ==================================================
        # Method 1: FlashInfer's mxfp8_quantize (swizzled)
        # ==================================================
        log("\n1. FlashInfer mxfp8_quantize (swizzled):")
        input_mxfp8, input_scale_fi = mxfp8_quantize(
            input_bf16, is_sf_swizzled_layout=True
        )
        weight_mxfp8, weight_scale_fi = mxfp8_quantize(
            weight_bf16, is_sf_swizzled_layout=True
        )

        log(
            f"   Input scale: shape={input_scale_fi.shape}, "
            f"dtype={input_scale_fi.dtype}, numel={input_scale_fi.numel()}"
        )
        log(
            f"   Weight scale: shape={weight_scale_fi.shape}, "
            f"dtype={weight_scale_fi.dtype}, numel={weight_scale_fi.numel()}"
        )

        # ==================================================
        # Method 2: FlashInfer non-swizzled + manual swizzle
        # ==================================================
        log("\n2. FlashInfer non-swizzled + manual swizzle:")
        _, input_scale_ns = mxfp8_quantize(input_bf16, is_sf_swizzled_layout=False)
        _, weight_scale_ns = mxfp8_quantize(weight_bf16, is_sf_swizzled_layout=False)

        # Reshape to 2D
        input_scale_2d = input_scale_ns.view(M, K // MXFP8_BLOCK_SIZE)
        weight_scale_2d = weight_scale_ns.view(N, K // MXFP8_BLOCK_SIZE)

        log(f"   Input scale 2D: shape={input_scale_2d.shape}")
        log(f"   Weight scale 2D: shape={weight_scale_2d.shape}")

        # Manual swizzle
        input_scale_manual = swizzle_mxfp8_scale(input_scale_2d, M=M, K=K)
        weight_scale_manual = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        log(
            f"   Input scale manual swizzled: shape={input_scale_manual.shape}, "
            f"numel={input_scale_manual.numel()}"
        )
        log(
            f"   Weight scale manual swizzled: shape={weight_scale_manual.shape}, "
            f"numel={weight_scale_manual.numel()}"
        )

        # ==================================================
        # Compare the two methods
        # ==================================================
        log("\n3. Comparison:")

        # Check sizes match
        input_size_match = input_scale_fi.numel() == input_scale_manual.numel()
        weight_size_match = weight_scale_fi.numel() == weight_scale_manual.numel()

        log(
            f"   Input scale size match: {input_size_match} "
            f"(FI={input_scale_fi.numel()}, Manual={input_scale_manual.numel()})"
        )
        log(
            f"   Weight scale size match: {weight_size_match} "
            f"(FI={weight_scale_fi.numel()}, Manual={weight_scale_manual.numel()})"
        )

        if input_size_match:
            input_values_match = torch.equal(input_scale_fi, input_scale_manual)
            log(f"   Input scale values match: {input_values_match}")
            if not input_values_match:
                diff = (input_scale_fi.int() - input_scale_manual.int()).abs()
                log(
                    f"   Input scale diff: max={diff.max()}, "
                    f"mean={diff.float().mean():.2f}"
                )

        if weight_size_match:
            weight_values_match = torch.equal(weight_scale_fi, weight_scale_manual)
            log(f"   Weight scale values match: {weight_values_match}")
            if not weight_values_match:
                diff = (weight_scale_fi.int() - weight_scale_manual.int()).abs()
                log(
                    f"   Weight scale diff: max={diff.max()}, "
                    f"mean={diff.float().mean():.2f}"
                )

        # ==================================================
        # Test mm_mxfp8 with both methods
        # ==================================================
        log("\n4. mm_mxfp8 test:")

        reference = torch.mm(input_bf16, weight_bf16.T)

        # Test with FlashInfer scales
        try:
            output_fi = mm_mxfp8(
                input_mxfp8,
                weight_mxfp8.T,
                input_scale_fi,
                weight_scale_fi,
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )
            cos_sim_fi = torch.nn.functional.cosine_similarity(
                reference.float().view(-1), output_fi.float().view(-1), dim=0
            ).item()
            log(f"   FlashInfer scales: cos_sim = {cos_sim_fi:.6f} ✓")
        except Exception as e:
            log(f"   FlashInfer scales: FAILED - {e}")
            cos_sim_fi = 0.0
            all_passed = False

        # Test with manual swizzled scales
        try:
            output_manual = mm_mxfp8(
                input_mxfp8,
                weight_mxfp8.T,
                input_scale_manual,
                weight_scale_manual,
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )
            cos_sim_manual = torch.nn.functional.cosine_similarity(
                reference.float().view(-1), output_manual.float().view(-1), dim=0
            ).item()
            log(f"   Manual swizzled scales: cos_sim = {cos_sim_manual:.6f} ✓")
        except Exception as e:
            log(f"   Manual swizzled scales: FAILED - {e}")
            cos_sim_manual = 0.0
            all_passed = False

        # Check if both pass
        if cos_sim_fi > 0.99 and cos_sim_manual > 0.99:
            log(f"\n   ✓ PASSED: Both methods work correctly for M={M}, N={N}, K={K}")
        elif cos_sim_fi > 0.99:
            log("\n   ✗ FAILED: Manual swizzle doesn't match FlashInfer format")
            all_passed = False
        else:
            log("\n   ✗ FAILED: Neither method works for this configuration")
            all_passed = False

    log(f"\n{'=' * 70}")
    if all_passed:
        log("SUMMARY: All tests PASSED ✓")
    else:
        log("SUMMARY: Some tests FAILED ✗")
        log("\nThe manual swizzle function may not produce the correct format.")
        log("Consider using FlashInfer's mxfp8_quantize for both input AND weight")
        log("by re-quantizing weights during process_weights_after_loading.")
    log(f"{'=' * 70}")

    return all_passed


def test_expected_scale_sizes():
    """Calculate expected swizzled scale sizes for various dimensions."""
    log("\n" + "=" * 70)
    log("Expected Swizzled Scale Sizes for F8_128x4 Layout")
    log("=" * 70)

    def calc_swizzled_size(M, K):
        """Calculate expected swizzled scale size."""
        num_m_tiles = (M + 127) // 128
        num_k_tiles = (K + 127) // 128
        m_padded = num_m_tiles * 128
        k_scale_padded = num_k_tiles * 4  # K/32 padded to 4
        return m_padded * k_scale_padded

    test_dims = [
        (256, 4096),  # Batch x hidden
        (4096, 4096),  # qkv projection
        (6144, 4096),  # qkv output
        (4096, 14336),  # up/gate proj
        (14336, 4096),  # down proj
        (28672, 4096),  # gate_up combined
    ]

    log(f"{'M':>8} {'K':>8} {'M_tiles':>8} {'K_tiles':>8} {'Expected Size':>15}")
    log("-" * 55)

    for M, K in test_dims:
        num_m_tiles = (M + 127) // 128
        num_k_tiles = (K + 127) // 128
        expected_size = calc_swizzled_size(M, K)
        log(f"{M:>8} {K:>8} {num_m_tiles:>8} {num_k_tiles:>8} {expected_size:>15}")


if __name__ == "__main__":
    log("MXFP8 Scale Format Diagnostic Tool")
    log("=" * 70)

    # First show expected sizes
    test_expected_scale_sizes()

    # Then run comparison
    try:
        success = compare_scale_formats()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
