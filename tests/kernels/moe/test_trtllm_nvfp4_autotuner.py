# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test to reproduce and verify the fix for the TRTLLM NVFP4 MoE autotuner
"using fallback tactic" bug.

The bug: vLLM was passing `hidden_states_scale` as a 1D flattened tensor
to the flashinfer `trtllm_fp4_block_scale_moe` kernel. The autotuner
expects this tensor to be 2D `[num_tokens, hidden_size // 16]` so that
dim 0 = `num_tokens`, matching all other input tensors. The shape mismatch
caused permanent autotuner cache misses and fallback to untuned tactics.

This test:
  1. Creates minimal NVFP4 MoE weights/inputs
  2. Runs autotuning (populates the cache)
  3. Runs inference with a FLATTENED scale (the bug) --> expects fallback
  4. Runs inference with a 2D scale (the fix) --> expects cache hit
"""

import logging
from collections.abc import Callable
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

# Skip immediately if not on Blackwell
if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "TRTLLM NVFP4 MoE requires Blackwell GPUs (SM10x).",
        allow_module_level=True,
    )

from flashinfer import ActivationType, RoutingMethodType, fp4_quantize
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Keep dimensions small so the test runs quickly, but large enough for the
# kernel to be valid.  All sizes must be multiples of 16 (FP4 block size).
NUM_EXPERTS = 8
HIDDEN_SIZE = 1024  # hidden_size
INTERMEDIATE_SIZE = 512  # intermediate_size (per-expert FFN width)
TOP_K = 2
SF_VEC_SIZE = 16  # NvFP4 scale-factor vector size


def _calculate_fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute global FP4 scale factor (offline calibration stand-in)."""
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_fp4(
    tensor: torch.Tensor, global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to NvFP4 with block scales."""
    fp4, sf = fp4_quantize(
        tensor.cuda(),
        global_sf.cuda(),
        SF_VEC_SIZE,
        False,  # use_ue8m0
        True,  # is_sf_swizzled_layout
    )
    return fp4, sf


def _quant_fp4_batched(
    weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize batched expert weights to NvFP4."""
    all_fp4, all_sf, all_gsf = [], [], []
    for i in range(num_experts):
        gsf = _calculate_fp4_global_scale(weights[i])
        fp4, sf = _quant_fp4(weights[i], gsf)
        all_fp4.append(fp4)
        all_sf.append(sf)
        all_gsf.append(gsf)
    return torch.stack(all_fp4), torch.stack(all_sf), torch.stack(all_gsf)


def _prepare_trtllm_fp4_weights(
    gemm1_weights_orig: torch.Tensor,
    gemm2_weights_orig: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """
    Prepare shuffled FP4 weights + scales for the TRTLLM kernel,
    mimicking the offline weight-preparation pipeline.
    """
    epilogue_tile_m = 128
    gemm1_intermediate = 2 * intermediate_size  # gated activation (SwiGLU)
    cache: dict[torch.Size, torch.Tensor] = {}

    # Quantize with swizzled layout (for the kernel)
    g1_fp4_sw, _, g1_gsf = _quant_fp4_batched(gemm1_weights_orig, num_experts)
    g2_fp4_sw, _, g2_gsf = _quant_fp4_batched(gemm2_weights_orig, num_experts)

    # Quantize with linear layout (for scale-factor shuffling)
    g1_sf_lin = []
    g2_sf_lin = []
    for i in range(num_experts):
        gsf1 = _calculate_fp4_global_scale(gemm1_weights_orig[i])
        fp4, sf = fp4_quantize(
            gemm1_weights_orig[i].cuda(), gsf1.cuda(), SF_VEC_SIZE, False, False
        )
        g1_sf_lin.append(sf)
        gsf2 = _calculate_fp4_global_scale(gemm2_weights_orig[i])
        fp4, sf = fp4_quantize(
            gemm2_weights_orig[i].cuda(), gsf2.cuda(), SF_VEC_SIZE, False, False
        )
        g2_sf_lin.append(sf)

    g1_sf_lin_t = torch.stack(g1_sf_lin)
    g2_sf_lin_t = torch.stack(g2_sf_lin)

    # Reshape for kernel expectations
    g1_fp4 = g1_fp4_sw.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate, hidden_size // 2
    )
    g1_sf = g1_sf_lin_t.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate, hidden_size // SF_VEC_SIZE
    )
    g2_fp4 = g2_fp4_sw.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    g2_sf = g2_sf_lin_t.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // SF_VEC_SIZE
    )

    # Shuffle weights + scales per expert
    g1_w_shuf, g1_s_shuf = [], []
    g2_w_shuf, g2_s_shuf = [], []
    for i in range(num_experts):
        # GEMM1 weights
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, g1_fp4[i].view(torch.uint8), epilogue_tile_m
        )
        g1_w_shuf.append(
            g1_fp4[i].view(torch.uint8)[perm.to(g1_fp4.device)].contiguous()
        )
        perm_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            g1_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=SF_VEC_SIZE,
        )
        g1_s_shuf.append(
            block_scale_interleave(
                g1_sf[i].view(torch.uint8)[perm_sf.to(g1_sf.device)].contiguous()
            )
        )

        # GEMM2 weights
        perm = get_w2_permute_indices_with_cache(
            cache, g2_fp4[i].view(torch.uint8), epilogue_tile_m
        )
        g2_w_shuf.append(
            g2_fp4[i].view(torch.uint8)[perm.to(g2_fp4.device)].contiguous()
        )
        perm_sf = get_w2_permute_indices_with_cache(
            cache,
            g2_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=SF_VEC_SIZE,
        )
        g2_s_shuf.append(
            block_scale_interleave(
                g2_sf[i].view(torch.uint8)[perm_sf.to(g2_sf.device)].contiguous()
            )
        )

    g1_w = torch.stack(g1_w_shuf)
    g1_s = (
        torch.stack(g1_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate, hidden_size // SF_VEC_SIZE)
    )
    g2_w = torch.stack(g2_w_shuf)
    g2_s = (
        torch.stack(g2_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // SF_VEC_SIZE)
    )

    # Compute per-expert output scale scalars
    hidden_sf_global = torch.ones(num_experts, device="cuda", dtype=torch.float32)
    scale_c = hidden_sf_global * g1_gsf.cuda() / (torch.ones_like(g1_gsf).cuda())
    scale_gate = hidden_sf_global * g1_gsf.cuda()
    scale_c2 = torch.ones_like(g2_gsf).cuda() * g2_gsf.cuda()

    return {
        "gemm1_weights": g1_w,
        "gemm1_scales": g1_s,
        "gemm2_weights": g2_w,
        "gemm2_scales": g2_s,
        "scale_c": scale_c,
        "scale_gate": scale_gate,
        "scale_c2": scale_c2,
        "hidden_sf_global": hidden_sf_global,
        "g1_gsf": g1_gsf,
    }


def _call_moe(
    hidden_states_fp4: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    routing_logits: torch.Tensor,
    static: dict[str, torch.Tensor],
    tune_max_num_tokens: int = 16384,
) -> list[torch.Tensor]:
    return trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=static["gemm1_weights"],
        gemm1_weights_scale=static["gemm1_scales"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=static["gemm2_weights"],
        gemm2_weights_scale=static["gemm2_scales"],
        gemm2_bias=None,
        output1_scale_scalar=static["scale_c"],
        output1_scale_gate_scalar=static["scale_gate"],
        output2_scale_scalar=static["scale_c2"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        do_finalize=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=tune_max_num_tokens,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    gemm1 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return _prepare_trtllm_fp4_weights(
        gemm1, gemm2, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS
    )


def _make_inputs(
    num_tokens: int, static: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create quantized hidden states + routing logits for *num_tokens*."""
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    gsf = static["hidden_sf_global"][0]  # scalar global scale
    fp4, sf = fp4_quantize(
        hidden,
        gsf,
        SF_VEC_SIZE,
        False,
        False,  # linear layout
    )
    # sf comes back as [num_tokens, hidden_size // SF_VEC_SIZE] after view
    sf = sf.view(torch.float8_e4m3fn).reshape(num_tokens, -1)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
    return fp4, sf, logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_autotuner() -> None:
    """Reset the singleton AutoTuner so each test starts with a clean cache."""
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()


class TacticTracker:
    """Context manager that monkey-patches AutoTuner.choose_one to record
    the (tactic, is_cache_hit) for every non-tuning call."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._orig_choose_one: Callable[..., Any] | None = None

    def __enter__(self) -> "TacticTracker":
        tuner = AutoTuner.get()
        self._orig_choose_one = tuner.choose_one

        tracker = self  # capture for closure

        def _patched_choose_one(custom_op, runners, tuning_config, inputs, **kw):
            assert tracker._orig_choose_one is not None
            runner, tactic = tracker._orig_choose_one(
                custom_op, runners, tuning_config, inputs, **kw
            )
            if not tuner.is_tuning_mode:
                is_fallback = tactic == -1
                tracker.calls.append(
                    {
                        "op": custom_op,
                        "tactic": tactic,
                        "is_fallback": is_fallback,
                    }
                )
                logger.info(
                    "[TacticTracker] %s --> tactic=%s (%s)",
                    custom_op,
                    tactic,
                    "FALLBACK" if is_fallback else "TUNED [tile_N, config_N]",
                )
            return runner, tactic

        tuner.choose_one = _patched_choose_one
        return self

    def __exit__(self, *exc: object) -> None:
        AutoTuner.get().choose_one = self._orig_choose_one

    @property
    def num_fallbacks(self) -> int:
        return sum(1 for c in self.calls if c["is_fallback"])

    @property
    def num_tuned(self) -> int:
        return sum(1 for c in self.calls if not c["is_fallback"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_2d_scale",
    [True, False],
    ids=["2d_scale_cache_hit", "flat_scale_fallback"],
)
def test_trtllm_nvfp4_autotuner_tactic_selection(
    static_weights: dict[str, torch.Tensor], use_2d_scale: bool
) -> None:
    """
    Simulate a realistic prefill workload: batch_size=64 with 8000 tokens
    per prompt.  In vLLM this is chunked, so the MoE kernel sees batches
    up to max_num_batched_tokens (commonly 8192 to 16384).

    Parametrized:
      - use_2d_scale=True  --> 2D scale [num_tokens, hidden_size//16] (fix)
                              --> all calls should use tuned tactics
      - use_2d_scale=False --> 1D flattened scale (bug)
                              --> all calls should fall back to tactic -1
    """
    _reset_autotuner()
    tuner = AutoTuner.get()

    large_tune_max = 16384  # covers prefill-sized batches

    # --- Phase 1: Autotune at the max batch size ---------------------------
    # Always autotune with 2D scale (the correct shape)
    autotune_tokens = 8192
    fp4, sf_2d, logits = _make_inputs(autotune_tokens, static_weights)

    with autotune():
        _call_moe(
            fp4,
            sf_2d,
            logits,
            static_weights,
            tune_max_num_tokens=large_tune_max,
        )
    torch.cuda.synchronize()

    cache_size_after_tune = len(tuner.profiling_cache)
    assert cache_size_after_tune > 0
    logger.info(
        "After autotuning (tune_max=%d): %d cached entries",
        large_tune_max,
        cache_size_after_tune,
    )

    # --- Phase 2: Inference at various realistic token counts --------------
    test_token_counts = [
        64,
        128,
        1024,
        4096,
        8192,
    ]

    with TacticTracker() as tracker:
        for num_tokens in test_token_counts:
            fp4_i, sf_2d_i, logits_i = _make_inputs(num_tokens, static_weights)

            # 2D (fix) --> cache hit; 1D flatten (bug) --> fallback tactic
            scale = sf_2d_i if use_2d_scale else sf_2d_i.flatten()

            result = _call_moe(
                fp4_i,
                scale,
                logits_i,
                static_weights,
                tune_max_num_tokens=large_tune_max,
            )
            torch.cuda.synchronize()

            assert result is not None
            assert result[0].shape[0] == num_tokens

    for num_tokens, call in zip(test_token_counts, tracker.calls):
        logger.info(
            "num_tokens=%d: scale_shape=%s --> tactic=%s (%s)",
            num_tokens,
            "2D" if use_2d_scale else "1D-flat",
            call["tactic"],
            "TUNED" if not call["is_fallback"] else "FALLBACK",
        )

    if use_2d_scale:
        # All calls should use tuned tactics (cache hit)
        assert tracker.num_fallbacks == 0, (
            f"Got {tracker.num_fallbacks} fallback(s) with 2D scale — "
            f"expected all cache hits"
        )
        assert tracker.num_tuned == len(test_token_counts)
        logger.info(
            "VERIFIED (2D scale): all %d calls used tuned tactics",
            tracker.num_tuned,
        )
    else:
        # All calls should fall back (cache miss due to shape mismatch)
        assert tracker.num_tuned == 0, (
            f"Got {tracker.num_tuned} tuned call(s) with flattened scale — "
            f"expected all fallbacks"
        )
        assert tracker.num_fallbacks == len(test_token_counts)
        logger.info(
            "VERIFIED (1D flat scale): all %d calls used fallback tactic -1",
            tracker.num_fallbacks,
        )
