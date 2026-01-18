# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


# ============================================================================
# Triton Kernel for MXFP8 Block-Scaled Matrix Multiplication
# ============================================================================


@triton.jit
def _mxfp8_triton_block_scaled_mm_kernel(
    # Pointers to inputs and output
    A,  # Input activation: [M, K] in float8_e4m3fn
    B,  # Weight: [N, K] in float8_e4m3fn (row-major, will be transposed)
    C,  # Output: [M, N] in bfloat16
    As,  # Input scales: [M, K // 32] in uint8 (e8m0 format)
    Bs,  # Weight scales: [N, K // 32] in uint8 (e8m0 format)
    # Shape for matmul
    M,
    N,
    K,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Should be multiple of 32 (MXFP8_BLOCK_SIZE)
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for MXFP8 block-scaled matrix multiplication.

    Computes C = A @ B.T where:
    - A is [M, K] activation in float8_e4m3fn with per-32-element row scales
    - B is [N, K] weight in float8_e4m3fn with per-32-element row scales
    - C is [M, N] output in bfloat16

    Scale format: e8m0 (exponent-only, stored as uint8)
    To convert e8m0 to bf16: (scale.to(int16) << 7).view(bfloat16)
    """
    # Program ID and work distribution
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to A and B data
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # Pointers to scales (one scale per 32 elements along K)
    # As shape: [M, K // 32], Bs shape: [N, K // 32]
    As_ptrs = As + offs_am * stride_As_m
    Bs_ptrs = Bs + offs_bn * stride_Bs_n

    # Accumulator in float32 for numerical accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # MXFP8 uses block size of 32 for scales
    MXFP8_GROUP_K: tl.constexpr = 32

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K

        # Load A and B tiles with masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_start, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k_start, other=0.0)

        # For each MXFP8 group (32 elements) within this K block,
        # we need to apply the corresponding scale
        # Since BLOCK_SIZE_K should be a multiple of 32, we process
        # BLOCK_SIZE_K // 32 scale groups per iteration

        # Scale offset for this K block
        offs_ks = k_start // MXFP8_GROUP_K

        # Load scales as uint8 and convert to bfloat16
        # e8m0 format: exponent only, convert via (val << 7) as bf16
        a_s_uint8 = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s_uint8 = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        # Convert e8m0 scales to bfloat16
        # This is equivalent to: (scale.to(int16) << 7).view(bfloat16)
        # In Triton, we need to do bit manipulation
        a_s_int16 = a_s_uint8.to(tl.int16)
        b_s_int16 = b_s_uint8.to(tl.int16)
        a_s_bf16_bits = a_s_int16 << 7
        b_s_bf16_bits = b_s_int16 << 7
        a_s = a_s_bf16_bits.to(tl.bfloat16, bitcast=True)
        b_s = b_s_bf16_bits.to(tl.bfloat16, bitcast=True)

        # Compute scaled dot product
        # For MXFP8, each 32-element group has one scale
        # When BLOCK_SIZE_K == 32, we have one scale per iteration
        # For larger BLOCK_SIZE_K, we'd need to handle multiple scales

        # Simple case: BLOCK_SIZE_K == MXFP8_GROUP_K (32)
        # Each tile gets one scale factor
        accumulator += tl.dot(a, tl.trans(b)) * (
            a_s[:, None].to(tl.float32) * b_s[None, :].to(tl.float32)
        )

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulator to output dtype and store
    c = accumulator.to(tl.bfloat16)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def get_mxfp8_kernel_configs(N: int, K: int) -> dict | None:
    """
    Return optimized configurations for the MXFP8 kernel.
    Returns a dictionary mapping batch sizes to kernel configurations.
    """
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=mxfp8,block_shape=[1,32].json"
    )

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        import json

        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for MXFP8 kernel.",
                config_file_path,
            )
            return {int(key): val for key, val in json.load(f).items()}

    return None


def mxfp8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    MXFP8 block-scaled matrix multiplication using Triton.

    Computes C = A @ B.T where:
    - A: [M, K] activation in float8_e4m3fn
    - B: [N, K] weight in float8_e4m3fn
    - As: [M, K // 32] activation scales in uint8 (e8m0 format)
    - Bs: [N, K // 32] weight scales in uint8 (e8m0 format)

    Returns:
        C: [M, N] output in output_dtype (bfloat16)
    """
    assert A.dtype == MXFP8_VALUE_DTYPE, f"A must be {MXFP8_VALUE_DTYPE}, got {A.dtype}"
    assert B.dtype == MXFP8_VALUE_DTYPE, f"B must be {MXFP8_VALUE_DTYPE}, got {B.dtype}"
    assert output_dtype == torch.bfloat16, "Only bfloat16 output is supported"

    # A: [M, K], B: [N, K] (will compute A @ B.T)
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    N, K_b = B.shape
    assert K_b == K, f"K dimension mismatch: {K} vs {K_b}"

    # Validate scale shapes
    assert As.shape == (M, K // MXFP8_BLOCK_SIZE), (
        f"As shape mismatch: expected {(M, K // MXFP8_BLOCK_SIZE)}, got {As.shape}"
    )
    assert Bs.shape == (N, K // MXFP8_BLOCK_SIZE), (
        f"Bs shape mismatch: expected {(N, K // MXFP8_BLOCK_SIZE)}, got {Bs.shape}"
    )

    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)

    # Get kernel configuration
    configs = get_mxfp8_kernel_configs(N, K)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default configuration optimized for MXFP8
        # BLOCK_SIZE_K must be 32 to match MXFP8 block size
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,  # Must match MXFP8_BLOCK_SIZE
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _mxfp8_triton_block_scaled_mm_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        **config,
    )

    return C


def _mxfp8_triton_mm_func(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Wrapper for custom op registration."""
    return mxfp8_triton_block_scaled_mm(
        qinput, weight, input_scale, weight_scale, output_dtype
    )


def _mxfp8_triton_mm_fake(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    M = qinput.size(0)
    N = weight.size(0)
    return torch.empty((M, N), dtype=output_dtype, device=qinput.device)


direct_register_custom_op(
    op_name="mxfp8_triton_mm",
    op_func=_mxfp8_triton_mm_func,
    fake_impl=_mxfp8_triton_mm_fake,
)


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from flashinfer import mxfp8_quantize as mxfp8_e4m3_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=is_sf_swizzled_layout)
    if x_scales.ndim == 1:
        if is_sf_swizzled_layout:
            # TODO: check this, maybe not required?
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
            # We must use the padded dimensions, not the original input dimensions
            def _round_up(val: int, mult: int) -> int:
                return (val + mult - 1) // mult * mult

            M = x.size(0)
            K = x.size(-1) // MXFP8_BLOCK_SIZE
            M_padded = _round_up(M, 128)
            K_padded = _round_up(K, 4)
            x_scales = x_scales.view(M_padded, K_padded)
        else:
            x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    """
    Cast MXFP8 scales from uint8 to BF16.
    The scales are stored in uint8 format and need to be converted to BF16
    by left-shifting by 7 bits (to form the exponent) and reinterpreting
    as bfloat16.
    Args:
        scales: uint8 tensor containing MXFP8 scales
    Returns:
        BF16 tensor with the converted scales
    """
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor to BF16.
    Args:
        x: FP8 E4M3 tensor to dequantize
        scales: uint8 tensor containing MXFP8 scales
    Returns:
        BF16 dequantized tensor
    """
    scales_bf16 = _cast_mxfp8_scales_to_bf16(scales)
    # Repeat scales along the last dimension to match the block size
    scales_expanded = scales_bf16.reshape(*x.shape[:-1], -1).repeat_interleave(
        MXFP8_BLOCK_SIZE, dim=-1
    )
    return x.to(torch.bfloat16) * scales_expanded


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile tracing.
    Returns empty tensors with the correct shapes and dtypes.
    """
    # FP8 quantized data has same shape as input
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    # Compute scale shape: one scale per block of 32 elements along last dim
    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                (M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                (B, M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        # Fallback for other dimensions
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=mxfp8_e4m3_quantize,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    """
    This class executes a MXFP8 linear layer.

    Supports three backends:
    - "triton": Uses custom Triton kernel (default)
    - "scaled_mm": Uses torch._scaled_mm (requires hardware support)
    - "fallback": Dequantizes to bf16 and uses standard linear (for debugging)
    """

    def __init__(self, use_fallback: bool = False, backend: str | None = None):
        """
        Initialize MXFP8 linear operation.

        Args:
            use_fallback: If True, use fallback (dequantize) mode.
                         Deprecated, use backend="fallback" instead.
            backend: One of "triton", "scaled_mm", or "fallback".
                    If None, defaults to "triton" (or "fallback" if use_fallback=True)
        """
        if backend is not None:
            assert backend in ("scaled_mm", "triton", "fallback"), (
                f"Unknown backend: {backend}. "
                "Supported: 'triton', 'scaled_mm', 'fallback'"
            )
            self.backend = backend
        else:
            self.backend = "fallback" if use_fallback else "triton"

        # Keep for backwards compatibility
        self.use_fallback = use_fallback or (backend == "fallback")

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backend == "fallback":
            return self._apply_fallback(input, weight, weight_scale, out_dtype, bias)
        elif self.backend == "triton":
            return self._apply_triton(input, weight, weight_scale, out_dtype, bias)
        else:  # scaled_mm
            return self._apply_scaled_mm(input, weight, weight_scale, out_dtype, bias)

    def _apply_fallback(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fallback implementation using manual dequantization for debugging."""
        # weight_scale comes in as float8_e8m0fnu
        # after process_weights_after_loading
        # It may be padded to [N_padded, K/32] and flattened
        # Convert back to uint8 for dequantization
        weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)

        out_features, in_features = weight.shape
        # Number of scale blocks along K dimension
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Compute padded dimensions (same logic as process_weights_after_loading)
        out_features_padded = (out_features + 127) // 128 * 128

        # Reshape to padded 2D, then slice to get original shape
        weight_scale_2d_padded = weight_scale_uint8.view(out_features_padded, scale_k)
        weight_scale_2d = weight_scale_2d_padded[:out_features, :]

        # Dequantize weight to bf16
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)

        # Standard linear operation
        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_triton(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Triton kernel implementation for MXFP8 linear.

        Uses the custom Triton kernel for MXFP8 block-scaled matrix multiplication.
        This is useful when torch._scaled_mm doesn't support MXFP8 or for platforms
        without native MXFP8 support.
        """
        assert weight.dtype == MXFP8_VALUE_DTYPE
        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"
        assert input.dtype == torch.bfloat16

        # Get weight dimensions
        out_features, in_features = weight.shape
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Quantize input to MXFP8 (non-swizzled for Triton kernel)
        swizzled = False
        input_flat = input.view(-1, input.shape[-1])
        input_mxfp8, input_mxfp8_scales = torch.ops.vllm.mxfp8_quantize(
            input_flat, swizzled
        )

        # Convert weight scales from float8_e8m0fnu to uint8 for Triton kernel
        # The Triton kernel handles the e8m0 → bf16 conversion internally
        weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)

        # Handle padded weight scales (same as fallback)
        out_features_padded = (out_features + 127) // 128 * 128
        weight_scale_2d = weight_scale_uint8.view(out_features_padded, scale_k)[
            :out_features, :
        ].contiguous()

        # Ensure input scales are 2D [M, K//32]
        input_mxfp8_scales = input_mxfp8_scales.view(
            input_flat.shape[0], -1
        ).contiguous()

        # Call the Triton kernel
        output = torch.ops.vllm.mxfp8_triton_mm(
            input_mxfp8,
            weight,
            input_mxfp8_scales,
            weight_scale_2d,
            out_dtype,
        )

        # Reshape output to match input batch dimensions
        output = output.view(*input.shape[:-1], out_features)

        if bias is not None:
            output = output + bias

        return output

    def _apply_scaled_mm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Weights should be mxfp8, weight_scale
        # is pre-processed to float8_e8m0fnu
        assert weight.dtype == MXFP8_VALUE_DTYPE
        # weight_scale is already pre-processed
        # in process_weights_after_loading
        assert weight_scale.dtype == torch.float8_e8m0fnu

        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"

        # From bf16 to mxfp8
        assert input.dtype == torch.bfloat16

        swizzled = True
        input_mxfp8, input_mxfp8_scales = torch.ops.vllm.mxfp8_quantize(input, swizzled)

        # For Blockwise 1x32 scaling, a and b should be float8,
        # scales should be float8_e8m0fnu and 1D contiguous
        # Use .view() to reinterpret uint8 bytes as float8_e8m0fnu
        # (not .to() which converts values)
        input_mxfp8_scales = input_mxfp8_scales.view(torch.float8_e8m0fnu).flatten()

        output = torch._scaled_mm(
            input_mxfp8,
            weight.t(),
            scale_a=input_mxfp8_scales,
            scale_b=weight_scale,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        if bias is not None:
            output = output + bias

        return output
