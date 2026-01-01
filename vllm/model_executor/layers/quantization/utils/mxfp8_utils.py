# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import swizzle_blockscale
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


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
            K = x.size(-1) // 32  # MXFP8 block size is 32
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
        32, dim=-1
    )
    return x.to(torch.bfloat16) * scales_expanded


# TODO: remove this
def mxfp8_e4m3_quantize_python(
    data: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # return mxfp8_e4m3_quantize(data, is_sf_swizzled_layout)

    # ---- NEW: support 3D ----
    if data.ndim == 2:
        data = data.unsqueeze(0)  # treat as batch=1
        squeeze_back = True
    elif data.ndim == 3:
        squeeze_back = False
    else:
        raise AssertionError("Input must be 2D or 3D (B, M, N)")

    B, M, N = data.shape
    block_size1 = 32
    block_size0 = 1

    assert N % block_size1 == 0 and M % block_size0 == 0, (
        "Data shape must be a multiple of tile size [1, 32]"
    )

    # FP8 limits
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    # Compute block grid
    blk_m = M // block_size0
    blk_n = N // block_size1

    # Reshape to blocks
    # B, blk_m, block0, blk_n, block1
    data_blocks = data.reshape(B, blk_m, block_size0, blk_n, block_size1)

    # Permute to (B, blk_m, blk_n, block0, block1)
    data_blocks = data_blocks.permute(0, 1, 3, 2, 4)

    # Flatten blocks → (B, blk_m, blk_n, block0*block1)
    data_blocks = data_blocks.to(torch.float32).flatten(start_dim=3)

    # Compute per-block max
    max_abs = torch.amax(torch.abs(data_blocks), dim=-1, keepdim=True)

    # Compute exponent
    descale = max_abs / max_dtype
    exponent = torch.ceil(torch.log2(descale))
    exponent = torch.clamp(exponent, min=-127, max=127) + 127
    exponent_uint8 = exponent.to(torch.uint8)

    # Compute scale_fp = 2^(127 - exponent)
    scale_fp = torch.where(
        exponent_uint8 == 0, 1.0, torch.exp2(127 - exponent_uint8.to(torch.float32))
    )

    # Scale + clamp
    data_lp = torch.clamp(data_blocks * scale_fp, min=-max_dtype, max=max_dtype)

    # Convert to FP8
    fp_data = data_lp.to(torch.float8_e4m3fn)

    # Undo block flattening
    fp_data = fp_data.reshape(B, blk_m, blk_n, block_size0, block_size1)
    fp_data = fp_data.permute(0, 1, 3, 2, 4)
    fp_data = fp_data.reshape(B, M, N)

    # Handle swizzled layout
    exponent_uint8 = exponent_uint8.reshape(B, blk_m, blk_n)
    if is_sf_swizzled_layout:
        exponent_uint8 = swizzle_blockscale(exponent_uint8)

    if squeeze_back:
        fp_data = fp_data[0]
        exponent_uint8 = exponent_uint8[0]

    return fp_data, exponent_uint8


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=mxfp8_e4m3_quantize,
    fake_impl=mxfp8_e4m3_quantize_python,
)


class Mxfp8LinearOp:
    """
    This class executes a MXFP8 linear layer.
    """

    def __init__(
        self,
    ):
        self.preferred_backend = "triton"
        if has_flashinfer():
            self.preferred_backend = "flashinfer"

    def _apply_flashinfer(
        self,
        input: torch.Tensor,
        input_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_was_2d = False
        if input.ndim == 2:
            input_was_2d = True
            # bmm_mxfp8 requires 3D shape
            input = input.unsqueeze(0)
            input_scales = input_scales.unsqueeze(0)

        if weight.ndim == 2:
            # bmm_mxfp8 requires 3D shape
            weight = weight.unsqueeze(0)
            weight_scale = weight_scale.unsqueeze(0)

        # Use bmm-mxfp8 from flashinfer
        output = torch.ops.vllm.bmm_mxfp8(
            A=input,  # Shape: [b, m, k]
            B=weight.transpose(-2, -1),  # Shape: [b, k, n]
            A_scale=input_scales,
            B_scale=weight_scale,
            dtype=out_dtype,
            backend="cudnn",
        )

        # Remove batch dimension if it was added
        if input_was_2d:
            output = output.squeeze(0)

        return output

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Weights should be mxfp8
        assert weight.dtype == torch.float8_e4m3fn
        assert weight_scale.dtype == torch.uint8

        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"

        # From bf16 to mxfp8
        assert input.dtype == torch.bfloat16
        swizzled = True
        input_mxfp8, input_mxfp8_scales = torch.ops.vllm.mxfp8_quantize(input, swizzled)

        assert input_mxfp8.dtype == weight.dtype
        assert input_mxfp8_scales.dtype == weight_scale.dtype

        if self.preferred_backend == "flashinfer":
            return self._apply_flashinfer(
                input=input_mxfp8,
                input_scales=input_mxfp8_scales,
                weight=weight,
                weight_scale=weight_scale,
                out_dtype=out_dtype,
                bias=bias,
            )

        # TODO: use torch._scaled_mm for MXFP8
        # see this PR: https://github.com/pytorch/pytorch/pull/147548
        input_bf16 = dequant_mxfp8_to_bf16(input, input_mxfp8_scales)
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale)

        return input_bf16 @ weight_bf16.T
