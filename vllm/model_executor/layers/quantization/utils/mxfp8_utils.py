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

# Check if Triton supports dot_scaled (requires Triton 3.x with Blackwell support)
try:
    # Check if tl.dot_scaled exists
    _HAS_DOT_SCALED = hasattr(tl, "dot_scaled")
except Exception:
    _HAS_DOT_SCALED = False


def supports_dot_scaled() -> bool:
    """Check if hardware supports tl.dot_scaled (Blackwell sm100+)."""
    if not _HAS_DOT_SCALED:
        return False
    # Check for Blackwell (compute capability 10.x or 11.x)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 10
    return False


# ============================================================================
# Scale Preshuffling for Tensor Core Layout (Blackwell)
# ============================================================================


def preshuffle_scales_for_tma(
    scales: torch.Tensor,
    block_m: int = 128,
    block_k: int = 128,
    vec_size: int = 32,
) -> torch.Tensor:
    """
    Preshuffle scales into tensor core layout for efficient TMA access.

    For MXFP8 on Blackwell, scales need to be in a packed block layout:
    Original: (M, K // VEC_SIZE) or (N, K // VEC_SIZE)
    Packed: (M // 128, K // VEC_SIZE // 4, 32, 4, 4) -> reshaped for TMA

    This layout allows contiguous memory access for each tensor core MMA
    in the fast inner loop.

    Args:
        scales: Input scales in shape [dim, K // 32] as uint8 (e8m0 format)
        block_m: Block size for M/N dimension (must be 128)
        block_k: Block size for K dimension (must be 128 for MXFP8)
        vec_size: Elements per scale (32 for MXFP8)

    Returns:
        Preshuffled scales ready for TMA access
    """
    dim, num_scale_k = scales.shape
    assert block_m == 128, "block_m must be 128 for tensor core layout"

    # Pad dim to multiple of 128 if needed
    dim_padded = ((dim + 127) // 128) * 128
    if dim_padded != dim:
        scales_padded = torch.zeros(
            (dim_padded, num_scale_k), dtype=scales.dtype, device=scales.device
        )
        scales_padded[:dim, :] = scales
        scales = scales_padded

    # Reshape into 5D tensor core layout:
    # (dim // 128, K // VEC_SIZE // 4, 32, 4, 4)
    # This ensures contiguous access for 128 rows at a time
    num_chunks_m = dim_padded // 128
    num_chunks_k = num_scale_k // 4 if num_scale_k >= 4 else 1

    # Pad K dimension if needed
    if num_scale_k < 4:
        scales_k_padded = torch.zeros(
            (dim_padded, 4), dtype=scales.dtype, device=scales.device
        )
        scales_k_padded[:, :num_scale_k] = scales
        scales = scales_k_padded
        num_chunks_k = 1

    # Reshape to (num_chunks_m, 128, num_chunks_k, 4)
    # then to (num_chunks_m, num_chunks_k, 32, 4, 4) for packed layout
    scales_reshaped = scales.view(num_chunks_m, 128, num_chunks_k, 4)
    # Permute to get the packed layout
    scales_reshaped = scales_reshaped.permute(0, 2, 1, 3).contiguous()
    # Reshape to (num_chunks_m, num_chunks_k, 32, 16) where 16 = 4*4
    scales_packed = scales_reshaped.view(num_chunks_m, num_chunks_k, 32, 16)

    # Final reshape for TMA: (1, num_chunks_m, num_chunks_k, 2, 256)
    # where 256 = 32 * 16 / 2
    scales_tma = scales_packed.view(1, num_chunks_m, num_chunks_k, 2, 256)

    return scales_tma


# ============================================================================
# Hardware-Accelerated Kernel using tl.dot_scaled (Blackwell)
# ============================================================================


@triton.jit
def _mxfp8_dot_scaled_kernel(
    # Pointers to inputs and output
    A,  # Input activation: [M, K] in float8_e4m3fn
    B,  # Weight: [N, K] in float8_e4m3fn
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
    BLOCK_SIZE_K: tl.constexpr,
    VEC_SIZE: tl.constexpr,  # 32 for MXFP8
):
    """
    Hardware-accelerated MXFP8 matmul using tl.dot_scaled on Blackwell.

    Uses 5th-gen Tensor Cores for block-scaled matrix multiplication.
    Computes C = (A * scale_a) @ (B * scale_b).T

    This kernel leverages hardware support for
    MXFP8 format on Blackwell GPUs.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Offsets for data
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    # Pointers for A: [M, K]
    offs_m = offs_am + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Pointers for B: [N, K], load as [K, N] for matmul
    offs_n = offs_bn + tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Scale pointers: one scale per VEC_SIZE elements
    # As: [M, K // VEC_SIZE], Bs: [N, K // VEC_SIZE]
    offs_scale_k = tl.arange(0, BLOCK_SIZE_K // VEC_SIZE)
    As_ptrs = As + offs_m[:, None] * stride_As_m + offs_scale_k[None, :] * stride_As_k
    Bs_ptrs = Bs + offs_n[:, None] * stride_Bs_n + offs_scale_k[None, :] * stride_Bs_k

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_idx * BLOCK_SIZE_K

        # Load A and B tiles
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k_start < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_mask = (offs_k[:, None] + k_start < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load scales for this K block
        scale_k_offset = k_start // VEC_SIZE
        as_mask = (offs_m[:, None] < M) & (
            offs_scale_k[None, :] + scale_k_offset < K // VEC_SIZE
        )
        bs_mask = (offs_n[:, None] < N) & (
            offs_scale_k[None, :] + scale_k_offset < K // VEC_SIZE
        )

        a_scales = tl.load(
            As_ptrs + scale_k_offset * stride_As_k, mask=as_mask, other=127
        )
        b_scales = tl.load(
            Bs_ptrs + scale_k_offset * stride_Bs_k, mask=bs_mask, other=127
        )

        # Use tl.dot_scaled for hardware-accelerated block-scaled matmul
        # Scales need to be in the right format for dot_scaled
        accumulator = tl.dot_scaled(
            a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
        )

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert to output dtype and store
    c = accumulator.to(tl.bfloat16)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def mxfp8_dot_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Hardware-accelerated MXFP8 matmul using tl.dot_scaled on Blackwell.

    This leverages 5th-gen Tensor Cores for native MXFP8 support.

    Args:
        A: [M, K] activation in float8_e4m3fn
        B: [N, K] weight in float8_e4m3fn
        As: [M, K // 32] activation scales in uint8 (e8m0 format)
        Bs: [N, K // 32] weight scales in uint8 (e8m0 format)

    Returns:
        C: [M, N] output in output_dtype (bfloat16)
    """
    assert A.dtype == MXFP8_VALUE_DTYPE
    assert B.dtype == MXFP8_VALUE_DTYPE
    assert output_dtype == torch.bfloat16

    M, K = A.shape
    N, K_b = B.shape
    assert K_b == K

    # Validate scale shapes
    assert As.shape == (M, K // MXFP8_BLOCK_SIZE)
    assert Bs.shape == (N, K // MXFP8_BLOCK_SIZE)

    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)

    # Configuration for Blackwell
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128  # Standard for MXFP8 on Blackwell
    VEC_SIZE = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    _mxfp8_dot_scaled_kernel[grid](
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        VEC_SIZE=VEC_SIZE,
        num_warps=8,
        num_stages=4,
    )

    return C


# ============================================================================
# Triton Kernels for MXFP8 Block-Scaled Matrix Multiplication
# ============================================================================


@triton.jit
def _mxfp8_dequant_gemm_kernel(
    # Pointers to inputs and output
    A,  # Input activation: [M, K] in bfloat16
    B,  # Weight: [N, K] in float8_e4m3fn (row-major, will be transposed)
    C,  # Output: [M, N] in bfloat16
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
    stride_Bs_n,
    stride_Bs_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Can be 32, 64, 128 (must be multiple of 32)
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized Triton kernel for bf16 input x MXFP8 weight matmul.

    Computes C = A @ B.T where:
    - A is [M, K] activation in bfloat16 (NO quantization needed!)
    - B is [N, K] weight in float8_e4m3fn with per-32-element row scales
    - C is [M, N] output in bfloat16

    This kernel dequantizes weights on-the-fly, avoiding input quantization.
    Supports BLOCK_SIZE_K as any multiple of 32 (32, 64, 128, etc.)
    """
    # MXFP8 scale block size (32 elements per scale)
    SCALE_BLOCK: tl.constexpr = 32
    # Number of scale groups within each K block
    NUM_SCALE_GROUPS: tl.constexpr = BLOCK_SIZE_K // SCALE_BLOCK

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

    # Pointers to A data: A is [M, K], load as [BLOCK_SIZE_M, BLOCK_SIZE_K]
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Pointers to B data: B is [N, K], but we load as [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # by swapping the indexing pattern - this avoids needing tl.trans()
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Pointers to weight scales: [N, K // 32]
    Bs_ptrs = Bs + offs_bn * stride_Bs_n

    # Accumulator in float32 for numerical accuracy
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iters):
        k_start = k * BLOCK_SIZE_K
        k_remaining = K - k_start

        # Load A tile: [BLOCK_SIZE_M, BLOCK_SIZE_K] in bf16
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)

        # Load B tile: [BLOCK_SIZE_K, BLOCK_SIZE_N] in fp8
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # Load weight scales and expand to match B shape
        # Each scale covers 32 elements along K
        # offs_k // SCALE_BLOCK gives which scale group each k index belongs to
        scale_base_idx = k * NUM_SCALE_GROUPS
        scale_indices = offs_k // SCALE_BLOCK  # [BLOCK_SIZE_K]

        # Load scales for each k position: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_s_uint8 = tl.load(
            Bs_ptrs[None, :]
            + scale_indices[:, None] * stride_Bs_k
            + scale_base_idx * stride_Bs_k
        )
        # Convert e8m0 to bf16
        b_s = (b_s_uint8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)

        # Dequantize B: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_bf16 = b_fp8.to(tl.bfloat16) * b_s

        # Compute dot product
        accumulator += tl.dot(a, b_bf16, out_dtype=tl.float32)

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
    BLOCK_SIZE_K: tl.constexpr,  # Must be 32 to match MXFP8_BLOCK_SIZE
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

    # Pointers to A data: A is [M, K], load as [BLOCK_SIZE_M, BLOCK_SIZE_K]
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Pointers to B data: B is [N, K], but we load as [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # by swapping the indexing pattern - this avoids needing tl.trans()
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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

        # Load A tile: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_start, other=0.0)
        # Load B tile: [BLOCK_SIZE_K, BLOCK_SIZE_N] - ready for dot, no transpose
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # Scale offset for this K block
        offs_ks = k_start // MXFP8_GROUP_K

        # Load scales as uint8 and convert to float32 for accumulation
        # e8m0 format: exponent only, convert via (val << 7) as bf16, then to f32
        a_s_uint8 = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s_uint8 = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        # Convert e8m0 scales to float32
        # This is: (scale.to(int16) << 7).view(bfloat16).to(float32)
        a_s = (a_s_uint8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True).to(tl.float32)
        b_s = (b_s_uint8.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True).to(tl.float32)

        # Compute scaled dot product: [M, K] @ [K, N] = [M, N]
        # No transpose needed because B is loaded as [K, N]
        accumulator += tl.dot(a, b) * (a_s[:, None] * b_s[None, :])

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
        # BLOCK_SIZE_K must be 32 to match MXFP8 block size (one scale per block)
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,  # Must match MXFP8_BLOCK_SIZE
            "GROUP_SIZE_M": 32,  # Larger group for better L2 cache reuse
            "num_warps": 4,
            "num_stages": 4,  # More stages for better pipelining
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


@functools.lru_cache
def get_mxfp8_dequant_gemm_configs(N: int, K: int) -> dict | None:
    """
    Return optimized configurations for the MXFP8 dequant GEMM kernel.
    Returns a dictionary mapping batch sizes to kernel configurations.

    Configs are loaded from JSON files generated by offline tuning.
    """
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},"
        f"dtype=mxfp8_dequant,block_shape=[1,32].json"
    )

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        import json

        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for MXFP8 dequant GEMM kernel.",
                config_file_path,
            )
            return {int(key): val for key, val in json.load(f).items()}

    return None


def mxfp8_dequant_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Optimized bf16 input x MXFP8 weight matmul with on-the-fly dequantization.

    Computes C = A @ B.T where:
    - A: [M, K] activation in bfloat16 (no quantization needed!)
    - B: [N, K] weight in float8_e4m3fn
    - Bs: [N, K // 32] weight scales in uint8 (e8m0 format)

    This avoids the expensive input quantization step.

    Returns:
        C: [M, N] output in output_dtype (bfloat16)
    """
    assert A.dtype == torch.bfloat16, f"A must be bfloat16, got {A.dtype}"
    assert B.dtype == MXFP8_VALUE_DTYPE, f"B must be {MXFP8_VALUE_DTYPE}, got {B.dtype}"
    assert output_dtype == torch.bfloat16, "Only bfloat16 output is supported"

    # A: [M, K], B: [N, K] (will compute A @ B.T)
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    N, K_b = B.shape
    assert K_b == K, f"K dimension mismatch: {K} vs {K_b}"

    # Validate scale shapes
    assert Bs.shape == (N, K // MXFP8_BLOCK_SIZE), (
        f"Bs shape mismatch: expected {(N, K // MXFP8_BLOCK_SIZE)}, got {Bs.shape}"
    )

    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    Bs = Bs.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)

    # Try to get optimized config from JSON file (offline tuning results)
    configs = get_mxfp8_dequant_gemm_configs(N, K)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        logger.debug("MXFP8 GEMM: M=%d, N=%d, K=%d using tuned config", M, N, K)
    else:
        # Select config based on matrix dimensions and batch size (M)
        # Llama-3 70B TP1: hidden=8192, intermediate=28672
        # Llama-3 70B TP2: hidden=4096, intermediate=14336 per GPU
        # BLOCK_SIZE_K can be 32, 64, 128 (must be multiple of 32)
        if M <= 16:
            # Very small batch (decode) - minimize launch overhead
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
            }
        elif M <= 64:
            # Small batch - balance occupancy and work per block
            config = {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 4,
                "num_warps": 4,
                "num_stages": 3,
            }
        elif N >= 4096 and K >= 4096:
            # Large batch + large matrices
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 4,
            }
        elif N >= 2048 or K >= 2048:
            # Large batch + medium matrices
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 4,
            }
        else:
            # Small matrices - default config
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 3,
            }
        logger.debug(
            "MXFP8 GEMM: M=%d, N=%d, K=%d using config BLOCK_M=%d, "
            "BLOCK_N=%d, BLOCK_K=%d",
            M,
            N,
            K,
            config["BLOCK_SIZE_M"],
            config["BLOCK_SIZE_N"],
            config["BLOCK_SIZE_K"],
        )

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )

    _mxfp8_dequant_gemm_kernel[grid](
        A,
        B,
        C,
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
        Bs.stride(0),
        Bs.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return C


def _mxfp8_dequant_gemm_func(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Wrapper for custom op registration."""
    return mxfp8_dequant_gemm(input, weight, weight_scale, output_dtype)


def _mxfp8_dequant_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    M = input.size(0)
    N = weight.size(0)
    return torch.empty((M, N), dtype=output_dtype, device=input.device)


direct_register_custom_op(
    op_name="mxfp8_dequant_gemm",
    op_func=_mxfp8_dequant_gemm_func,
    fake_impl=_mxfp8_dequant_gemm_fake,
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
    - "dot_scaled": Uses tl.dot_scaled for hardware-accelerated MXFP8 (Blackwell)
    - "triton": Uses custom Triton kernel with on-the-fly dequantization
    - "fallback": Dequantizes to bf16 and uses standard linear (for debugging)

    Note: torch._scaled_mm does NOT support MXFP8 block scaling format.
    """

    def __init__(self, use_fallback: bool = False, backend: str | None = None):
        """
        Initialize MXFP8 linear operation.

        Args:
            use_fallback: If True, use fallback (dequantize) mode.
                         Deprecated, use backend="fallback" instead.
            backend: One of "dot_scaled", "triton", or "fallback".
                    If None, auto-selects best backend:
                    - "dot_scaled" on Blackwell (sm100+) if supported
                    - "triton" otherwise
        """
        if backend is not None:
            assert backend in ("dot_scaled", "triton", "fallback"), (
                f"Unknown backend: {backend}. "
                "Supported: 'dot_scaled', 'triton', 'fallback'"
            )
            self.backend = backend
        elif use_fallback:
            self.backend = "fallback"
        else:
            # Auto-select best backend
            if supports_dot_scaled():
                self.backend = "dot_scaled"
                logger.info("Using dot_scaled backend for MXFP8 (Blackwell HW accel)")
            else:
                self.backend = "triton"

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
        elif self.backend == "dot_scaled":
            return self._apply_dot_scaled(input, weight, weight_scale, out_dtype, bias)
        else:  # triton
            return self._apply_triton(input, weight, weight_scale, out_dtype, bias)

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

    def _apply_dot_scaled(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Hardware-accelerated MXFP8 using tl.dot_scaled on Blackwell.

        Quantizes input to FP8 and uses tl.dot_scaled for native MXFP8 support
        on 5th-gen Tensor Cores.
        """
        assert weight.dtype == MXFP8_VALUE_DTYPE
        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"
        assert input.dtype == torch.bfloat16

        # Get weight dimensions
        out_features, in_features = weight.shape
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Flatten input for the GEMM
        input_flat = input.view(-1, input.shape[-1])
        M = input_flat.shape[0]

        # Quantize input to MXFP8
        # Note: not using swizzled layout for dot_scaled path
        input_fp8, input_scales = torch.ops.vllm.mxfp8_quantize(input_flat, False)

        # Convert weight scales from float8_e8m0fnu to uint8
        weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)

        # Handle padded weight scales
        out_features_padded = (out_features + 127) // 128 * 128
        weight_scale_2d = weight_scale_uint8.view(out_features_padded, scale_k)[
            :out_features, :
        ].contiguous()

        # Ensure input scales are 2D: [M, K // 32]
        if input_scales.ndim == 1:
            input_scales = input_scales.view(M, -1)

        # Call the hardware-accelerated dot_scaled kernel
        output = mxfp8_dot_scaled_mm(
            input_fp8,
            weight,
            input_scales,
            weight_scale_2d,
            out_dtype,
        )

        # Reshape output to match input batch dimensions
        output = output.view(*input.shape[:-1], out_features)

        if bias is not None:
            output = output + bias

        return output

    def _apply_triton(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Optimized Triton kernel for MXFP8 linear with on-the-fly weight dequant.

        Uses bf16 input directly (no quantization!) and dequantizes FP8 weights
        on-the-fly within the kernel. This is much faster than quantizing input.
        """
        assert weight.dtype == MXFP8_VALUE_DTYPE
        assert out_dtype == torch.bfloat16, "Only bfloat16 is supported as out_dtype"
        assert input.dtype == torch.bfloat16

        # Get weight dimensions
        out_features, in_features = weight.shape
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Flatten input for the GEMM
        input_flat = input.view(-1, input.shape[-1])

        # Convert weight scales from float8_e8m0fnu to uint8 for Triton kernel
        weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)

        # Handle padded weight scales
        out_features_padded = (out_features + 127) // 128 * 128
        weight_scale_2d = weight_scale_uint8.view(out_features_padded, scale_k)[
            :out_features, :
        ].contiguous()

        # Call the optimized dequant GEMM kernel (no input quantization!)
        output = torch.ops.vllm.mxfp8_dequant_gemm(
            input_flat,
            weight,
            weight_scale_2d,
            out_dtype,
        )

        # Reshape output to match input batch dimensions
        output = output.view(*input.shape[:-1], out_features)

        if bias is not None:
            output = output + bias

        return output
