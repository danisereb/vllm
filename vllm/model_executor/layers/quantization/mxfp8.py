# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from math import ceil
from typing import Any, Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.layer import Attention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp8_fake_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    register_moe_scaling_factors,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    Mxfp8LinearOp,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8


class Mxfp8MoeBackend(Enum):
    NONE = 0
    FLASHINFER_TRTLLM = 1


# TODO: use function from here?
# vllm.model_executor.layers.quantization.utils.flashinfer_utils
def rotate_flashinfer_fp8_moe_weights(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor, gemm1_scales, gemm2_scales
):
    # TODO: Hack for mxfp8
    from flashinfer import (
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )

    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    # hidden_size = gemm1_weights.shape[-1]
    # intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    # gemm1_weights_fp8_interleaved = []
    # for i in range(num_experts):
    #     gemm1_weights_fp8_interleaved.append(
    #         reorder_rows_for_gated_act_gemm(gemm1_weights[i])
    #     )

    # Stack weights and scales for all experts
    # gemm1_weights_fp8_interleaved = \
    # torch.stack(gemm1_weights_fp8_interleaved).reshape(
    #     num_experts, 2 * intermediate_size, hidden_size
    # )
    gemm1_weights_fp8_interleaved = gemm1_weights

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_scales_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm1_weights_fp8_interleaved[i], epilogue_tile_m)
            # gemm1_weights_fp8_interleaved[i]
        )

        gemm1_scales_shuffled.append(
            shuffle_matrix_sf_a(gemm1_scales[i], 128).contiguous()
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
            # gemm2_weights[i]
        )
        gemm2_scales_shuffled.append(
            shuffle_matrix_sf_a(gemm2_scales[i], 128).contiguous()
        )

    # Stack weights for all experts
    return (
        torch.stack(gemm1_weights_fp8_shuffled),
        torch.stack(gemm2_weights_fp8_shuffled).view(torch.float8_e4m3fn),
        torch.stack(gemm1_scales_shuffled),
        torch.stack(gemm2_scales_shuffled),
    )
    # gemm1_weights.data =
    # gemm2_weights.data = \
    # torch.stack(gemm2_weights_fp8_shuffled).view(torch.float8_e4m3fn)

    # gemm1_scales.data = torch.stack(gemm1_scales_shuffled)
    # gemm2_scales.data = torch.stack(gemm2_scales_shuffled)


def pad_to(t: torch.Tensor, dim: int, pad_to: int) -> torch.Tensor:
    prev_shape = list(t.shape)
    prev_shape[dim] = pad_to
    padded_t = t.new_zeros(*prev_shape)
    if dim == 2:
        padded_t[:, :, : t.shape[dim]] = t
    elif dim == 1:
        padded_t[:, : t.shape[dim], :] = t
    else:
        raise ValueError()
    return padded_t


class Mxfp8Config(QuantizationConfig):
    """
    Example config:
    "quantization_config": {
        "quant_method": "mxfp8",
        "ignored_layers": []
    },
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

        # MXFP8 block size is 32
        self.weight_block_size = [1, 32]

        logger.info_once("Using MXFP8 quantization")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_serialized = "mxfp8" in quant_method
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)

        assert is_serialized, "MXFP8 is only supported in serialized format"
        return cls(ignored_layers=ignored_layers)

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()

            logger.debug("Using MXFP8 for layer %s", prefix)
            return Mxfp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp8MoEMethod(self, layer)
        elif isinstance(layer, Attention):
            # TODO: Add support for MXFP8 Attention
            logger.debug_once(
                "MXFP8 attention layer is not implemented. "
                "Skipping quantization for this layer.",
                scope="local",
            )

        # TODO: return None?
        return UnquantizedLinearMethod


class Mxfp8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Mxfp8Config) -> None:
        self.quant_config = quant_config

        # TODO: check
        self.out_dtype = torch.get_default_dtype()

        assert current_platform.is_cuda(), "MXFP8 is only supported on CUDA"

        self.mxfp8_linear = Mxfp8LinearOp()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Store metadata on the layer for later use
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Create weight parameter in F8_E4M3 format
        # Shape: [output_size_per_partition, input_size_per_partition]
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # MXFP8 uses block size of 32
        mxfp8_block_size = 32

        # Create weight scale parameter in U8 format (E8M0 - power-of-2 exponents)
        # MXFP8 has one scale per block of 32 elements along the K dimension
        # Shape: [output_size_per_partition, ceil(input_size_per_partition / 32)]
        num_scale_elements = (
            input_size_per_partition + mxfp8_block_size - 1
        ) // mxfp8_block_size
        weight_scale = BlockQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                num_scale_elements,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"scale_type": "weight_scale"})

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError("MXFP8 weights must be in float8_e4m3fn format")

        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError("MXFP8 weight_scale must be in uint8 format (E8M0)")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            bias=bias,
        )


def validate_moe_block_sizes(
    intermediate_size_per_partition: int,
    hidden_size: int,
    weight_block_size: list[int],
    tp_size: int = 1,
) -> None:
    """
    Validate that MoE weight dimensions are compatible with block quantization.

    This function validates alignment requirements for block-wise quantized
    MoE weights with tensor parallelism.

    Args:
        intermediate_size_per_partition: The intermediate size per partition
            (output size for gate/up, input size for down).
        hidden_size: The hidden size of the model.
        weight_block_size: Block size as [block_n, block_k].
        tp_size: Tensor parallel size (default 1).

    Raises:
        ValueError: If dimensions are not divisible by block sizes.
    """
    block_n, block_k = weight_block_size[0], weight_block_size[1]

    # NOTE: To ensure proper alignment of the block-wise quantization
    # scales, the output_size of the weights for both the gate and up
    # layers must be divisible by block_n.
    # Required by column parallel or enabling merged weights
    if intermediate_size_per_partition % block_n != 0:
        raise ValueError(
            f"The output_size of gate's and up's weight = "
            f"{intermediate_size_per_partition} is not divisible by "
            f"weight quantization block_n = {block_n}."
        )

    # Required by row parallel (down projection)
    if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
        raise ValueError(
            f"The input_size of down's weight = "
            f"{intermediate_size_per_partition} is not divisible by "
            f"weight quantization block_k = {block_k}."
        )


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


class Mxfp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Mxfp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.layer = layer
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None

        self.mxfp8_backend = Mxfp8MoeBackend.NONE

        self.flashinfer_moe_backend: FlashinferMoeBackend | None = None

        if self.mxfp8_backend == Mxfp8MoeBackend.FLASHINFER_TRTLLM:
            if not self.moe.is_act_and_mul:
                logger.info_once(
                    "Non-gated MoE is not supported for TENSORRT_LLM backend"
                )
            else:
                self.flashinfer_moe_backend = FlashinferMoeBackend.TENSORRT_LLM

        logger.info_once(f"Using Flashinfer MoE backend {self.flashinfer_moe_backend}")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype  # TODO: do we need this?
        layer.weight_block_size = None
        quant_method = None

        # TODO: add assert for MXFP8 weights?
        # Assume MXFP8 weights are FP8 E4M3
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()

            # Validate block sizes for MoE weights
            validate_moe_block_sizes(
                intermediate_size_per_partition=intermediate_size_per_partition,
                hidden_size=hidden_size,
                weight_block_size=self.weight_block_size,
                tp_size=tp_size,
            )
        numu_shards = 2 if self.moe.is_act_and_mul else 1

        intermediate_size_per_partition = (
            self._maybe_increase_intermediate_size_for_mxfp8(
                intermediate_size_per_partition
            )
        )
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                numu_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # MXFP8 weight scales
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                numu_shards * intermediate_size_per_partition,
                hidden_size // 32,  # TODO: replace 32 with variable
                dtype=MXFP8_SCALE_DTYPE,
            ),
            requires_grad=False,
        )
        # TODO: Change get_tensor_model_parallel_world_size after TP is fixed
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                # get_tensor_model_parallel_world_size()
                # * intermediate_size_per_partition
                intermediate_size_per_partition // 32,  # TODO: replace 32 with variable
                dtype=MXFP8_SCALE_DTYPE,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # TODO: check if this is what we want for MXFP8
        quant_method = FusedMoeWeightScaleSupported.BLOCK.value

        assert quant_method is not None, "quant_method should be set"
        extra_weight_attrs.update({"quant_method": quant_method})

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        set_weight_attrs(w13_weight, {"quant_method": quant_method})
        set_weight_attrs(w2_weight, {"quant_method": quant_method})

        # TODO: check (input scales in MXFP8 ?)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _maybe_increase_intermediate_size_for_mxfp8(
        self, intermediate_size_per_partition: int
    ) -> int:
        # For MXFP8, we need to pad the weight
        # tensors to be deivisible by tp_size * 32
        divisible_by = 32
        if intermediate_size_per_partition % divisible_by != 0:
            increase = divisible_by - intermediate_size_per_partition % divisible_by
            logger.debug_once(
                f"Padding intermediate_size_per_partition from "
                f"{intermediate_size_per_partition} to "
                f"{intermediate_size_per_partition + increase} for MXFP8."
            )
            intermediate_size_per_partition += increase
        return intermediate_size_per_partition

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # -------------------------
        # w13 processing
        # -------------------------
        w13_q, w13_scale = layer.w13_weight, layer.w13_weight_scale

        dq_w13 = dequant_mxfp8_to_bf16(w13_q, w13_scale).contiguous()
        layer.w13_weight = torch.nn.Parameter(dq_w13.data, requires_grad=False)

        # -------------------------
        # w2 processing
        # -------------------------
        w2_q, w2_scale_full = layer.w2_weight, layer.w2_weight_scale

        # Select expert block
        blk = ceil(w2_q.shape[-1] / 32)
        start = layer.ep_rank * blk
        end = (layer.ep_rank + 1) * blk
        w2_scale = w2_scale_full[..., start:end]

        dq_w2 = dequant_mxfp8_to_bf16(w2_q, w2_scale).contiguous()
        layer.w2_weight = torch.nn.Parameter(dq_w2.data, requires_grad=False)

    def _process_weights_after_loading_flashinfer(self, layer: torch.nn.Module) -> None:
        # Pad weights
        layer.intermediate_size_per_partition = round_up(
            layer.intermediate_size_per_partition, 128
        )
        layer.w13_weight = torch.nn.Parameter(
            pad_to(
                layer.w13_weight, 1, layer.intermediate_size_per_partition
            ).contiguous(),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            pad_to(
                layer.w13_weight_scale,
                1,
                layer.intermediate_size_per_partition,
            )
            .to(dtype=MXFP8_SCALE_DTYPE)
            .contiguous(),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            pad_to(
                layer.w2_weight, 2, layer.intermediate_size_per_partition
            ).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            pad_to(
                layer.w2_weight_scale,
                2,
                layer.intermediate_size_per_partition // 32,
            )
            .to(dtype=MXFP8_SCALE_DTYPE)
            .contiguous(),
            requires_grad=False,
        )

        gemm1_w, gemm2_w, gemm1_s, gemm2_s = rotate_flashinfer_fp8_moe_weights(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale.to(dtype=MXFP8_SCALE_DTYPE),
            layer.w2_weight_scale.to(dtype=MXFP8_SCALE_DTYPE),
        )
        layer.w13_weight_shuffled = torch.nn.Parameter(gemm1_w, requires_grad=False)
        layer.w2_weight_shuffled = torch.nn.Parameter(gemm2_w, requires_grad=False)
        layer.w13_scales_shuffled = torch.nn.Parameter(gemm1_s, requires_grad=False)
        layer.w2_scales_shuffled = torch.nn.Parameter(gemm2_s, requires_grad=False)

        # TODO: needed?
        # del layer.w13_weight
        # del layer.w2_weight
        # del layer.w13_weight_scale
        # del layer.w2_weight_scale

        # Dequant
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.local_num_experts):
            start = 0
            for shard_id in range(2):
                dq_weight = per_tensor_dequantize(
                    layer.w13_weight[expert_id][start : start + shard_size, :],
                    layer.w13_weight_scale[expert_id][shard_id],
                )
                layer.w13_weight[expert_id][start : start + shard_size, :], _ = (
                    ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                )
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        if self.flashinfer_moe_backend is not None:
            # NOTE: weights have to be swapped since the activation is
            # applied on different half for flashinfer vs vllm
            assert not self.block_quant
            register_moe_scaling_factors(layer)
            w13_weight = swap_w13_to_w31(layer.w13_weight.data)
            # TODO: hack to avoid ruff
            # if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            #     rotate_flashinfer_fp8_moe_weights(w13_weight, w2_weight)
            layer.w13_weight.data = w13_weight.data

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.flashinfer_moe_backend is not None:
            assert self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
            self._process_weights_after_loading_flashinfer(layer)
        else:
            self._process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            return mxfp8_fake_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_shuffled,
                w2_scale=layer.w2_weight_shuffled,
            )

        return mxfp8_fake_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    @property
    def supports_eplb(self) -> bool:
        # TODO: check
        return True

    @property
    def allow_inplace(self) -> bool:
        # TODO: check
        return True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # After process_weights_after_loading, weights are dequantized to bf16.
        # Use unquantized fused_experts kernel for inference.
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        # Expert selection
        topk_weights, topk_ids = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
