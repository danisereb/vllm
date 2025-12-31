# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
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
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.quantization.utils.w8a8_utils import Mxfp8LinearOp
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


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

            return Mxfp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # TODO: Add support for MXFP8 MoE.
            logger.debug_once(
                "MXFP8 MoE layer is not implemented. "
                "Skipping quantization for this layer.",
                scope="local",
            )
        elif isinstance(layer, Attention):
            # TODO: Add support for MXFP8 Attention.
            logger.debug_once(
                "MXFP8 attention layer is not implemented. "
                "Skipping quantization for this layer.",
                scope="local",
            )
        return None


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
                dtype=torch.float8_e4m3fn,
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
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"scale_type": "weight_scale"})

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("MXFP8 weights must be in float8_e4m3fn format")

        if layer.weight_scale.dtype != torch.uint8:
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
