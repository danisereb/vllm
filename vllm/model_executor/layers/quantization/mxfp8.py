# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

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
    def from_config(cls, config):
        return cls()

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

            # TODO: Add support for MXFP8 Linear Method.
            logger.debug_once(
                "MXFP8 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
                scope="local",
            )
            return UnquantizedLinearMethod()
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
