# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    is_layer_skipped,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Mxfp8LinearOp,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class Mxfp8Config(QuantizationConfig):
    """Configuration for MXFP8 quantization.

    MXFP8 (Microscaling FP8) uses FlashInfer for efficient quantized matrix
    multiplication on SM100 GPUs. Supports dynamic activation quantization
    with per-tensor weight quantization.

    Requirements:
    - CUDA with compute capability 10.0 (SM100)
    - FlashInfer installed
    - BF16 model weights
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8_linear"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """MXFP8 supports BF16 activations."""
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """MXFP8 requires SM100 (compute capability 10.0)."""
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        """No config file needed - uses model weights directly."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        """Create config from model's quantization config."""
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(ignored_layers=ignored_layers)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """Apply vLLM mapper to ignored layers."""
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Get quantization method for a layer.

        MXFP8 quantization is applied only to attention linear layers.
        MoE layers (including shared experts and latent projections) are skipped.
        """
        # Verify platform support
        if not (
            current_platform.is_cuda()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            raise ValueError(
                "MXFP8 requires CUDA with compute capability 10.0 (SM100) "
                "and FlashInfer installed. Please check your environment."
            )

        # Skip MoE-related layers - check prefix for MoE indicators
        moe_indicators = [
            ".mixer.shared_experts",  # NemotronHMLP
            ".mixer.fc1_latent_proj",  # Latent MoE
            ".mixer.fc2_latent_proj",  # Latent MoE
            ".mixer.gate",  # ReplicatedLinear
            ".experts",  # FusedMoE experts (SharedFusedMoE)
        ]
        is_moe_layer = any(indicator in prefix for indicator in moe_indicators)

        if isinstance(layer, LinearBase):
            # Skip MoE-related linear layers
            if is_moe_layer:
                return UnquantizedLinearMethod()

            # Check if layer is explicitly ignored
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()

            logger.debug("Enabling MXFP8 quantization for: %s", prefix)
            # Apply MXFP8 quantization to attention linear layers
            return Mxfp8LinearMethod(self)
        elif isinstance(layer, Attention):
            # KV cache quantization not yet supported for MXFP8
            return None
        elif isinstance(layer, FusedMoE):
            # MoE quantization not yet supported for MXFP8
            return None
        return None


class Mxfp8LinearMethod(LinearMethodBase):
    """Linear method for MXFP8 quantization.

    Supports loading BF16 model checkpoints and quantizing weights/activations
    to MXFP8 format using FlashInfer.

    MXFP8 uses:
    - Dynamic per-token activation quantization
    - Per-tensor weight quantization
    - Block size of 32 for MXFP8 format
    """

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

        # Verify requirements
        if not (
            current_platform.is_cuda()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            raise ValueError(
                "MXFP8 requires CUDA with compute capability 10.0 (SM100) "
                "and FlashInfer installed"
            )

        # Initialize the MXFP8 linear operator
        # MXFP8 uses dynamic per-token activation quantization
        self.mxfp8_linear = Mxfp8LinearOp(
            act_quant_static=False,  # Dynamic activation quantization
            act_quant_group_shape=GroupShape.PER_TENSOR,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for MXFP8 quantization."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # For MXFP8, weights are stored in BF16 and quantized at runtime
        # Weight scale is per-tensor
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,  # Keep as BF16
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale (per-tensor)
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(1, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        weight_scale[:] = 1.0  # Initialize to 1.0
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Process weights after loading from checkpoint.

        For MXFP8, we quantize weights to MXFP8 format and compute weight scales.
        """
        from flashinfer.fp8_quantization import mxfp8_quantize

        # Quantize weights to MXFP8
        # MXFP8 uses block size of 32
        weight_2d = layer.weight.data.view(-1, layer.weight.shape[-1])
        qweight, weight_scale = mxfp8_quantize(
            input=weight_2d,
            is_sf_swizzled_layout=True,
        )

        # Store quantized weight and scale
        # Weight needs to be transposed for the GEMM operation
        layer.weight = Parameter(qweight.t(), requires_grad=False)

        # Extract per-tensor scale (take first element if multiple scales)
        if weight_scale.numel() > 1:
            # If we get multiple scales, take the max for per-tensor quantization
            weight_scale_value = weight_scale.max()
        else:
            weight_scale_value = (
                weight_scale.view(-1)[0] if weight_scale.numel() == 1 else weight_scale
            )

        layer.weight_scale = Parameter(
            weight_scale_value.unsqueeze(0)
            if weight_scale_value.dim() == 0
            else weight_scale_value,
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply MXFP8 linear layer."""
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=x.dtype,  # Output in same dtype as input (BF16)
            bias=bias,
        )
