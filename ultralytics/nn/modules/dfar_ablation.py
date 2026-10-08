from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

try:
    from .dfae import ConvBNAct, DepthwiseSeparableConv, FrequencyEnhancementBranch
except ImportError:  # Allows: python ultralytics/nn/modules/dfae_ablation.py
    from dfae import ConvBNAct, DepthwiseSeparableConv, FrequencyEnhancementBranch


class DFAE_FEB(nn.Module):
    """FEB-only DFAE variant for the internal ablation experiment.

    This module retains the same local-texture extraction and channel
    integration stem as the complete DFAE, but removes MCB. The FEB output is
    projected by a 1x1 convolution so that its output dimensions remain
    identical to those of the complete DFAE.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output channels. If omitted, it equals
            ``in_channels``.
        alpha_init: Initial value of the learnable frequency fusion weight.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # Same input stem as the complete DFAE:
        # F_c = Conv_1x1(F_in + Conv_dw_3x3(F_in)).
        self.local_texture = DepthwiseSeparableConv(
            in_channels,
            in_channels,
            kernel_size=3,
        )
        self.channel_integration = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )

        # Retain FEB and remove MCB.
        self.feb = FrequencyEnhancementBranch(
            channels=out_channels,
            alpha_init=alpha_init,
        )

        # Keep the same output channel dimension as the complete DFAE.
        self.output_projection = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=1,
        )

    @property
    def alpha(self) -> Tensor:
        """Expose alpha for logging and sensitivity analysis."""
        return self.feb.alpha

    def forward(
            self,
            x: Tensor,
            return_intermediates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if x.ndim != 4:
            raise ValueError(f"DFAE_FEB expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        fused = self.channel_integration(x + self.local_texture(x))
        frequency_feature = self.feb(fused)
        output = self.output_projection(frequency_feature)

        if not return_intermediates:
            return output

        intermediates = {
            "fused": fused,
            "frequency": frequency_feature,
        }
        return output, intermediates
if __name__ == "__main__":
    module = DFAE_FEB(64, 64, alpha_init=0.1)
    sample = torch.randn(2, 64, 81, 79, requires_grad=True)
    prediction, features = module(sample, return_intermediates=True)
    prediction.mean().backward()
    print("input:", tuple(sample.shape))
    print("output:", tuple(prediction.shape))
    print("alpha:", module.alpha.item())
    print("intermediates:", {k: tuple(v.shape) for k, v in features.items()})