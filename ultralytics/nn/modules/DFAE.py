"""Dual-domain Feature Adaptive Enhancement (DFAE).

This implementation follows the DFAE description in the AMFC-Net-dfae_feb.yaml manuscript:

1. Residual local-texture extraction and 1x1 channel integration.
2. A frequency enhancement branch (FEB) based on a fixed Haar DWT.
3. A multi-scale context branch (MCB) with dilation rates 1, 2, and 3.
4. Concatenation of FEB and MCB features followed by 1x1 fusion.

The implementation has no dependency on an external wavelet package.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Convolution followed by batch normalization and SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise convolution followed by pointwise channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class HaarDWT2D(nn.Module):
    """Single-level orthonormal 2-D Haar discrete wavelet transform.

    For odd spatial dimensions, the right or bottom boundary is replicated
    once before decomposition. The returned subbands have spatial size
    ``ceil(H / 2) x ceil(W / 2)``.
    """

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"HaarDWT2D expects BCHW input, got {tuple(x.shape)}")

        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        # The factor 1/2 gives the orthonormal 2-D Haar transform.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class FrequencyEnhancementBranch(nn.Module):
    """Frequency-domain enhancement branch (FEB)."""

    def __init__(self, channels: int, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.dwt = HaarDWT2D()
        self.high_frequency_conv = DepthwiseSeparableConv(
            in_channels=3 * channels,
            out_channels=channels,
            kernel_size=3,
        )

        # Directly learn alpha to match the manuscript and its initialization
        # sensitivity experiment. It is intentionally not passed through a
        # sigmoid, so alpha_init=0.0 and alpha_init=1.0 retain their meanings.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        ll, lh, hl, hh = self.dwt(x)

        low = F.interpolate(
            ll,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        low_enhanced = low * x

        high = torch.cat((lh, hl, hh), dim=1)
        high = self.high_frequency_conv(high)

        # A standard DWT halves the spatial resolution. This upsampling is
        # required before adding the high-frequency and low-frequency terms.
        high = F.interpolate(
            high,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return self.alpha * low_enhanced + (1.0 - self.alpha) * high


class MultiScaleContextBranch(nn.Module):
    """Spatial multi-scale context branch (MCB)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels < 3:
            raise ValueError("MCB out_channels must be at least 3")

        # Distribute output channels across the three branches while ensuring
        # that their concatenation contains exactly out_channels channels.
        base, remainder = divmod(out_channels, 3)
        branch_channels = [base + int(i < remainder) for i in range(3)]

        self.branches = nn.ModuleList(
            [
                ConvBNAct(
                    in_channels,
                    branch_channels[i],
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
                for i, dilation in enumerate((1, 2, 3))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class DFAE(nn.Module):
    """Dual-domain Feature Adaptive Enhancement module.

    Args:
        in_channels: Number of channels in the input feature map.
        out_channels: Number of output channels. If omitted, it is equal to
            ``in_channels``.
        alpha_init: Initial value of the learnable FEB fusion weight alpha.

    Input:
        Tensor with shape ``[B, in_channels, H, W]``.

    Output:
        Tensor with shape ``[B, out_channels, H, W]``.
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

        self.in_channels = in_channels
        self.out_channels = out_channels

        # F_c = Conv_1x1(F_in + Conv_dw_3x3(F_in))
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

        self.feb = FrequencyEnhancementBranch(
            channels=out_channels,
            alpha_init=alpha_init,
        )
        self.mcb = MultiScaleContextBranch(
            in_channels=out_channels,
            out_channels=out_channels,
        )

        # F_out = Conv_1x1(Concat(F_MCB, F_FEB))
        self.output_fusion = ConvBNAct(
            2 * out_channels,
            out_channels,
            kernel_size=1,
        )
@property
def alpha(self) -> Tensor:
    """Expose the learnable alpha parameter for logging."""
    return self.feb.alpha

def forward(
    self,
    x: Tensor,
    return_intermediates: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    if x.ndim != 4:
        raise ValueError(f"DFAE expects BCHW input, got {tuple(x.shape)}")
    if x.shape[1] != self.in_channels:
        raise ValueError(
            f"Expected {self.in_channels} input channels, got {x.shape[1]}"
        )

    fused = self.channel_integration(x + self.local_texture(x))
    frequency_feature = self.feb(fused)
    spatial_feature = self.mcb(fused)
    output = self.output_fusion(
        torch.cat((spatial_feature, frequency_feature), dim=1)
    )

    if not return_intermediates:
        return output

    intermediates = {
        "fused": fused,
        "frequency": frequency_feature,
        "spatial": spatial_feature,
    }
    return output, intermediates


if __name__ == "__main__":
    # Shape and gradient smoke test, including odd spatial dimensions.
    module = DFAE(in_channels=64, out_channels=64, alpha_init=0.1)
    sample = torch.randn(2, 64, 81, 79, requires_grad=True)
    prediction, features = module(sample, return_intermediates=True)
    prediction.mean().backward()

    print("input:", tuple(sample.shape))
    print("output:", tuple(prediction.shape))
    print("alpha:", module.alpha.item())
    print("intermediates:", {k: tuple(v.shape) for k, v in features.items()})