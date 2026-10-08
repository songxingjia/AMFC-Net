"""Spatial Feedback Gated Network (SFGN) and SFG-AIFI.

Place this file at ``ultralytics/nn/modules/sfgn.py``.

SFG_AIFI preserves the positional encoding and multi-head self-attention of
Ultralytics AIFI, while replacing its channel-only FFN with SFGN.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Convolution followed by batch normalization and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: Tuple[int, int] | int,
        padding: Tuple[int, int] | int = 0,
        dilation: Tuple[int, int] | int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        layers = [
            nn.Conv2d(
                c1,
                c2,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by a pointwise projection."""

    def __init__(self, c1: int, c2: int, activate: bool = True) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(c1, c1, 3, padding=1, groups=c1)
        self.pointwise = ConvBNAct(c1, c2, 1, activate=activate)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SFGN(nn.Module):
    """Spatial Feedback Gated Network.

    Args:
        channels: Feature channel dimension.
        pool_ratio: Spatial reduction ratio used to build the feedback map.

    Inputs:
        h_before: Feature before self-attention, shape ``[B, C, H, W]``.
        h_after: Feature after self-attention, shape ``[B, C, H, W]``.
    """

    def __init__(self, channels: int, pool_ratio: int = 2) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if pool_ratio <= 0:
            raise ValueError("pool_ratio must be positive")

        self.channels = channels
        self.pool_ratio = pool_ratio

        # gamma = Up(Conv_3x1(Conv_1x3(Conv_r=2(AvgPool(H_before)))))
        self.context_dilated = ConvBNAct(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
        )
        self.context_horizontal = ConvBNAct(
            channels,
            channels,
            (1, 3),
            padding=(0, 1),
        )
        self.context_vertical = ConvBNAct(
            channels,
            channels,
            (3, 1),
            padding=(1, 0),
            activate=False,
        )

        # Obtain H_after^(1) and H_after^(2) from a lightweight spatial
        # projection, then use the first branch to generate the gate.
        self.feature_projection = DepthwiseSeparableConv(channels, 2 * channels)
        self.gate_depthwise = DepthwiseSeparableConv(channels, channels)
        self.gate_projection = nn.Conv2d(channels, channels, 1, bias=True)

    def _spatial_feedback(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pooled_size = (
            max(1, math.ceil(height / self.pool_ratio)),
            max(1, math.ceil(width / self.pool_ratio)),
        )
        gamma = F.adaptive_avg_pool2d(x, pooled_size)
        gamma = self.context_dilated(gamma)
        gamma = self.context_horizontal(gamma)
        gamma = self.context_vertical(gamma)
        return F.interpolate(
            gamma,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def forward_with_gate(self, h_before: Tensor, h_after: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the SFGN output and its spatial modulation map."""
        if h_before.shape != h_after.shape:
            raise ValueError(
                "h_before and h_after must have identical shapes, got "
                f"{tuple(h_before.shape)} and {tuple(h_after.shape)}"
            )

        gamma = self._spatial_feedback(h_before)
        h1, h2 = self.feature_projection(h_after).chunk(2, dim=1)
        gate = torch.sigmoid(self.gate_projection(self.gate_depthwise(h1) + gamma))
        return gate * h2, gate

    def forward(self, h_before: Tensor, h_after: Tensor) -> Tensor:
        output, _ = self.forward_with_gate(h_before, h_after)
        return output
class SFG_AIFI(nn.Module):
    """AIFI with its conventional FFN replaced by SFGN.

    The constructor follows Ultralytics AIFI so that the YAML entry
    ``[-1, 1, SFG_AIFI, [1024, 8]]`` can be parsed as
    ``SFG_AIFI(c1, 1024, 8)``. ``cm`` is retained for configuration
    compatibility; the channel-only FFN itself is not instantiated.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        if c1 % num_heads != 0:
            raise ValueError(f"c1={c1} must be divisible by num_heads={num_heads}")
        if c1 % 4 != 0:
            raise ValueError("c1 must be divisible by 4 for 2-D sine-cosine encoding")
        if normalize_before:
            raise NotImplementedError(
                "SFG_AIFI currently follows the post-normalization AIFI used by RT-DETR"
            )

        self.c1 = c1
        self.cm = cm  # Retained for YAML/API compatibility and experiment logging.
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=c1,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.sfgn = SFGN(c1)

    @staticmethod
    def build_2d_sincos_position_embedding(
        width: int,
        height: int,
        embed_dim: int,
        like: Tensor,
        temperature: float = 10000.0,
    ) -> Tensor:
        """Build row-major 2-D sine-cosine positional embeddings."""
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")

        dtype = torch.float32
        grid_h = torch.arange(height, device=like.device, dtype=dtype)
        grid_w = torch.arange(width, device=like.device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=like.device, dtype=dtype) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_h = grid_h.flatten()[:, None] * omega[None, :]
        out_w = grid_w.flatten()[:, None] * omega[None, :]
        position = torch.cat(
            (out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()),
            dim=1,
        )
        return position.unsqueeze(0).to(dtype=like.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"SFG_AIFI expects BCHW input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.c1:
            raise ValueError(f"Expected {self.c1} channels, got {channels}")

        h_before = x
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        position = self.build_2d_sincos_position_embedding(
            width,
            height,
            channels,
            like=x,
        )

        query = key = tokens + position
        attended = self.attention(
            query,
            key,
            value=tokens,
            need_weights=False,
        )[0]
        h_after_tokens = self.norm1(tokens + self.dropout1(attended))
        h_after = h_after_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )

        gated = self.sfgn(h_before, h_after)
        gated_tokens = gated.flatten(2).transpose(1, 2)
        output_tokens = self.norm2(h_after_tokens + self.dropout2(gated_tokens))
        return output_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        ).contiguous()


if __name__ == "__main__":
    module = SFG_AIFI(c1=256, cm=1024, num_heads=8)
    sample = torch.randn(2, 256, 20, 20, requires_grad=True)
    output = module(sample)
    output.mean().backward()
    print("input:", tuple(sample.shape))
    print("output:", tuple(output.shape))
