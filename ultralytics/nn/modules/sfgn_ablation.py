from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from .sfgn import DepthwiseSeparableConv, SFG_AIFI
except ImportError:  # Allows: python ultralytics/nn/modules/sfgn_ablation.py
    from sfgn import DepthwiseSeparableConv, SFG_AIFI


class FeatureGatedNetwork(nn.Module):
    """Feature-gating branch without the spatial feedback map gamma.

    The feature after self-attention is projected into two branches
    ``H_after^(1)`` and ``H_after^(2)``. The first branch generates the gate,
    and the second branch is modulated by that gate:

        G = sigmoid(Conv_1x1(Conv_dw_3x3(H_after^(1))))
        H_out = G * H_after^(2)

    ``h_before`` is accepted to keep the same interface as the complete SFGN,
    but is intentionally unused because spatial feedback is removed.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = int(channels)
        self.feature_projection = DepthwiseSeparableConv(
            channels,
            2 * channels,
        )
        self.gate_depthwise = DepthwiseSeparableConv(channels, channels)
        self.gate_projection = nn.Conv2d(channels, channels, 1, bias=True)

    def forward_with_gate(
        self,
        h_before: Tensor,
        h_after: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if h_before.shape != h_after.shape:
            raise ValueError(
                "h_before and h_after must have identical shapes, got "
                f"{tuple(h_before.shape)} and {tuple(h_after.shape)}"
            )

        h1, h2 = self.feature_projection(h_after).chunk(2, dim=1)
        gate = torch.sigmoid(
            self.gate_projection(self.gate_depthwise(h1))
        )
        return gate * h2, gate

    def forward(self, h_before: Tensor, h_after: Tensor) -> Tensor:
        output, _ = self.forward_with_gate(h_before, h_after)
        return output
class FG_AIFI(SFG_AIFI):
    """AIFI whose FFN is replaced by feature gating without feedback.

    The self-attention, positional encoding, residual connections, and layer
    normalization are inherited unchanged from ``SFG_AIFI``. Only its complete
    SFGN block is replaced with ``FeatureGatedNetwork``.

    The constructor intentionally matches ``SFG_AIFI`` and Ultralytics AIFI so
    the YAML entry ``[-1, 1, FG_AIFI, [1024, 8]]`` is parsed consistently.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        normalize_before: bool = False,
    ) -> None:
        super().__init__(
            c1=c1,
            cm=cm,
            num_heads=num_heads,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.sfgn = FeatureGatedNetwork(c1)


if __name__ == "__main__":
    module = FG_AIFI(c1=256, cm=1024, num_heads=8)
    sample = torch.randn(2, 256, 20, 20, requires_grad=True)
    output = module(sample)
    output.mean().backward()
    print("input:", tuple(sample.shape))
    print("output:", tuple(output.shape))