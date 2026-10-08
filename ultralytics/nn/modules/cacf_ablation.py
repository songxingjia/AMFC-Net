from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .cacf import ConvBNAct
except ImportError:  # Allows: python ultralytics/nn/modules/cacf_ablation.py
    from cacf import ConvBNAct


class CACF_Common(nn.Module):
    """CACF variant that generates weights from common information only.

    After channel and spatial alignment, this variant uses

        F_common = F_s + F_h
        A = sigmoid(Conv_3x3(F_common))
        F_out = A * F_s + (1 - A) * F_h

    The difference cue ``F_s - F_h`` is completely removed.

    Args:
        in_channels: Input channel dimensions ``[C_s, C_h]`` inserted by the
            custom ``parse_model`` branch in ``tasks.py``.
        out_channels: Unified output channel dimension.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError(
                "CACF_Common expects two input channel values, got "
                f"{list(in_channels)}"
            )
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        shallow_channels, deep_channels = map(int, in_channels)
        self.in_channels = (shallow_channels, deep_channels)
        self.out_channels = int(out_channels)

        self.shallow_projection = (
            nn.Identity()
            if shallow_channels == out_channels
            else ConvBNAct(shallow_channels, out_channels)
        )
        self.deep_projection = (
            nn.Identity()
            if deep_channels == out_channels
            else ConvBNAct(deep_channels, out_channels)
        )

        # Common-information-only weight generator.
        self.weight_generator = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward_with_weight(
            self,
            features: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError("CACF_Common forward expects [F_s, F_h]")

        shallow, deep = features
        shallow = self.shallow_projection(shallow)
        deep = self.deep_projection(deep)

        if deep.shape[-2:] != shallow.shape[-2:]:
            deep = F.interpolate(
                deep,
                size=shallow.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        common = shallow + deep
        weight = torch.sigmoid(self.weight_generator(common))
        fused = weight * shallow + (1.0 - weight) * deep
        return fused, weight

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        fused, _ = self.forward_with_weight(features)
        return fused
if __name__ == "__main__":
    module = CACF_Common(in_channels=[256, 256], out_channels=256)
    shallow = torch.randn(2, 256, 40, 40, requires_grad=True)
    deep = torch.randn(2, 256, 40, 40, requires_grad=True)
    output, weight = module.forward_with_weight([shallow, deep])
    output.mean().backward()
    print("output:", tuple(output.shape))
    print("weight:", tuple(weight.shape))