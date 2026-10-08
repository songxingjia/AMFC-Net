import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

try:
    from mmcv.ops.carafe import CARAFE as MMCV_CARAFE
except ImportError:
    MMCV_CARAFE = None


class LPRM_efficient(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, upscale: int = 1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.upscale = upscale

        self.scale_factor = dilation
        self.scale_sq = self.scale_factor * self.scale_factor
        self.carafe_op = None
        if MMCV_CARAFE is not None:
            self.carafe_op = MMCV_CARAFE(kernel_size=self.kernel_size, group_size=1, scale_factor=upscale)

        if self.scale_factor > 1:
            self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
            self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def _apply_carafe_or_fallback(self, value_feat: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
        if self.carafe_op is not None:
            mask = mask_pred.softmax(dim=1).to(value_feat.dtype)
            return self.carafe_op(value_feat, mask)
        return F.interpolate(
            value_feat,
            size=mask_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, mask_pred: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1:
            return self._apply_carafe_or_fallback(value_feat, mask_pred)

        b, c, h, w = value_feat.shape
        k_sq = self.kernel_size * self.kernel_size
        s = self.scale_factor
        hs, ws = h // s, w // s

        unshuffled_value = self.pixel_unshuffle(value_feat)
        unshuffled_value = unshuffled_value.view(b, c, self.scale_sq, hs, ws).permute(0, 2, 1, 3, 4)
        unshuffled_value = unshuffled_value.reshape(b * self.scale_sq, c, hs, ws)

        unshuffled_mask = self.pixel_unshuffle(mask_pred)
        unshuffled_mask = unshuffled_mask.view(b, k_sq, self.scale_sq, hs, ws).permute(0, 2, 1, 3, 4)
        unshuffled_mask = unshuffled_mask.reshape(b * self.scale_sq, k_sq, hs, ws)

        refined_unshuffled = self._apply_carafe_or_fallback(unshuffled_value, unshuffled_mask)
        refined_unshuffled = refined_unshuffled.view(b, self.scale_sq, c, hs, ws)
        refined_shuffled = refined_unshuffled.permute(0, 2, 1, 3, 4).reshape(b, c * self.scale_sq, hs, ws)
        return self.pixel_shuffle(refined_shuffled)


class LPRMAlignUpModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        align_corners: bool = False,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.align_corners = align_corners

        low_res_channels, high_res_channels = in_channels
        predictor_in_channels = low_res_channels + high_res_channels
        compressed_channels = max(predictor_in_channels // compress_ratio, 1)

        self.compress_conv_low = nn.Conv2d(low_res_channels, compressed_channels, 1, bias=True)
        self.compress_conv_high = nn.Conv2d(high_res_channels, compressed_channels, 1, bias=True)
        self.lprm = LPRM_efficient(channels=compressed_channels, kernel_size=kernel_size, upscale=2)
        self.lpr_conv = nn.Conv2d(compressed_channels, kernel_size**2, 3, padding=1)
        self.conv_final = Conv(low_res_channels, out_channels, 1)

    def forward(self, inputs) -> torch.Tensor:
        x_low, guidance_high_aligned = inputs
        _, _, target_h, target_w = guidance_high_aligned.shape

        compressed_low = self.compress_conv_low(x_low)
        compressed_high = self.compress_conv_high(guidance_high_aligned)

        lpr_feat = F.interpolate(
            self.lpr_conv(compressed_low),
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        ) + F.interpolate(
            self.lpr_conv(compressed_high),
            size=(x_low.size(-2) * 2, x_low.size(-1) * 2),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        x_low = self.lprm(lpr_feat, x_low)
        x_low_upsampled = F.interpolate(
            x_low,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return self.conv_final(x_low_upsampled)
