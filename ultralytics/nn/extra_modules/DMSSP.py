import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class SpatialAttention_DMSSP(nn.Module):
    def forward(self, x):
        mean_values = torch.mean(x, dim=(2, 3), keepdim=True)
        max_values = torch.amax(x, dim=(2, 3), keepdim=True)
        min_values = torch.amin(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(-2, -1), keepdim=True)
        q_ij = (((x - mean_values) ** 2) / (max_values - min_values + 1e-8)) * var
        return torch.sigmoid(q_ij) * x


class DMSSP(nn.Module):
    def __init__(self, in_channels, out_channels, channel_split=(1, 1, 2)):
        super().__init__()
        if in_channels % sum(channel_split) != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by sum(channel_split)={sum(channel_split)}")

        split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(split_ratio[1] * in_channels)
        self.embed_dims_2 = int(split_ratio[2] * in_channels)
        self.embed_dims_0 = in_channels - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = in_channels

        self.spatial_att = SpatialAttention_DMSSP()
        self.belt = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1)

        self.atrous_block6 = Conv(self.embed_dims_2, self.embed_dims_2, k=3, d=6)
        self.atrous_block12 = Conv(self.embed_dims_1, self.embed_dims_1, k=3, d=12)
        self.atrous_block18 = Conv(self.embed_dims_0, self.embed_dims_0, k=3, d=18)
        self.pw_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = F.interpolate(self.conv(self.mean(x)), size=size, mode="bilinear")
        x_s = image_features + self.belt * self.spatial_att(x)

        x_6 = self.atrous_block6(x[:, : self.embed_dims_2, ...])
        x_12 = self.atrous_block12(x[:, self.embed_dims_2 : self.embed_dims_2 + self.embed_dims_1, ...])
        x_18 = self.atrous_block18(x[:, self.embed_dims - self.embed_dims_0 :, ...])
        x = self.pw_conv(torch.cat([x_6, x_12, x_18], dim=1))
        return self.proj(x + x_s)
