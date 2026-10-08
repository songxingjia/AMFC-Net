import torch
import torch.nn as nn
from einops import rearrange

from ultralytics.nn.modules.conv import Conv


class CrossAttention_S(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, features):
        query_feature, value_feature = features
        _, _, height, width = query_feature.shape

        qk = self.qk_dwconv(self.qk(query_feature))
        query, key = qk.chunk(2, dim=1)
        value = self.v_dwconv(self.v(value_feature))

        query = rearrange(query, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        key = rearrange(key, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        value = rearrange(value, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        fused = attention @ value
        fused = rearrange(fused, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=height, w=width)
        return self.project_out(fused)


class HAFFormer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, bias=False):
        super().__init__()
        if len(in_dim) != 2:
            raise ValueError(f"HAFFormer expects two input feature maps, got {len(in_dim)}")

        self.conv1x1_1 = Conv(in_dim[0], out_dim, 1) if in_dim[0] != out_dim else nn.Identity()
        self.conv1x1_2 = Conv(in_dim[1], out_dim, 1) if in_dim[1] != out_dim else nn.Identity()

        self.mhca_rgb = CrossAttention_S(out_dim, num_heads=num_heads, bias=bias)
        self.mhca_ir = CrossAttention_S(out_dim, num_heads=num_heads, bias=bias)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
        )
        self.gate = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias)

    def forward(self, features):
        feature_a, feature_b = features

        feature_a = self.conv1x1_1(feature_a)
        feature_b = self.conv1x1_2(feature_b)

        refined_a = self.mhca_rgb([feature_a, feature_b]) + feature_a
        refined_b = self.mhca_ir([feature_b, feature_a]) + feature_b

        gate = self.gate(self.fuse(torch.cat((refined_a, refined_b), dim=1))).sigmoid()
        return gate * refined_a + (1.0 - gate) * refined_b
