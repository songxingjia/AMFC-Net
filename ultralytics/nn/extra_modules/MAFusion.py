import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class ChannelAttentionIR(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(dim, hidden, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden, dim, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = self.fc(self.gmp(x).view(b, c))
        return self.sigmoid(attn).view(b, c, 1, 1)


class Laplacian(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        _, c, _, _ = x.size()
        weight = self.weight.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)


class SpatialAttentionIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobelxy()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode="reflect", bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_edge = self.sobel(torch.mean(x, dim=1, keepdim=True))
        x_edge = (x_edge - x_edge.min()) / (x_edge.max() - x_edge.min() + 1e-6)
        return self.sigmoid(self.conv(x_edge))


class ChannelAttentionVIS(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(dim, hidden, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden, dim, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = self.fc(self.gap(x).view(b, c))
        return self.sigmoid(attn).view(b, c, 1, 1)


class SpatialAttentionVIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = Laplacian()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode="reflect", bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_laplacian = torch.abs(self.laplacian(torch.mean(x, dim=1, keepdim=True)))
        x_laplacian = (x_laplacian - x_laplacian.min()) / (x_laplacian.max() - x_laplacian.min() + 1e-6)
        return self.sigmoid(self.conv(x_laplacian))


class MAFusion(nn.Module):
    def __init__(self, in_dims, out_dim, reduction_ir=8, reduction_vis=16):
        super().__init__()
        self.channel_attn_ir = ChannelAttentionIR(out_dim, reduction_ir)
        self.spatial_attn_ir = SpatialAttentionIR()
        self.channel_attn_vis = ChannelAttentionVIS(out_dim, reduction_vis)
        self.spatial_attn_vis = SpatialAttentionVIS()
        self.pa = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=True), nn.Sigmoid())
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)
        self.conv_vis = nn.Conv2d(in_dims[0], out_dim, kernel_size=1, bias=True)
        self.conv_ir = nn.Conv2d(in_dims[1], out_dim, kernel_size=1, bias=True)

    def forward(self, inputs):
        x_vis, x_ir = inputs
        x_vis = self.conv_vis(x_vis)
        x_ir = self.conv_ir(x_ir)

        x_ir_attn = x_ir * self.channel_attn_ir(x_ir)
        x_ir_attn = x_ir_attn * self.spatial_attn_ir(x_ir_attn)
        x_vis_attn = x_vis * self.channel_attn_vis(x_vis)
        x_vis_attn = x_vis_attn * self.spatial_attn_vis(x_vis_attn)

        attn_pixel = self.pa(torch.cat([x_ir_attn, x_vis_attn], dim=1))
        result = attn_pixel * x_ir_attn + (1 - attn_pixel) * x_vis_attn
        return self.conv(result)
