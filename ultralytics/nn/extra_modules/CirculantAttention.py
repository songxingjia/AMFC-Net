import torch
import torch.nn as nn


class Linear(nn.Linear):
    """Linear layer for complex-valued inputs."""

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__(in_features, out_features, False, device, dtype)

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        x = torch.nn.functional.linear(x, weight).transpose(-2, -1)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return torch.view_as_complex(x.contiguous())


class CirculantAttention(nn.Module):
    """Circulant Attention extracted from the YOLO11 source primitive."""

    def __init__(self, dim, proj_drop=0.0):
        super().__init__()
        self.qkv = Linear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.SiLU())
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        orig_dtype = x.dtype

        t = self.gate(x)
        x = x.permute(0, 2, 3, 1)
        if x.is_cuda and x.dtype == torch.float16:
            x = x.float()
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm="ortho")

        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))
        x = torch.conj(attn) * v
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm="ortho")

        x = x.permute(0, 3, 1, 2).to(orig_dtype) * t
        x = self.proj(x)
        return x
