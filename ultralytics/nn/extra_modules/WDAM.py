import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    DWTForward = None
    DWTInverse = None


def _meshgrid_ij(*tensors):
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        return torch.meshgrid(*tensors)


class WDAM(nn.Module):
    """Wavelet-domain attention module extracted from the YOLO11 source primitive."""

    def __init__(self, dim, num_heads=8, window_size=5, shift_size=2, bias=False):
        super().__init__()
        if DWTForward is None or DWTInverse is None:
            raise ImportError("WDAM requires pytorch_wavelets to be installed.")

        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dwt = DWTForward(J=1, wave="haar")
        self.idwt = DWTInverse(wave="haar")

        self.high_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
        )
        self.high_out = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(_meshgrid_ij(torch.arange(window_size), torch.arange(window_size)))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def window_partition(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(-1, c, ws, ws)

    def window_reverse(self, windows, h, w):
        ws = self.window_size
        b = int(windows.shape[0] / (h * w / ws / ws))
        c = windows.shape[1]
        x = windows.view(b, h // ws, w // ws, c, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, c, h, w)

    def shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        return x

    def reverse_shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x

    def window_attention(self, q, k, v):
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)

        n = self.window_size * self.window_size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        return torch.matmul(v, attn.transpose(-2, -1))

    def forward(self, x):
        _, _, h, w = x.shape

        ll, yh = self.dwt(x)
        yh = yh[0]
        lh, hl, hh = yh[:, :, 0, :, :], yh[:, :, 1, :, :], yh[:, :, 2, :, :]

        filter_hv = self.high_conv(torch.cat([lh, hl], dim=1))

        qkv = self.qkv_dwconv(self.qkv(ll))
        q, k, v_inp = qkv.chunk(3, dim=1)
        v = v_inp * filter_hv + v_inp

        x_shifted = self.shift(ll, self.shift_size)
        q = self.window_partition(x_shifted)
        k = self.window_partition(x_shifted)
        v = self.window_partition(v)

        b_win, c_q, ws, _ = q.shape
        q = q.view(b_win, self.num_heads, c_q // self.num_heads, ws * ws)
        k = k.view(b_win, self.num_heads, c_q // self.num_heads, ws * ws)
        v = v.view(b_win, self.num_heads, c_q // self.num_heads, ws * ws)

        out = self.window_attention(q, k, v)
        out = out.view(b_win, c_q, ws, ws)
        out = self.window_reverse(out, h // 2, w // 2)
        out = self.reverse_shift(out, self.shift_size)
        out = self.project_out(out)

        yh = self.high_out(torch.cat([lh, hl, hh], dim=1))
        lh, hl, hh = yh.chunk(3, dim=1)
        yh = torch.stack([lh, hl, hh], dim=2)
        return self.idwt((out, [yh]))
