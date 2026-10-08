import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierSR(nn.Module):
    """Fourier-domain spectral refinement block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        if in_channels % num_blocks != 0:
            raise ValueError(
                f"in_channels {in_channels} should be divisible by num_blocks {num_blocks}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = in_channels // self.num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2)
        )
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x
        dtype = x.dtype
        x = x.float()
        batch, channels, height, width = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        x = x.reshape(batch, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        weight = torch.view_as_complex(self.w.float().contiguous())
        x = torch.einsum("bkihw,kio->bkohw", x, weight)

        w1, w2, b = self.w1.float(), self.w2.float(), self.b.float()
        o1_real = F.relu(
            torch.mul(x.real, w1[0].unsqueeze(dim=0))
            - torch.mul(x.imag, w1[1].unsqueeze(dim=0))
            + b[0, :, :, None, None]
        )
        o1_imag = F.relu(
            torch.mul(x.imag, w2[0].unsqueeze(dim=0))
            + torch.mul(x.real, w2[1].unsqueeze(dim=0))
            + b[1, :, :, None, None]
        )

        x = torch.stack([o1_real, o1_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(batch, channels, x.shape[3], x.shape[4])

        x = torch.fft.irfft2(x, s=(height, width), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return self.conv_1x1(x + bias)
