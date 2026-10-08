import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class Branch(nn.Module):
    def __init__(self, channels: int, dw_expand: int, dilation: int = 1) -> None:
        super().__init__()
        dw_channels = dw_expand * channels
        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels=dw_channels,
                out_channels=dw_channels,
                kernel_size=3,
                padding=dilation,
                stride=1,
                groups=dw_channels,
                bias=True,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, channels, 1, 1) * y + bias.view(1, channels, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, _, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FreMLP(nn.Module):
    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(channels, expand * channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_freq = torch.fft.rfft2(x, norm="backward")

        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        return torch.fft.irfft2(x_out, s=(height, width), norm="backward")


class FrequencyCM(nn.Module):
    def __init__(
        self,
        inc: int,
        ouc: int,
        dw_expand: int = 2,
        dilations: list[int] | tuple[int, ...] = (1,),
        extra_depth_wise: bool = True,
    ) -> None:
        super().__init__()
        self.dw_channel = dw_expand * inc
        self.extra_conv = (
            nn.Conv2d(
                inc,
                inc,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=inc,
                bias=True,
                dilation=1,
            )
            if extra_depth_wise
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(
            in_channels=inc,
            out_channels=self.dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )

        self.branches = nn.ModuleList([Branch(inc, dw_expand, dilation=d) for d in dilations])
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=self.dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(
            in_channels=self.dw_channel // 2,
            out_channels=inc,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )

        self.norm1 = LayerNorm2d(inc)
        self.norm2 = LayerNorm2d(inc)
        self.freq = FreMLP(channels=inc, expand=1)
        self.gamma = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True)
        self.conv_final = Conv(inc, ouc, k=1) if inc != ouc else nn.Identity()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x_step1 = self.norm1(inp)
        x_freq = self.freq(x_step1)
        x = inp + x_freq * self.gamma
        x_low = x
        x_hf = self.norm2(x)
        x_hf = self.conv1(self.extra_conv(x_hf))

        z = 0
        for branch in self.branches:
            z = z + branch(x_hf)

        z = self.sg1(z)
        x_hf = self.sca(z) * z
        x_high = self.conv3(x_hf)
        y = x_low + x_high * self.beta
        return self.conv_final(y)
