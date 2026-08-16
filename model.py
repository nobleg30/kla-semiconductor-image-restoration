import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        mean = xf.mean(dim=1, keepdim=True)
        var = (xf - mean).square().mean(dim=1, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + self.eps)
        y = y.to(dtype)
        return y * self.weight.to(dtype) + self.bias.to(dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class FastNAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, dw_channels, 1)
        self.dw = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()

        gated_channels = dw_channels // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Conv2d(gated_channels, gated_channels, 1)
        self.pw2 = nn.Conv2d(gated_channels, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, 1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg1(y)
        y = y * self.sca(self.pool(y))
        y = self.pw2(y)
        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg2(y)
        y = self.ffn2(y)
        return x + self.gamma * y


class FastNAFSR(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 12, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[FastNAFBlock(width) for _ in range(blocks)])
        self.up = nn.Conv2d(width, scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            x.clamp(0.0, 1.0),
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        f = self.intro(x)
        f = self.body(f)
        residual = self.shuffle(self.up(f))
        return base + residual
