# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tiny-object modules for YOLO26: SPD stem + multi-scale P3 fusion. Export-safe (reshape + standard convs only)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("SPDConv", "MSFuse")


class SPDConv(nn.Module):
    """Space-to-depth downsampling (Sunkara & Luo 2022).

    Rearranges 2x2 pixel blocks into channels, then applies a non-strided conv. Replaces a stride-2 Conv without
    discarding tiny-object signal; output spatial size is H/2 x W/2 so pyramid strides are preserved.
    """

    def __init__(self, c1: int, c2: int, k: int = 3):
        """Initialize SPDConv with input/output channels and kernel size."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Space-to-depth rearrange followed by a standard convolution."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)


class MSFuse(nn.Module):
    """Multi-scale P3 context enrichment via 1x1 projections, nearest upsample, and a learned gated add."""

    def __init__(self, c2: int, ch: list[int]):
        """Initialize MSFuse with output channels c2 and input channel list ch=[c_p3, c_p4, c_p5, ...]."""
        super().__init__()
        self.base = Conv(ch[0], c2, 1, 1) if ch[0] != c2 else nn.Identity()
        self.proj = nn.ModuleList(Conv(c, c2, 1, 1) for c in ch[1:])
        self.gate = nn.Parameter(torch.zeros(1, c2, 1, 1))  # sigmoid(0)=0.5 start

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse P3 with upsampled, projected coarser-scale features."""
        p3 = self.base(xs[0])
        fused = sum(
            F.interpolate(proj(x), size=p3.shape[2:], mode="nearest") for proj, x in zip(self.proj, xs[1:])
        )
        return p3 + torch.sigmoid(self.gate) * fused
