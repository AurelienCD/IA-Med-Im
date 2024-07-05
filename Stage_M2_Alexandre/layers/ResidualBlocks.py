"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.CNNBlock import CNNBlock


def optimal_residual_block(in_channels, out_channels, time_embedding_dim: Optional[int] = None, downsample: int = 4) -> nn.Module:
        return ResidualBlock(in_channels, out_channels, time_embedding_dim)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: Optional[int] = None, groups=None, dropout=0.0) -> None:
        super(ResidualBlock, self).__init__()
        self.cnn_block1 = CNNBlock(in_channels, out_channels, groups)
        self.cnn_block2 = CNNBlock(out_channels, out_channels, groups, dropout=dropout)

        self.adapt_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, out_channels * 2))

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = 0, 1
        if hasattr(self, 'mlp') and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        output = self.cnn_block1(x, scale_shift)
        output = self.cnn_block2(output)
        output = output + self.adapt_conv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        return output


class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: int = 4, time_embedding_dim: Optional[int] = None) -> None:
        super(ResidualBottleneckBlock, self).__init__()

        assert in_channels % downsample == 0  # Checking that the in_channels can be divided by downsample

        self.cnn_block1 = CNNBlock(in_channels, in_channels // downsample)
        self.cnn_block2 = CNNBlock(in_channels // downsample, in_channels // downsample)
        self.cnn_block3 = CNNBlock(in_channels // downsample, out_channels)

        self.adapt_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, (in_channels // downsample) * 2))

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = 0, 1
        if hasattr(self, 'mlp') and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        output = self.cnn_block1(x)
        output = self.cnn_block2(output, scale_shift)
        output = self.cnn_block3(output)
        output = self.conv3(output) + self.adapt_conv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        return output
