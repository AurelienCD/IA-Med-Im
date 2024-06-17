"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def optimal_residual_block(in_channels, out_channels, time_embedding_dim: Optional[int] = None, downsample: int = 4) -> nn.Module:
    if in_channels >= 64:
        return ResidualBottleneckBlock(in_channels, out_channels, downsample, time_embedding_dim)
    else:
        return ResidualBlock(in_channels, out_channels, time_embedding_dim)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: Optional[int] = None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.adapt_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=False)

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, out_channels * 2))

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale, shift = 0, 1
        if hasattr(self, 'mlp') and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale, shift = time_embedding.chunk(2, dim=1)

        output = F.relu(self.bn1(self.conv1(x)))
        output = output * (scale + 1) + shift
        output = self.bn2(self.conv2(output))
        output = output + self.adapt_conv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        output = F.relu(output)
        return output


class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: int = 4, time_embedding_dim: Optional[int] = None) -> None:
        super(ResidualBottleneckBlock, self).__init__()

        assert in_channels % downsample == 0  # Checking that the in_channels can be divided by downsample

        self.conv1 = nn.Conv2d(in_channels, in_channels // downsample, kernel_size=1, stride=1, padding="same",
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // downsample)
        self.conv2 = nn.Conv2d(in_channels // downsample, in_channels // downsample, kernel_size=3, stride=1,
                               padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // downsample)
        self.conv3 = nn.Conv2d(in_channels // downsample, out_channels, kernel_size=1, stride=1, padding="same",
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.adapt_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=False)

        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, (in_channels // downsample) * 2))

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale, shift = 0, 1
        if hasattr(self, 'mlp') and time_embedding is not None:
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale, shift = time_embedding.chunk(2, dim=1)

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = output * (scale + 1) + shift  # application de timestep embedding

        output = self.conv3(output) + self.adapt_conv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        output = F.relu(self.bn3(output))
        return output
