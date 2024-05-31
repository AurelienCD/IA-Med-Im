"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.adaptconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=False)

    def forward(self, x) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = output + self.adaptconv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        output = F.relu(output)
        return output


class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsample: int = 4) -> None:
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

        self.adaptconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=False)

    def forward(self, x) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.conv3(output) + self.adaptconv(
            x)  # Adding residual signal adapt to the same number of channels with a conv 1x1
        output = F.relu(self.bn3(output))
        return output
