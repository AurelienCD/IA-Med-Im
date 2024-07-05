import torch.nn as nn
from layers.WeightStandardizedConv2d import WeightStandardizedConv2d


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, groups=None, dropout=0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channel) if groups is None else nn.GroupNorm(groups, in_channel)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv = WeightStandardizedConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv(x)

        if scale_shift is not None:  # rescaling and shifting based on embedding
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return x
