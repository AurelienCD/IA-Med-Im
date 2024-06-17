"""
Alexandre LECLERCQ
Licence: MIT

This module contain all the Utils layer who are too tiny to constitute their own module by themselves.
All those modules aren't planned to be used as standalone element but more as foundation block to build real layers
"""

import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x) + x


class PreNorm(nn.Module):
    def __init__(self, type: str, dim, block: nn.Module):

        super().__init__()

        if type == 'LN':
            self.norm = nn.LayerNorm(dim)
        elif type == 'BN':
            self.norm = nn.BatchNorm2d(dim)
        elif type == 'GN':
            self.norm = nn.GroupNorm(dim, dim)
        else:
            raise NotImplementedError(type)

        self.block = block

    def forward(self, x):
        return self.block(self.norm(x))


class Dropout(nn.Module):
    def __init__(self, block: nn.Module, prob: float):
        super().__init__()
        self.block = block
        self.prob = prob

    def forward(self, x):
        return nn.Dropout(p=self.prob)(self.block(x))
