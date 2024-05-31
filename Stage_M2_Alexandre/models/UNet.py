"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
import torch.nn as nn
from layers.ResidualBlocks import ResidualBottleneckBlock, ResidualBlock
from models.BaseModel import BaseModel


class UNet(BaseModel):

    def __init__(self, in_channels: int, depth: int, num_classes: int, task_name: str, dataset_name: str) -> None:
        super().__init__(task_name=task_name, dataset_name=dataset_name)

        self.depth = depth

        self.encoder_blocks = nn.ModuleDict([['firstBlock', ResidualBlock(in_channels=in_channels, out_channels=64)]])

        for i in range(depth):
            self.encoder_blocks['conv_encoder{}'.format(i + 1)] = self.__contraction_block(in_channels=64 * (2 ** i), out_channels=64 * (2 ** i) * 2)

        self.decoder_blocks = nn.ModuleDict([['midDecoder', nn.ConvTranspose2d(in_channels=64 * (2 ** (depth - 1)) * 2,
                                              out_channels=64 * (2 ** (depth - 1)),
                                              kernel_size=3, stride=2, padding=1, output_padding=1)]])

        for i in range(depth - 1):
            self.decoder_blocks['conv_decoder{}'.format(depth - i - 1)] = self.__expansion_block(in_channels=64 * (2 ** (depth - i - 2)) * 2 * 2, out_channels=64 * (2 ** (depth - i - 2)))

        self.decoder_blocks['final_block'] = ResidualBottleneckBlock(in_channels=64 * 2, out_channels=num_classes)

    def forward(self, x) -> torch.Tensor:
        encode_blocks = [self.encoder_blocks['firstBlock'](x)]

        for i in range(self.depth):
            encode_blocks.append(self.encoder_blocks['conv_encoder{}'.format(i + 1)](encode_blocks[-1]))

        decode_blocks = [self.decoder_blocks['midDecoder'](encode_blocks[-1])]

        for i in range(self.depth - 1):
            cat_layer = torch.cat((decode_blocks[-1], encode_blocks[self.depth - i - 1]), 1)
            decode_blocks.append(self.decoder_blocks['conv_decoder{}'.format(self.depth - i - 1)](cat_layer))

        cat_layer = torch.cat((decode_blocks[-1], encode_blocks[0]), dim=1)
        result = self.decoder_blocks['final_block'](cat_layer)

        return result

    @staticmethod
    def __contraction_block(in_channels, out_channels):
        block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBottleneckBlock(in_channels, out_channels)
        )
        return block

    @staticmethod
    def __expansion_block(in_channels, out_channels):
        block = nn.Sequential(
            ResidualBottleneckBlock(in_channels, out_channels),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )
        return block
