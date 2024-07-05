"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

from typing import Optional
import torch
import torch.nn as nn
from layers.Utils import Residual, PreNorm
from layers.Attention import LinearAttention, Attention
from layers.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings
from layers.ResidualBlocks import optimal_residual_block


class TemporalUNet(nn.Module):

    def __init__(self, in_channels, out_channels, channel_scale=32, channels_mult=(1, 2, 4, 8, 16),
                 condition: bool = False):
        super(TemporalUNet, self).__init__()

        self.condition = condition

        # if condition is true
        # the model receive a concat input of real input + condition
        in_channels = in_channels * 2 if condition else in_channels

        channels = [i * channel_scale for i in channels_mult]
        in_out_channels = list(zip(channels[:-1], channels[1:]))

        time_embedding_dim = channel_scale * 4

        self.time_embeddings = nn.Sequential(
            SinusoidalPositionEmbeddings(channel_scale),
            nn.Linear(in_features=channel_scale, out_features=time_embedding_dim),
            nn.GELU(),
            nn.Linear(in_features=time_embedding_dim, out_features=time_embedding_dim),
        )

        self.start_block = nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=1)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        for i, (in_dim, out_dim) in enumerate(in_out_channels):
            self.down_blocks.append(self.__contraction_block(in_channels=in_dim, out_channels=out_dim,
                                                             time_embedding_dim=time_embedding_dim))

        self.middle_block = nn.ModuleList([
            optimal_residual_block(channels[-1], channels[-1], time_embedding_dim),
            # use of dot-product attention block for the middle_block to avoid to lose too many details
            Residual(PreNorm('BN', channels[-1], Attention(dim=channels[-1]))),
            optimal_residual_block(channels[-1], channels[-1], time_embedding_dim)
        ])

        for i, (out_dim, in_dim) in enumerate(reversed(in_out_channels)):
            self.up_blocks.append(
                self.__expansion_block(in_channels=in_dim, out_channels=out_dim, time_embedding_dim=time_embedding_dim))

        self.final_block = nn.ModuleList([
            optimal_residual_block(in_channels=channels[0] * 2, out_channels=channels[0],
                                   time_embedding_dim=time_embedding_dim),
            nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1),
        ])

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                conditionnal_input: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        :param x: Batch x Channels x Height x Width
        :param timestep: Batch x 1
        :param conditionnal_input: Batch x Channels x Height x Width
        :return: Batch x Channels x Height x Width
        """
        if self.condition and conditionnal_input is not None:
            x = torch.cat((x, conditionnal_input), dim=1)

        t = self.time_embeddings(timestep)

        x = self.start_block(x)

        skip_connection = [x.clone()]

        for res_block1, res_block2, attention_block, down in self.down_blocks:
            x = res_block1(x, t)
            x = res_block2(x, t)
            x = attention_block(x)
            x = down(x)
            skip_connection.append(x)

        res_block1, attention_block, res_block2 = self.middle_block
        x = res_block1(x, t)
        x = attention_block(x)
        x = res_block2(x, t)

        for res_block1, res_block2, attention_block, up in self.up_blocks:
            skip_connection_features_map = skip_connection.pop()

            x = torch.cat((x, skip_connection_features_map), dim=1)
            x = res_block1(x, t)

            x = torch.cat((x, skip_connection_features_map), dim=1)
            x = res_block2(x, t)

            x = attention_block(x)
            x = up(x)

        x = torch.cat((x, skip_connection.pop()), dim=1)
        res_block, final_conv, = self.final_block
        x = res_block(x, t)
        x = final_conv(x)

        return x

    @staticmethod
    def __contraction_block(in_channels, out_channels, time_embedding_dim):
        block = nn.Sequential(
            optimal_residual_block(in_channels=in_channels, out_channels=in_channels,
                                   time_embedding_dim=time_embedding_dim),
            optimal_residual_block(in_channels=in_channels, out_channels=out_channels,
                                   time_embedding_dim=time_embedding_dim),
            Residual(PreNorm('BN', out_channels, LinearAttention(dim=out_channels))),
            nn.MaxPool2d(kernel_size=2),
        )
        return block

    @staticmethod
    def __expansion_block(in_channels, out_channels, time_embedding_dim):
        block = nn.Sequential(
            optimal_residual_block(in_channels=2 * in_channels, out_channels=in_channels,
                                   time_embedding_dim=time_embedding_dim),
            optimal_residual_block(in_channels=2 * in_channels, out_channels=in_channels,
                                   time_embedding_dim=time_embedding_dim),
            Residual(PreNorm('BN', in_channels, LinearAttention(dim=in_channels))),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )
        return block
