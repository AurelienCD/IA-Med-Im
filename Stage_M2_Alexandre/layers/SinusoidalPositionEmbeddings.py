"""
Author: Alexandre LECLERCQ
License: MIT
"""
import math
import torch

import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    based on the equation of Vaswani. et al. "Attention is all you need" p.6
    """

    def __init__(self, embedding_dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = embedding_dim

    def forward(self, position: torch.Tensor):
        """
        :param position: batch of positions : batch_size X 1
        :return: batch of position embeddings : batch_size X embedding_dim
        """
        device = position.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(- torch.arange(half_dim, device=device) * embeddings)  # vector 1D of length half_dim
        embeddings = position.reshape(-1, 1) * embeddings.reshape(1, -1)  # batch_size X 1  *  1 X dim / 2 = batch_size X dim/2
        # batch_size X (dim/2 + dim/2) = batch_size X dim
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
