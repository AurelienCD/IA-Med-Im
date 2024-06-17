"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
from einops import rearrange, einsum


class Attention(nn.Module):
    """
    This an implementation of the multi-head attention mechanism.
    Based on: Vaswani. et al. "Attention is all you need" p.3-5
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim ** -0.5  # (1 / sqrt(dim))
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        # project x to learned q k v spaces
        # Q : b x hidden_dim x h x w
        # K : b x hidden_dim x h x w
        # V : b x hidden_dim x h x w
        # qkv: Tuple(Q, K, V)
        qkv: Tuple[torch.Tensor] = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            # b : batch_size
            # (head d) : correspond to hidden_dim
            # however, we know that hidden_dim = head * d_k, d_v, d_q
            # Thus, we decompose (h c) to
            # h : the number of heads
            # d : the dimension of K, V or Q (d_k, d_v, d_q)
            # finally, we linearize the height and width
            # we apply this operation to element of the list qkv which contains the 3 tensor Q, K and V.
            lambda t: rearrange(t, 'b (head d) h w -> b head d (h w)', head=self.heads), qkv
        )

        # dot product (Q K^T)
        similarities: torch.Tensor = einsum(q, k, "b head d i, b head d j -> b head i j")
        similarities *= self.scale
        # we subtract the max value to ensure stable computation of the softmax
        # (the exponential grows too fast and can quickly reach uncomputable value)
        similarities = similarities - similarities.amax(dim=-1, keepdim=True).detach()
        attention: torch.Tensor = torch.softmax(similarities, dim=-1)

        # dot product (Attention V^T)
        output = einsum(attention, v, "b head i j, b head d k -> b head i d")
        # we rearrange the output array with the usual shape for convolution:
        # batch x channel x height x width
        # each channel correspond to the i-th attention head.
        output = rearrange(output, 'b head (x y) d -> b (head d) x y', x=h, y=w)

        return self.to_out(output)


class LinearAttention(nn.Module):
    """
    This an implementation of the  multi-head linear attention mechanism.
    Based on: Katharopoulos. et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    """

    def __init__(self, dim, heads=4, dim_head=32):
        """
        :param dim: the output channel dimension
        :param heads: number of attention heads
        :param dim_head: the dimension for each attention head
        """
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv: Tuple[torch.Tensor] = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) h w -> b head d (h w)', head=self.heads), qkv
        )

        q = torch.softmax(q, dim=-1)
        k = torch.softmax(q, dim=-2)

        global_context: torch.Tensor = einsum(k, v, "b head k n, b head v n -> b head k v")

        # reminder: dim_q = dim_k
        output: torch.Tensor = einsum(global_context, q, 'b head k v, b head k n -> b head v n')
        output = rearrange(output, 'b head c (x y) -> b (head c) x y', x=h, y=w)

        return self.to_out(output)
