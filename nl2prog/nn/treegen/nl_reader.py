import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn import SeparableConv1d
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask
from nl2prog.nn.functional \
    import index_embeddings, gelu, lne_to_nel, nel_to_lne
from .gating import Gating


class NLReaderBlock(nn.Module):
    def __init__(self,
                 char_embed_size: int, hidden_size: int,
                 n_heads: int, dropout: float, block_idx: int):
        super(NLReaderBlock, self).__init__()
        self.block_idx = block_idx
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.gating = Gating(hidden_size, char_embed_size, hidden_size,
                             hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv1 = SeparableConv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv2 = SeparableConv1d(hidden_size, hidden_size, 3, padding=1)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: PaddedSequenceWithMask,
                char_embed: torch.Tensor) -> \
            Tuple[PaddedSequenceWithMask, torch.Tensor]:
        """
        Parameters
        ----------
        input: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        char_embed: torch.Tensor
            (L, N, char_embed_size) where L is the sequence length,
            N is the batch size.

        Returns
        -------
        output: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        attn_weights: torch.Tensor
            (N, L, L) where N is the batch size and L is the sequence length.
        """
        L, N, _ = input.data.shape
        h_in = input.data
        h = h_in + index_embeddings(h_in, self.block_idx)
        h, attn = self.attention(
            h, h, h,
            key_padding_mask=input.mask.permute(1, 0) == 0)
        h = h + h_in
        h = self.norm1(h)

        h_in = h
        h = self.gating(h, char_embed)
        h = self.dropout(h)
        h = h + h_in
        h = self.norm2(h)

        h_in = h
        h = h * input.mask.to(h.dtype).reshape(L, N, 1)
        h = lne_to_nel(h)
        h = self.conv1(h)
        h = self.dropout(h)
        h = gelu(h)
        h = h * input.mask.to(h.dtype).reshape(L, N, 1).permute(1, 2, 0)
        h = self.conv2(h)
        h = self.dropout(h)
        h = gelu(h)
        h = nel_to_lne(h)
        h = h + h_in
        h = self.norm3(h)

        return PaddedSequenceWithMask(h, input.mask), attn


class NLReader(nn.Module):
    def __init__(self,
                 char_embed_size: int, hidden_size: int,
                 n_heads: int, dropout: float, n_blocks: int):
        super(NLReader, self).__init__()
        self.blocks = [NLReaderBlock(
            char_embed_size, hidden_size, n_heads, dropout, i
        ) for i in range(n_blocks)]
        for i, block in enumerate(self.blocks):
            self.add_module("block_{}".format(i), block)

    def forward(self, token_embed: PaddedSequenceWithMask,
                char_embed: torch.Tensor) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        token_embed: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        char_embed: torch.Tensor
            (L, N, char_embed_size) where L is the sequence length,
            N is the batch size.

        Returns
        -------
        PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        input = token_embed
        for block in self.blocks:
            input, _ = block(input, char_embed)
        return input
