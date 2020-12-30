from typing import Tuple

import torch
import torch.nn as nn

from mlprogram.nn import SeparableConv1d
from mlprogram.nn.functional import gelu, index_embeddings, lne_to_nel, nel_to_lne
from mlprogram.nn.treegen.gating import Gating
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class EncoderBlock(nn.Module):
    def __init__(self,
                 char_embed_size: int, hidden_size: int,
                 n_head: int, dropout: float, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.attention = nn.MultiheadAttention(hidden_size, n_head, dropout)
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


class Encoder(nn.Module):
    def __init__(self,
                 char_embedding_size: int, hidden_size: int,
                 n_head: int, dropout: float, n_block: int):
        super().__init__()
        self.blocks = [EncoderBlock(
            char_embedding_size, hidden_size, n_head, dropout, i
        ) for i in range(n_block)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)

    def forward(self,
                word_nl_feature: PaddedSequenceWithMask,
                char_nl_feature: PaddedSequenceWithMask) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        word_nl_feature: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length, hidden_size).
            The padding value should be -1.
        char_nl_feature: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length, char_embedding_size).
            The padding value should be -1.

        Returns
        -------
        nl_query_features: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        block_input = word_nl_feature
        for block in self.blocks:
            block_input, _ = block(block_input, char_nl_feature.data)
        return block_input
