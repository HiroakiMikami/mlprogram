from typing import Tuple

import torch
import torch.nn as nn

from mlprogram.nn import EmbeddingWithMask, SeparableConv1d
from mlprogram.nn.functional import gelu, index_embeddings, lne_to_nel, nel_to_lne
from mlprogram.nn.treegen.embedding import ElementEmbedding
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
                 n_token: int, n_char: int, max_token_length: int,
                 char_embed_size: int, hidden_size: int,
                 n_head: int, dropout: float, n_block: int):
        super().__init__()
        self.n_char = n_char
        self.query_embed = nn.Embedding(n_token, hidden_size)
        self.query_elem_embed = ElementEmbedding(
            EmbeddingWithMask(n_char, hidden_size,
                              n_char),
            max_token_length, hidden_size, char_embed_size)

        self.blocks = [EncoderBlock(
            char_embed_size, hidden_size, n_head, dropout, i
        ) for i in range(n_block)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)

    def forward(self,
                word_nl_query: PaddedSequenceWithMask,
                char_nl_query: PaddedSequenceWithMask) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        word_nl_query: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length).
        char_nl_query: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length, max_token_len).
            The padding value should be -1.

        Returns
        -------
        nl_query_features: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        token_nl_query = word_nl_query
        e_token_query = self.query_embed(token_nl_query.data)
        char_nl_query = torch.where(
            char_nl_query.data != -1,
            char_nl_query.data,
            torch.full_like(char_nl_query.data, self.n_char)
        )
        e_char_query = self.query_elem_embed(char_nl_query)
        block_input = PaddedSequenceWithMask(e_token_query,
                                             token_nl_query.mask)
        for block in self.blocks:
            block_input, _ = block(block_input, e_char_query)
        return block_input
