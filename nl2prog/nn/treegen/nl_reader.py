import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn import SeparableConv1d, EmbeddingWithMask
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask
from nl2prog.nn.functional \
    import index_embeddings, gelu, lne_to_nel, nel_to_lne
from .gating import Gating
from .embedding import ElementEmbedding


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
                 token_num: int, char_num: int, max_token_len: int,
                 char_embed_size: int, hidden_size: int,
                 n_heads: int, dropout: float, n_blocks: int):
        super(NLReader, self).__init__()
        self.char_num = char_num
        self.query_embed = nn.Embedding(token_num, hidden_size)
        self.query_elem_embed = ElementEmbedding(
            EmbeddingWithMask(char_num, hidden_size,
                              char_num),
            max_token_len, hidden_size, char_embed_size)

        self.blocks = [NLReaderBlock(
            char_embed_size, hidden_size, n_heads, dropout, i
        ) for i in range(n_blocks)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)

    def forward(self, input: Tuple[PaddedSequenceWithMask,
                                   PaddedSequenceWithMask]) \
            -> Tuple[PaddedSequenceWithMask, None]:
        """
        Parameters
        ----------
        input
            token_query: rnn.PaddedSequenceWithMask
                The minibatch of sequences.
                The shape of each sequence is (sequence_length).
            char_query: rnn.PaddedSequenceWithMask
                The minibatch of sequences.
                The shape of each sequence is (sequence_length, max_token_len).
                The padding value should be -1.

        Returns
        -------
        PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        other_features: None
        """
        token_query, char_query = input
        e_token_query = self.query_embed(token_query.data)
        char_query = \
            char_query.data + (char_query.data == -1) * (self.char_num + 1)
        e_char_query = self.query_elem_embed(char_query)
        input = PaddedSequenceWithMask(e_token_query, token_query.mask)
        for block in self.blocks:
            input, _ = block(input, e_char_query)
        return input, None
