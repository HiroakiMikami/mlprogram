import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn.utils.rnn import PaddedSequenceWithMask
from nl2prog.nn.functional import gelu


class DecoderBlock(nn.Module):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int, out_size: int,
                 n_heads: int, dropout: float):
        super(DecoderBlock, self).__init__()
        self.ast_attention = nn.MultiheadAttention(feature_size, n_heads,
                                                   dropout)
        self.norm1 = nn.LayerNorm(feature_size)
        self.nl_attention = nn.MultiheadAttention(feature_size, n_heads,
                                                  dropout)
        self.norm2 = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(out_size)

    def forward(self, query: PaddedSequenceWithMask,
                nl_feature: PaddedSequenceWithMask,
                ast_feature: PaddedSequenceWithMask) -> \
            Tuple[PaddedSequenceWithMask, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query: PaddedSequenceWithMask
            (L_q, N, query_size) where L_q is the sequence length,
            N is the batch size.
        nl_feature: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        ast_feature: PaddedSequenceWithMask
            (L_ast, N, ast_feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        output: PaddedSequenceWithMask
            (L_q, N, out_size) where L_q is the sequence length,
            N is the batch_size.
        nl_attn_weights: torch.Tensor
            (N, L_nl, L_q) where N is the batch size,
            L_nl is the sequence length of NL query,
            L_q is the query sequence length.
        ast_attn_weights: torch.Tensor
            (N, L_ast, L_q) where N is the batch size,
            L_ast is the sequence length of ast,
            L_q is the query sequence length.
        """
        L_q, N, _ = query.data.shape
        h_in = query.data
        h, ast_attn = self.ast_attention(
            key=ast_feature.data, query=h_in, value=ast_feature.data,
            key_padding_mask=ast_feature.mask.permute(1, 0) == 0)
        h = h + h_in
        h = self.norm1(h)

        h_in = h
        h, nl_attn = self.nl_attention(
            key=nl_feature.data, query=h, value=nl_feature.data,
            key_padding_mask=nl_feature.mask.permute(1, 0) == 0)
        h = h + h_in
        h = self.norm2(h)

        h_in = h
        h = h * query.mask.to(h.dtype).reshape(L_q, N, 1)
        h = self.fc1(h.view(L_q * N, -1))
        h = self.dropout(h)
        h = gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h.view(L_q, N, -1)
        h = h * query.mask.to(h.dtype).reshape(L_q, N, 1)
        h = h + h_in
        h = self.norm3(h)

        return PaddedSequenceWithMask(h, query.mask), nl_attn, ast_attn


class Decoder(nn.Module):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int, out_size: int,
                 n_heads: int, dropout: float, n_blocks: int):
        super(Decoder, self).__init__()
        self.blocks = []
        for i in range(n_blocks):
            block = DecoderBlock(
                feature_size, hidden_size,
                out_size if i == n_blocks - 1 else feature_size,
                n_heads, dropout
            )
            self.blocks.append(block)
            self.add_module("block_{}".format(i), block)

    def forward(self, query: PaddedSequenceWithMask,
                nl_feature: torch.Tensor,
                ast_feature: torch.Tensor) -> \
            PaddedSequenceWithMask:
        """
        Parameters
        ----------
        query: PaddedSequenceWithMask
            (L_q, N, query_size) where L_q is the sequence length,
            N is the batch size.
        nl_feature: torch.Tensor
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        ast_feature: torch.Tensor
            (L_ast, N, ast_feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        output: PaddedSequenceWithMask
            (L_q, N, out_size) where L_q is the sequence length,
            N is the batch_size.
        """
        input = query
        for block in self.blocks:
            input, _, _ = block(query, nl_feature, ast_feature)
        return input
