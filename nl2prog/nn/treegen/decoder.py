import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn.utils.rnn import PaddedSequenceWithMask
from nl2prog.nn.functional import gelu
from nl2prog.nn.treegen.embedding import ActionEmbedding


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
            (L_ast, N, query_size) where L_ast is the sequence length,
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
            (L_ast, N, out_size) where L_ast is the sequence length,
            N is the batch_size.
        nl_attn_weights: torch.Tensor
            (N, L_nl, L_ast) where N is the batch size,
            L_nl is the sequence length of NL query,
            L_ast is the ast sequence length.
        ast_attn_weights: torch.Tensor
            (N, L_ast, L_ast) where N is the batch size,
            L_ast is the sequence length of ast.
        """
        L_ast, N, _ = query.data.shape
        device = query.data.device
        attn_mask = \
            torch.nn.Transformer.generate_square_subsequent_mask(None, L_ast)\
            .to(device=device)
        h_in = query.data
        h, ast_attn = self.ast_attention(
            key=ast_feature.data, query=h_in, value=ast_feature.data,
            key_padding_mask=ast_feature.mask.permute(1, 0) == 0,
            attn_mask=attn_mask)
        h = h + h_in
        h = self.norm1(h)

        h_in = h
        h, nl_attn = self.nl_attention(
            key=nl_feature.data, query=h, value=nl_feature.data,
            key_padding_mask=nl_feature.mask.permute(1, 0) == 0)
        h = h + h_in
        h = self.norm2(h)

        h_in = h
        h = h * query.mask.to(h.dtype).reshape(L_ast, N, 1)
        h = self.fc1(h.view(L_ast * N, -1))
        h = self.dropout(h)
        h = gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h.view(L_ast, N, -1)
        h = h * query.mask.to(h.dtype).reshape(L_ast, N, 1)
        h = h + h_in
        h = self.norm3(h)

        return PaddedSequenceWithMask(h, query.mask), nl_attn, ast_attn


class Decoder(nn.Module):
    def __init__(self,
                 rule_num: int, token_num: int, feature_size: int,
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

        self.action_embedding = \
            ActionEmbedding(rule_num, token_num, feature_size)

    def forward(self, query: PaddedSequenceWithMask,
                nl_feature: PaddedSequenceWithMask,
                other_feature: None,
                ast_feature: PaddedSequenceWithMask,
                states: None = None) -> Tuple[PaddedSequenceWithMask, None]:
        """
        Parameters
        ----------
        query: PaddedSequenceWithMask
            (L_ast, N, 3) where L_ast is the sequence length,
            N is the batch size.
            Each action will be encoded by the tuple of
            (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).
            The padding value should be -1.
        nl_feature: torch.Tensor
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        other_feature
            dummy arguments
        ast_feature: torch.Tensor
            (L_ast, N, ast_feature_size) where L_ast is the sequence length,
            N is the batch size.
        states
            dummy arguments

        Returns
        -------
        output: PaddedSequenceWithMask
            (L_ast, N, out_size) where L_ast is the sequence length,
            N is the batch_size.
        states:
        """
        embed = \
            self.action_embedding(query.data)
        input = PaddedSequenceWithMask(embed, query.mask)
        for block in self.blocks:
            input, _, _ = block(input, nl_feature, ast_feature)
        return input, _
