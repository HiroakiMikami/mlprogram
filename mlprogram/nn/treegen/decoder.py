from typing import Tuple, cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.functional import gelu
from mlprogram.nn.treegen.embedding import ElementEmbedding
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


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
                 rule_num: int, max_depth: int,
                 feature_size: int, hidden_size: int, out_size: int,
                 n_heads: int, dropout: float, n_blocks: int):
        super(Decoder, self).__init__()
        self.rule_num = rule_num
        self.blocks = []
        for i in range(n_blocks):
            block = DecoderBlock(
                feature_size, hidden_size,
                out_size if i == n_blocks - 1 else feature_size,
                n_heads, dropout
            )
            self.blocks.append(block)
            self.add_module(f"block_{i}", block)

        self.query_embed = ElementEmbedding(
            EmbeddingWithMask(rule_num, feature_size, rule_num),
            max_depth, feature_size, feature_size
        )

    def forward(self, inputs: Environment) -> Environment:
        """
        Parameters
        ----------
        nl_query_features: torch.Tensor
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        action_queries: PaddedSequenceWithMask
            (L_ast, N, max_depth) where L_ast is the sequence length,
            N is the batch size.
            This tensor encodes the path from the root node to the target node.
            The padding value should be -1.
        action_features: PaddedSequenceWithMask
            (L_ast, N, ast_feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        action_features: PaddedSequenceWithMask
            (L_ast, N, out_size) where L_ast is the sequence length,
            N is the batch_size.
        """
        nl_query_features = cast(PaddedSequenceWithMask,
                                 inputs.states["nl_query_features"])
        action_queries = cast(PaddedSequenceWithMask,
                              inputs.states["action_queries"])
        action_features = cast(PaddedSequenceWithMask,
                               inputs.states["action_features"])
        q = action_queries.data + \
            (action_queries.data == -1) * (self.rule_num + 1)
        embed = self.query_embed(q)
        input = PaddedSequenceWithMask(embed, action_queries.mask)
        for block in self.blocks:
            input, _, _ = block(input, nl_query_features, action_features)
        inputs.states["action_features"] = input
        return inputs
