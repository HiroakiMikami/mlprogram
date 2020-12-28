from typing import Tuple

import torch
import torch.nn as nn

from mlprogram.nn import EmbeddingWithMask, TreeConvolution
from mlprogram.nn.functional import (
    gelu,
    index_embeddings,
    lne_to_nel,
    nel_to_lne,
    position_embeddings,
)
from mlprogram.nn.treegen.embedding import (
    ActionEmbedding,
    ActionSignatureEmbedding,
    ElementEmbedding,
)
from mlprogram.nn.treegen.gating import Gating
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class ActionSequenceReaderBlock(nn.Module):
    def __init__(self,
                 rule_embed_size: int, hidden_size: int,
                 tree_conv_kernel_size: int,
                 n_head: int, dropout: float, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.attention = nn.MultiheadAttention(hidden_size, n_head, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.gating = Gating(hidden_size, rule_embed_size, hidden_size,
                             hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv1 = TreeConvolution(hidden_size, hidden_size,
                                     tree_conv_kernel_size)
        self.conv2 = TreeConvolution(hidden_size, hidden_size,
                                     tree_conv_kernel_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: PaddedSequenceWithMask,
                depth: torch.Tensor,
                rule_embed: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> \
            Tuple[PaddedSequenceWithMask, torch.Tensor]:
        """
        Parameters
        ----------
        input: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        depth: torch.Tensor
            (L, N) where L is the sequence length,
            N is the batch size.
        rule_embed: torch.Tensor
            (L, N, rule_embed_size) where L is the sequence length,
            N is the batch size.
        adjacency_matrix: torch.Tensor
            (N, L, L) where N is the batch size, L is the sequence length.

        Returns
        -------
        output: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        attn_weights: torch.Tensor
            (N, L, L) where N is the batch size and L is the sequence length.
        """
        L, N, hidden_size = input.data.shape
        device = input.data.device
        h_in = input.data
        h = h_in + \
            index_embeddings(h_in, self.block_idx) + \
            position_embeddings(depth, self.block_idx, hidden_size)
        attn_mask = \
            torch.nn.Transformer.generate_square_subsequent_mask(None, L)\
            .to(device=device)
        h, attn = self.attention(
            h, h, h,
            key_padding_mask=input.mask.permute(1, 0) == 0,
            attn_mask=attn_mask)
        h = h + h_in
        h = self.norm1(h)

        h_in = h
        h = self.gating(h, rule_embed)
        h = self.dropout(h)
        h = h + h_in
        h = self.norm2(h)

        h_in = h
        h = h * input.mask.to(h.dtype).reshape(L, N, 1)
        h = lne_to_nel(h)
        h = self.conv1(h, adjacency_matrix)
        h = self.dropout(h)
        h = gelu(h)
        h = h * input.mask.to(h.dtype).reshape(L, N, 1).permute(1, 2, 0)
        h = self.conv2(h, adjacency_matrix)
        h = self.dropout(h)
        h = gelu(h)
        h = nel_to_lne(h)
        h = h + h_in
        h = self.norm3(h)

        return PaddedSequenceWithMask(h, input.mask), attn


class DecoderBlock(nn.Module):
    def __init__(self,
                 feature_size: int,
                 hidden_size: int, out_size: int,
                 n_head: int, dropout: float):
        super().__init__()
        self.ast_attention = nn.MultiheadAttention(feature_size, n_head,
                                                   dropout)
        self.norm1 = nn.LayerNorm(feature_size)
        self.nl_attention = nn.MultiheadAttention(feature_size, n_head,
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
                 n_rule: int, n_token: int, n_node_type: int,
                 max_depth: int, max_arity: int,
                 rule_embedding_size: int,
                 encoder_hidden_size: int, decoder_hidden_size: int,
                 out_size: int,
                 tree_conv_kernel_size: int,
                 n_head: int,
                 dropout: float,
                 n_encoder_block: int,
                 n_decoder_block: int):
        super().__init__()
        self.n_rule = n_rule
        self.action_embed = ActionEmbedding(n_rule, n_token, encoder_hidden_size)
        self.elem_embed = ElementEmbedding(
            ActionSignatureEmbedding(n_token, n_node_type,
                                     encoder_hidden_size),
            max_arity + 1, encoder_hidden_size, rule_embedding_size)
        self.query_embed = ElementEmbedding(
            EmbeddingWithMask(n_rule, encoder_hidden_size, n_rule),
            max_depth, encoder_hidden_size, encoder_hidden_size
        )
        self.encoder_blocks = [ActionSequenceReaderBlock(
            rule_embedding_size, encoder_hidden_size, tree_conv_kernel_size,
            n_head, dropout, i
        ) for i in range(n_encoder_block)]
        for i, block in enumerate(self.encoder_blocks):
            self.add_module(f"encoder_block_{i}", block)
        self.decoder_blocks = []
        for i in range(n_decoder_block):
            block = DecoderBlock(
                encoder_hidden_size, decoder_hidden_size,
                out_size if i == n_decoder_block - 1 else encoder_hidden_size,
                n_head, dropout
            )
            self.decoder_blocks.append(block)
            self.add_module(f"decoder_block_{i}", block)

    def forward(self,
                nl_query_features: PaddedSequenceWithMask,
                action_queries: PaddedSequenceWithMask,
                previous_actions: PaddedSequenceWithMask,
                previous_action_rules: PaddedSequenceWithMask,
                depthes: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> PaddedSequenceWithMask:
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
        previous_acitons: rnn.PaddedSequenceWithMask
            The previous action sequence.
            The encoded tensor with the shape of
            (len(action_sequence) + 1, 3). Each action will be encoded by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the reference).
            The padding value should be -1.
        previous_action_rules: rnn.PaddedSequenceWithMask
            The rule of previous action sequence.
            The shape of each sequence is
            (action_length, max_arity + 1, 3).
        depthes: torch.Tensor
            The depth of actions. The shape is (L, B) where L is the
            sequence length, B is the batch size.
        adjacency_matrix: torch.Tensor
            The adjacency matrix. The shape is (B, L, L) where B is the
            batch size, L is the sequence length.

        Returns
        -------
        action_features: PaddedSequenceWithMask
            (L_ast, N, out_size) where L_ast is the sequence length,
            N is the batch_size.
        """
        e_action = self.action_embed(previous_actions.data)
        e_rule_action = self.elem_embed(previous_action_rules.data)
        action_features = PaddedSequenceWithMask(e_action, previous_actions.mask)
        for block in self.encoder_blocks:
            action_features, _ = block(action_features, depthes, e_rule_action,
                                       adjacency_matrix)

        q = action_queries.data + \
            (action_queries.data == -1) * (self.n_rule + 1)
        embed = self.query_embed(q)
        input = PaddedSequenceWithMask(embed, action_queries.mask)
        for block in self.decoder_blocks:
            input, _, _ = block(input, nl_query_features, action_features)
        return input
