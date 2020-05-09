import torch
import torch.nn as nn
from typing import Tuple

from mlprogram.nn import TreeConvolution
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.nn.functional \
    import index_embeddings, position_embeddings, gelu, lne_to_nel, nel_to_lne
from .gating import Gating
from .embedding \
    import ActionEmbedding, ActionSignatureEmbedding, ElementEmbedding


class ActionSequenceReaderBlock(nn.Module):
    def __init__(self,
                 rule_embed_size: int, hidden_size: int,
                 tree_conv_kernel_size: int,
                 n_heads: int, dropout: float, block_idx: int):
        super(ActionSequenceReaderBlock, self).__init__()
        self.block_idx = block_idx
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, dropout)
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


class ActionSequenceReader(nn.Module):
    def __init__(self,
                 rule_num: int, token_num: int, node_type_num: int,
                 max_arity: int, rule_embed_size: int, hidden_size: int,
                 tree_conv_kernel_size: int, n_heads: int, dropout: float,
                 n_blocks: int):
        super(ActionSequenceReader, self).__init__()
        self.blocks = [ActionSequenceReaderBlock(
            rule_embed_size, hidden_size, tree_conv_kernel_size,
            n_heads, dropout, i
        ) for i in range(n_blocks)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"block_{i}", block)
        self.action_embed = ActionEmbedding(rule_num, token_num, hidden_size)
        self.elem_embed = ElementEmbedding(
            ActionSignatureEmbedding(token_num, node_type_num,
                                     hidden_size),
            max_arity + 1, hidden_size, rule_embed_size)

    def forward(self,
                action_sequence: Tuple[PaddedSequenceWithMask,
                                       PaddedSequenceWithMask,
                                       torch.Tensor, torch.Tensor]) -> \
            PaddedSequenceWithMask:
        """
        Parameters
        ----------
        action_sequence
            previous_aciton: rnn.PaddedSequenceWithMask
                The previous action sequence.
                The encoded tensor with the shape of
                (len(action_sequence) + 1, 3). Each action will be encoded by
                the tuple of (ID of the applied rule, ID of the inserted token,
                the index of the word copied from the query).
                The padding value should be -1.
            rule_previous_action: rnn.PaddedSequenceWithMask
                The rule of previous action sequence.
                The shape of each sequence is
                (action_length, max_arity + 1, 3).
            depth: torch.Tensor
                The depth of actions. The shape is (L, B) where L is the
                sequence length, B is the batch size.
            adjacency_matrix: torch.Tensor
                The adjacency matrix. The shape is (B, L, L) where B is the
                batch size, L is the sequence length.

        Returns
        -------
        PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        previous_action, rule_previous_action, depth, adjacency_matrix = \
            action_sequence
        e_action = self.action_embed(previous_action.data)
        e_rule_action = self.elem_embed(rule_previous_action.data)
        input = PaddedSequenceWithMask(e_action, previous_action.mask)
        for block in self.blocks:
            input, _ = block(input, depth, e_rule_action, adjacency_matrix)
        return input
