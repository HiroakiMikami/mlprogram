import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn import TreeConvolution
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask
from nl2prog.nn.functional \
    import index_embeddings, position_embeddings, gelu, lne_to_nel, nel_to_lne
from .gating import Gating


class ASTReaderBlock(nn.Module):
    def __init__(self,
                 rule_embed_size: int, hidden_size: int,
                 tree_conv_kernel_size: int,
                 n_heads: int, dropout: float, block_idx: int):
        super(ASTReaderBlock, self).__init__()
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
        h_in = input.data
        h = h_in + \
            index_embeddings(h_in, self.block_idx) + \
            position_embeddings(depth, self.block_idx, hidden_size)
        attn_mask = \
            torch.nn.Transformer.generate_square_subsequent_mask(None, L)
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


class ASTReader(nn.Module):
    def __init__(self,
                 rule_embed_size: int, hidden_size: int,
                 tree_conv_kernel_size: int,
                 n_heads: int, dropout: float, n_blocks: int):
        super(ASTReader, self).__init__()
        self.blocks = [ASTReaderBlock(
            rule_embed_size, hidden_size, tree_conv_kernel_size,
            n_heads, dropout, i
        ) for i in range(n_blocks)]
        for i, block in enumerate(self.blocks):
            self.add_module("block_{}".format(i), block)

    def forward(self,
                seq_embed: PaddedSequenceWithMask,
                depth: torch.Tensor,
                rule_embed: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> \
            PaddedSequenceWithMask:
        """
        Parameters
        ----------
        seq_embed: PaddedSequenceWithMask
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
        PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        input = seq_embed
        for block in self.blocks:
            input, _ = block(input, depth, rule_embed, adjacency_matrix)
        return input
