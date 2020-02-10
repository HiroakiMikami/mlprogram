import torch
from typing import Tuple
import torch.nn as nn

from nl2prog.nn import EmbeddingWithMask


class QueryEmbedding(nn.Module):
    def __init__(self, vocab_num: int, character_vocab_num: int,
                 max_seq_num: int,
                 embedding_dim: int, elem_embedding_dim: int,
                 elem_feature_dim: int):
        super(QueryEmbedding, self).__init__()
        self.character_vocab_num = character_vocab_num
        self.embed = nn.Embedding(vocab_num, embedding_dim)
        self.elem_embed = EmbeddingWithMask(character_vocab_num,
                                            elem_embedding_dim,
                                            character_vocab_num)
        self.elem_to_seq = nn.Conv1d(elem_embedding_dim, elem_feature_dim,
                                     max_seq_num, bias=False)

    def forward(self, sequence: torch.Tensor, char_seq: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sequence: torch.Tensor
            The shape is (L, N). where L is the sequence length and
            N is the batch size.
        char_seq: torch.Tensor
            The shape is (L, N, max_seq_len). where L is the sequence length
            and N is the batch size. The padding value should be -1.

        Returns
        -------
        seq_embed: torch.Tensor
            The shape is (L, N, embedding_dim). where L is the sequence length
            and N is the batch size.
        char_seq_embed: torch.Tensor
            The shape is (L, N, elem_feature_dim). where L is the sequence
            length and N is the batch size.
        """
        L, N = sequence.shape[:2]
        n = char_seq.shape[2]
        seq_embed = self.embed(sequence)
        char_seq = char_seq + (char_seq == -1) * (self.character_vocab_num + 1)
        char_seq_embed = self.elem_embed(char_seq)
        char_seq_embed = self.elem_to_seq(
            char_seq_embed.reshape(L * N, n, -1).permute(0, 2, 1)
        ).reshape(L, N, -1)
        return seq_embed, char_seq_embed


class RuleEmbedding(nn.Module):
    def __init__(self, rule_num: int, token_num: int, node_type_num: int,
                 max_arity: int,
                 embedding_dim: int, elem_embedding_dim: int,
                 elem_feature_dim: int):
        super(RuleEmbedding, self).__init__()
        self.rule_num = rule_num
        self.token_num = token_num
        self.node_type_num = node_type_num
        self.rule_embed = EmbeddingWithMask(rule_num, embedding_dim, rule_num)
        self.token_embed = EmbeddingWithMask(token_num + 1, embedding_dim,
                                             token_num + 1)
        self.elem_node_type_embed = \
            EmbeddingWithMask(node_type_num, elem_embedding_dim, node_type_num)
        self.elem_token_embed = \
            EmbeddingWithMask(token_num + 1, elem_embedding_dim, token_num + 1)
        self.elem_to_seq = nn.Conv1d(elem_embedding_dim, elem_feature_dim,
                                     max_arity + 1, bias=False)

    def forward(self, sequence: torch.Tensor, elem_seq: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sequence: torch.Tensor
            The shape is (L, N, 3). where L is the sequence length and
            N is the batch size. The padding value should be -1.
            [:, :, 0] represent the rule IDs, [:, :, 1] represent the token
            IDs, [:, :, 2] represent the indexes of the queries.
        elem_seq: torch.Tensor
            The shape is (L, N, max_arity + 1, 3). where L is the
            sequence length and N is the batch size.
            The padding value should be -1.
            [:, :, :, 0] represent the IDs of the node types, [:, :, :, 1]
            represent the token IDs, [:, :, :, 2] represent the indexes of the
            queries.

        Returns
        -------
        seq_embed: torch.Tensor
            The shape is (L, N, embedding_dim). where L is the sequence length
            and N is the batch size.
        elem_seq_embed: torch.Tensor
            The shape is (L, N, elem_feature_dim). where L is the sequence
            length and N is the batch size.
        """
        L, N = sequence.shape[:2]
        n = elem_seq.shape[2]

        rule_seq = sequence[:, :, 0]
        rule_seq = rule_seq + (rule_seq == -1) * (self.rule_num + 1)

        token_seq = sequence[:, :, 1]
        copy_seq = (token_seq == -1) * (sequence[:, :, 2] != -1)
        # copy_seq => self.token_num
        token_seq = token_seq + copy_seq * (self.token_num + 1)
        token_seq = token_seq + (token_seq == -1) * (self.token_num + 2)

        seq_embed = self.rule_embed(rule_seq) + self.token_embed(token_seq)

        elem_node_type_seq = elem_seq[:, :, :, 0]
        elem_node_type_seq = \
            elem_node_type_seq + \
            (elem_node_type_seq == -1) * (self.node_type_num + 1)
        elem_token_seq = elem_seq[:, :, :, 1]
        elem_copy_seq = (elem_token_seq == -1) * (elem_seq[:, :, :, 2] != -1)
        elem_token_seq = elem_token_seq + elem_copy_seq * (self.token_num + 2)
        elem_token_seq = \
            elem_token_seq + (elem_token_seq == -1) * (self.token_num + 2)

        elem_seq_emebed = \
            self.elem_node_type_embed(elem_node_type_seq) + \
            self.elem_token_embed(elem_token_seq)
        elem_seq_embed = self.elem_to_seq(
            elem_seq_emebed.reshape(L * N, n, -1).permute(0, 2, 1)
        ).reshape(L, N, -1)
        return seq_embed, elem_seq_embed
