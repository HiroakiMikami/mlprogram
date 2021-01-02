from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.action_sequence import PreviousActionsEmbedding
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class ElementEmbedding(nn.Module):
    def __init__(self, embed: nn.Module, max_elem_num: int,
                 elemwise_embedding_dim: int,
                 embedding_dim: int):
        super(ElementEmbedding, self).__init__()
        self.embed = embed
        self.elem_to_seq = nn.Conv1d(elemwise_embedding_dim, embedding_dim,
                                     max_elem_num, bias=False)

    def forward(self, input) -> torch.Tensor:
        embed = self.embed(input)
        dims = embed.shape
        embed = embed.reshape(np.prod(dims[:-2]), dims[-2], dims[-1]) \
            .permute(0, 2, 1)
        return self.elem_to_seq(embed).reshape(*dims[:-2], -1)


class ActionSignatureEmbedding(nn.Module):
    def __init__(self, token_num: int, node_type_num: int,
                 embedding_dim: int):
        super(ActionSignatureEmbedding, self).__init__()
        self.token_num = token_num
        self.node_type_embed = EmbeddingWithMask(node_type_num, embedding_dim, -1)
        self.token_embed = EmbeddingWithMask(token_num + 1, embedding_dim, -1)

    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        signature: torch.Tensor
            The shape is (-1, 3)
            The padding value should be -1.
            [:, 0] represent the IDs of the node types, [:, 1]
            represent the token IDs, [:, 2] represent the indexes of the
            queries.

        Returns
        -------
        embed: torch.Tensor
            The shape is (-1, feature_dim).
        """
        dims = signature.shape
        signature = signature.reshape(-1, dims[-1])

        node_type_seq = signature[:, 0]
        token_seq = signature[:, 1]
        reference_seq = (token_seq == -1) * (signature[:, 2] != -1)
        token_seq = token_seq + reference_seq * (self.token_num + 1)

        embed = self.node_type_embed(node_type_seq) + self.token_embed(token_seq)
        return embed.reshape(*dims[:-1], -1)


class NlEmbedding(nn.Module):
    def __init__(self,
                 n_token: int, n_char: int, max_token_length: int,
                 char_embedding_size: int, embedding_size: int):
        super().__init__()
        self.word_embed = EmbeddingWithMask(n_token, embedding_size, -1)
        self.char_embed = ElementEmbedding(
            EmbeddingWithMask(n_char, embedding_size, -1),
            max_token_length, embedding_size, char_embedding_size)

    def forward(self,
                word_nl_query: PaddedSequenceWithMask,
                char_nl_query: PaddedSequenceWithMask
                ) -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        e_word = self.word_embed(word_nl_query.data)
        e_char = self.char_embed(char_nl_query.data)
        return (PaddedSequenceWithMask(e_word, word_nl_query.mask),
                PaddedSequenceWithMask(e_char, char_nl_query.mask))


class QueryEmbedding(nn.Module):
    def __init__(self,
                 n_rule: int,
                 max_depth: int,
                 embedding_size: int):
        super().__init__()
        self.query_embed = ElementEmbedding(
            EmbeddingWithMask(n_rule, embedding_size, -1),
            max_depth, embedding_size, embedding_size
        )

    def forward(self,
                action_queries: PaddedSequenceWithMask,
                ) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        action_queries: PaddedSequenceWithMask
            (L_ast, N, max_depth) where L_ast is the sequence length,
            N is the batch size.
            This tensor encodes the path from the root node to the target node.
            The padding value should be -1.

        Returns
        -------
        action_query_features: PaddedSequenceWithMask
            (L_ast, N, hidden_size) where L_ast is the sequence length,
            N is the batch_size.
        """
        embed = self.query_embed(action_queries.data)
        return PaddedSequenceWithMask(embed, action_queries.mask)


class ActionEmbedding(nn.Module):
    def __init__(self,
                 n_rule: int, n_token: int, n_node_type: int,
                 max_arity: int,
                 rule_embedding_size: int,
                 embedding_size: int):
        super().__init__()
        self.action_embed = PreviousActionsEmbedding(n_rule, n_token, embedding_size)
        self.elem_embed = ElementEmbedding(
            ActionSignatureEmbedding(n_token, n_node_type,
                                     embedding_size),
            max_arity + 1, embedding_size, rule_embedding_size)

    def forward(self,
                previous_actions: PaddedSequenceWithMask,
                previous_action_rules: PaddedSequenceWithMask
                ) -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        """
        Parameters
        ----------
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

        Returns
        -------
        e_action: PaddedSequenceWithMask
            (L_ast, N, embedding_size) where L_ast is the sequence length,
            N is the batch_size.
        e_rule_action: PaddedSequenceWithMask
            (L_ast, N, rule_embedding_size) where L_ast is the sequence length,
            N is the batch_size.
        """
        e_action = self.action_embed(previous_actions)
        e_rule_action = self.elem_embed(previous_action_rules.data)
        return (e_action,
                PaddedSequenceWithMask(e_rule_action, previous_action_rules.mask))
