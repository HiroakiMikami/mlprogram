import torch
import torch.nn as nn
from typing import Dict, Optional, cast

from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.nn import EmbeddingWithMask
from mlprogram.datatypes import Tensor


class ActionSequenceReader(nn.Module):
    def __init__(self, num_rules: int, num_tokens: int, num_node_types: int,
                 node_type_embedding_size: int, embedding_size: int):
        """
        Constructor

        Parameters
        ----------
        num_rules: int
            The number of rules
        num_tokens: int
            The number of tokens
        num_node_types: int
            The number of node types
        node_type_embedding_size: int
            Size of each node-type embedding vector
        embedding_size: int
            Size of each embedding vector
        """
        super(ActionSequenceReader, self).__init__()
        self._num_rules = num_rules
        self._num_tokens = num_tokens
        self._num_node_types = num_node_types
        self._rule_embed = EmbeddingWithMask(
            num_rules, embedding_size, num_rules)
        self._token_embed = EmbeddingWithMask(
            num_tokens, embedding_size, num_tokens)
        self._node_type_embed = EmbeddingWithMask(
            num_node_types, node_type_embedding_size, num_node_types)
        nn.init.normal_(self._rule_embed.weight, std=0.01)
        nn.init.normal_(self._token_embed.weight, std=0.01)
        nn.init.normal_(self._node_type_embed.weight, std=0.01)

    def forward(self, **inputs: Optional[Tensor]) \
            -> Dict[str, Optional[Tensor]]:
        """
        Parameters
        ----------
        actions: rnn.PackedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the node types, ID of the parent-action's
            rule, the index of the parent action).
            The padding value should be -1.
        previous_actions: rnn.PackedSequenceWithMask
            The input sequence of previous action. Each action is
            represented by the tuple of (ID of the applied rule, ID of
            the inserted token, the index of the word copied from
            the query).
            The padding value should be -1.

        Returns
        -------
        action_features: rnn.PackedSequenceWithMask
            The embeddings of action sequence
        parent_indexes: PaddedSequenceWithMask
            The indexes of the parent nodes
        """
        actions = cast(PaddedSequenceWithMask, inputs["actions"])
        previous_actions = cast(PaddedSequenceWithMask,
                                inputs["previous_actions"])
        L_a, B, _ = actions.data.shape
        node_types, parent_rule, parent_index = torch.split(
            actions.data, 1, dim=2)  # (L_a, B, 1)
        prev_rules, prev_tokens, _ = torch.split(
            previous_actions.data, 1, dim=2)  # (L_a, B, 1)

        # Change the padding value
        node_types = node_types + \
            (node_types == -1).long() * (self._num_node_types + 1)
        node_types = node_types.reshape([L_a, B])
        parent_rule = parent_rule + \
            (parent_rule == -1).long() * (self._num_rules + 1)
        parent_rule = parent_rule.reshape([L_a, B])
        prev_rules = prev_rules + \
            (prev_rules == -1).long() * (self._num_rules + 1)
        prev_rules = prev_rules.reshape([L_a, B])
        prev_tokens = prev_tokens + \
            (prev_tokens == -1).long() * (self._num_tokens + 1)
        prev_tokens = prev_tokens.reshape([L_a, B])

        # Embed previous actions
        prev_action_embed = self._rule_embed(prev_rules) + \
            self._token_embed(prev_tokens)  # (L_a, B, embedding_size)
        # Embed action
        node_type_embed = self._node_type_embed(
            node_types)  # (L_a, B, node_type_embedding_size)
        parent_rule_embed = self._rule_embed(
            parent_rule)  # (L_a, B, embedding_size)

        # Decode embeddings
        feature = torch.cat(
            [prev_action_embed, node_type_embed, parent_rule_embed],
            dim=2)  # (L_a, B, input_size)
        inputs["action_features"] = \
            PaddedSequenceWithMask(feature, actions.mask)
        inputs["parent_indexes"] = \
            PaddedSequenceWithMask(parent_index, actions.mask)
        return inputs
