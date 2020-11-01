from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class ActionSequenceReader(nn.Module):
    def __init__(self, n_rule: int, n_token: int, hidden_size: int):
        super().__init__()
        self.n_rule = n_rule
        self.n_token = n_token
        self._rule_embed = EmbeddingWithMask(
            n_rule, hidden_size, n_rule)
        self._token_embed = EmbeddingWithMask(
            n_token, hidden_size, n_token)

    def forward(self, inputs: Environment) -> Environment:
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

        Returns
        -------
        action_features: PaddedSequenceWithMask
            (L, N, hidden_size) where L is the sequence length,
            N is the batch size.
        """
        previous_actions = cast(PaddedSequenceWithMask,
                                inputs.states["previous_actions"])
        L_a, B, _ = previous_actions.data.shape
        prev_rules, prev_tokens, _ = torch.split(
            previous_actions.data, 1, dim=2)  # (L_a, B, 1)

        # Change the padding value
        prev_rules = torch.where(
            prev_rules == -1,
            torch.full_like(prev_rules, self.n_rule),
            prev_rules
        )
        prev_rules = prev_rules.reshape([L_a, B])
        prev_tokens = torch.where(
            prev_tokens == -1,
            torch.full_like(prev_tokens, self.n_token),
            prev_tokens
        )
        prev_tokens = prev_tokens.reshape([L_a, B])

        # Embed previous actions
        feature = self._rule_embed(prev_rules) + \
            self._token_embed(prev_tokens)  # (L_a, B, embedding_size)
        inputs.states["action_features"] = \
            PaddedSequenceWithMask(feature, previous_actions.mask)
        return inputs
