from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn import PointerNet
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Predictor(nn.Module):
    def __init__(self, feature_size: int, reference_feature_size: int,
                 rule_size: int, token_size: int, hidden_size: int):
        super(Predictor, self).__init__()
        self.select = nn.Linear(feature_size, 3)
        self.rule = nn.Linear(feature_size, rule_size)
        self.token = nn.Linear(feature_size, token_size)
        self.reference = PointerNet(feature_size, reference_feature_size,
                                    hidden_size)

    def forward(self, inputs: Environment) -> Environment:
        """
        Parameters
        ----------
        reference_features: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        action_features: PaddedSequenceWithMask
            (L_ast, N, feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        rule_probs: PaddedSequenceWithMask
            (L_ast, N, rule_size) where L_ast is the sequence length,
            N is the batch_size.
        token_probs: PaddedSequenceWithMask
           (L_ast, N, token_size) where L_ast is the sequence length,
            N is the batch_size.
        reference_probs: PaddedSequenceWithMask
            (L_ast, N, L_nl) where L_ast is the sequence length,
            N is the batch_size.
        """
        reference_features = cast(PaddedSequenceWithMask,
                                  inputs.states["reference_features"])
        action_features = cast(PaddedSequenceWithMask,
                               inputs.states["action_features"])
        rule_pred = self.rule(action_features.data)
        rule_prob = torch.softmax(rule_pred, dim=2)

        token_pred = self.token(action_features.data)
        token_prob = torch.softmax(token_pred, dim=2)

        select = self.select(action_features.data)
        select_prob = torch.softmax(select, dim=2)

        reference_log_prob = \
            self.reference(action_features.data, reference_features)
        reference_prob = torch.exp(reference_log_prob)

        rule_log_prob = select_prob[:, :, 0:1] * rule_prob
        token_log_prob = select_prob[:, :, 1:2] * token_prob
        reference_log_prob = select_prob[:, :, 2:3] * reference_prob
        if self.training:
            inputs.outputs["rule_probs"] = \
                PaddedSequenceWithMask(rule_log_prob, action_features.mask)
            inputs.outputs["token_probs"] = \
                PaddedSequenceWithMask(token_log_prob, action_features.mask)
            inputs.outputs["reference_probs"] = \
                PaddedSequenceWithMask(reference_log_prob,
                                       action_features.mask)
        else:
            inputs.outputs["rule_probs"] = rule_log_prob[-1, :, :]
            inputs.outputs["token_probs"] = token_log_prob[-1, :, :]
            inputs.outputs["reference_probs"] = reference_log_prob[-1, :, :]

        return inputs
