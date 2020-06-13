import torch
import torch.nn as nn
from typing import Any, cast

from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class NL2ProgLoss(nn.Module):
    def __init__(self):
        super(NL2ProgLoss, self).__init__()

    def forward(self, **inputs: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        rule_probs: PaddedSequenceWithMask
            The probabilities of apply-rule. The shape is (L_a, B, num_rules).
        token_probs: PaddedSequenceWithMask
            The probabilities of gen-token. The shape is (L_a, B, num_tokens).
        copy_probs: PaddedSequenceWithMask
            The probabilities of copy-token. The shape is
            (L_a, B, query_length).
        ground_truth_actions: PaddedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).
            The padding value should be -1.
        """
        rule_probs = cast(PaddedSequenceWithMask, inputs["rule_probs"])
        token_probs = cast(PaddedSequenceWithMask, inputs["token_probs"])
        copy_probs = cast(PaddedSequenceWithMask, inputs["copy_probs"])
        ground_truth_actions = cast(PaddedSequenceWithMask,
                                    inputs["ground_truth_actions"])
        L_a, B, num_rules = rule_probs.data.shape
        _, _, num_tokens = token_probs.data.shape
        _, _, query_length = copy_probs.data.shape

        gt_rule, gt_token, gt_copy = torch.split(
            ground_truth_actions.data, 1, dim=2)  # (L_a, B, 1)
        gt_rule = gt_rule.reshape([L_a, B])  # (L_a, B)
        gt_token = gt_token.reshape([L_a, B])  # (L_a, B)
        gt_copy = gt_copy.reshape([L_a, B])  # (L_a, B)
        # Change padding value
        rule = gt_rule + (gt_rule == -1).long() * (num_rules + 1)
        token = gt_token + (gt_token == -1).long() * (num_tokens + 1)
        copy = gt_copy + (gt_copy == -1).long() * (query_length) + 1

        device = gt_rule.device

        rule = torch.eye(num_rules + 1,
                         device=device)[gt_rule]  # (L_a, B, num_rules + 1)
        rule = rule[:, :, :-1]  # (L_a, B, num_rules)
        # (L_a, B, num_tokens + 1)
        token = torch.eye(num_tokens + 1,
                          device=device)[gt_token]
        token = token[:, :, :-1]  # (L_a, B, num_tokens)
        # (L_a, B, query_length + 1)
        copy = torch.eye(query_length + 1,
                         device=device)[gt_copy]
        # (L_a, B, query_length)
        copy = copy[:, :, :-1]

        rule_prob_tensor = rule_probs.data * rule  # (L_a, B, num_rules)
        rule_prob_tensor = torch.sum(rule_prob_tensor, dim=2)  # (L_a, B)
        token_prob_tensor = token_probs.data * token  # (L_a, B, num_tokens)
        token_prob_tensor = torch.sum(token_prob_tensor, dim=2)  # (L_a, B)
        copy_prob_tensor = copy_probs.data * copy  # (L_a, B, query_length)
        copy_prob_tensor = torch.sum(copy_prob_tensor, dim=2)  # (L_a, B)

        prob = rule_prob_tensor + token_prob_tensor + copy_prob_tensor
        prob = prob + (prob < 1e-7).to(rule_prob_tensor.dtype) * \
            1e-7  # avoid zero division

        likelihood = torch.log(prob)  # (L_a, B)
        loss = -likelihood * \
            ground_truth_actions.mask.to(rule_prob_tensor.dtype)  # (L_a, B)
        return torch.mean(torch.sum(loss, dim=0))
