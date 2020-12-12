from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(Loss, self).__init__()
        self.reduction = reduction
        assert self.reduction == "mean" or self.reduction == "sum" or \
            self.reduction == "none"

    def forward(self, inputs: Environment) -> Environment:
        """
        Parameters
        ----------
        rule_probs: PaddedSequenceWithMask
            The probabilities of apply-rule. The shape is (L_a, B, num_rules).
        token_probs: PaddedSequenceWithMask
            The probabilities of gen-token. The shape is (L_a, B, num_tokens).
        reference_probs: PaddedSequenceWithMask
            The probabilities of reference-token. The shape is
            (L_a, B, reference_length).
        ground_truth_actions: PaddedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the reference).
            The padding value should be -1.
        """
        rule_probs = cast(PaddedSequenceWithMask,
                          inputs["rule_probs"])
        token_probs = cast(PaddedSequenceWithMask,
                           inputs["token_probs"])
        reference_probs = \
            cast(PaddedSequenceWithMask, inputs["reference_probs"])
        ground_truth_actions = cast(
            PaddedSequenceWithMask,
            inputs["ground_truth_actions"])
        L_a, B, num_rules = rule_probs.data.shape
        _, _, num_tokens = token_probs.data.shape
        _, _, reference_length = reference_probs.data.shape

        gt_rule, gt_token, gt_reference = torch.split(
            ground_truth_actions.data, 1, dim=2)  # (L_a, B, 1)
        gt_rule = gt_rule.reshape([L_a, B])  # (L_a, B)
        gt_token = gt_token.reshape([L_a, B])  # (L_a, B)
        gt_reference = gt_reference.reshape([L_a, B])  # (L_a, B)
        # Change padding value
        rule = gt_rule + (gt_rule == -1).long() * (num_rules + 1)
        token = gt_token + (gt_token == -1).long() * (num_tokens + 1)
        reference = gt_reference + \
            (gt_reference == -1).long() * (reference_length) + 1

        device = gt_rule.device

        rule = torch.eye(num_rules + 1,
                         device=device)[gt_rule]  # (L_a, B, num_rules + 1)
        rule = rule[:, :, :-1]  # (L_a, B, num_rules)
        # (L_a, B, num_tokens + 1)
        token = torch.eye(num_tokens + 1,
                          device=device)[gt_token]
        token = token[:, :, :-1]  # (L_a, B, num_tokens)
        # (L_a, B, reference_length + 1)
        reference = torch.eye(reference_length + 1,
                              device=device)[gt_reference]
        # (L_a, B, reference_length)
        reference = reference[:, :, :-1]

        rule_prob_tensor = rule_probs.data * rule  # (L_a, B, num_rules)
        rule_prob_tensor = torch.sum(rule_prob_tensor, dim=2)  # (L_a, B)
        token_prob_tensor = token_probs.data * token  # (L_a, B, num_tokens)
        token_prob_tensor = torch.sum(token_prob_tensor, dim=2)  # (L_a, B)
        reference_prob_tensor = \
            reference_probs.data * reference  # (L_a, B, reference_length)
        reference_prob_tensor = \
            torch.sum(reference_prob_tensor, dim=2)  # (L_a, B)

        prob = rule_prob_tensor + token_prob_tensor + reference_prob_tensor
        prob = prob + (prob < 1e-7).to(rule_prob_tensor.dtype) * \
            1e-7  # avoid zero division

        likelihood = torch.log(prob)  # (L_a, B)
        loss = -likelihood * \
            ground_truth_actions.mask.to(rule_prob_tensor.dtype)  # (L_a, B)
        if self.reduction == "mean":
            inputs["action_sequence_loss"] = torch.mean(
                torch.sum(loss, dim=0))
        elif self.reduction == "sum":
            inputs["action_sequence_loss"] = torch.sum(loss)
        else:
            inputs["action_sequence_loss"] = torch.sum(loss, dim=0)
        return inputs
