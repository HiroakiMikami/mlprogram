import torch.nn as nn
from typing import Tuple
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class TrainModel(nn.Module):
    def __init__(self,
                 input_reader: nn.Module,  action_sequence_reader: nn.Module,
                 decoder: nn.Module, predictor: nn.Module):
        super(TrainModel, self).__init__()
        self.input_reader = input_reader
        self.action_sequence_reader = action_sequence_reader
        self.decoder = decoder
        self.predictor = predictor

    def forward(self, input, action_sequence, query) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask,
                     PaddedSequenceWithMask]:
        """
        Parameters
        ----------
        input
        action_sequence
        query

        Returns
        -------
        rule_pred: rnn.PackedSequenceWithMask
            The probabilities of apply-rule
        token_pred: PaddedSequenceWithMask
            The probabilities of gen-token
        copy_pred: PaddedSequenceWithMask
            The probabilities of copy-token
        """
        nl_feature = self.input_reader(input)
        ast_feature = self.action_sequence_reader(action_sequence)
        feature, _ = self.decoder(query, nl_feature, ast_feature)
        return self.predictor(nl_feature, feature)
