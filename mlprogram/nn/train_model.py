import torch.nn as nn
from typing import Dict, Any


class TrainModel(nn.Module):
    def __init__(self,
                 input_reader: nn.Module,  action_sequence_reader: nn.Module,
                 decoder: nn.Module, predictor: nn.Module):
        super(TrainModel, self).__init__()
        self.input_reader = input_reader
        self.action_sequence_reader = action_sequence_reader
        self.decoder = decoder
        self.predictor = predictor

    def forward(self, **inputs: Any) \
            -> Dict[str, Any]:
        """
        Parameters
        ----------
        inputs

        Returns
        -------
        rule_pred: rnn.PackedSequenceWithMask
            The probabilities of apply-rule
        token_pred: PaddedSequenceWithMask
            The probabilities of gen-token
        copy_pred: PaddedSequenceWithMask
            The probabilities of copy-token
        """
        inputs = self.input_reader(**inputs)
        inputs = self.action_sequence_reader(**inputs)
        inputs = self.decoder(**inputs)
        return self.predictor(**inputs)
