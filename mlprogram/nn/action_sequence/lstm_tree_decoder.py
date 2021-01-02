from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from mlprogram.nn.action_sequence.lstm_decoder import InjectInput
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


def query_history(history: torch.FloatTensor, index: torch.LongTensor) \
        -> torch.FloatTensor:
    """
    Return the hidden states of the specified indexes

    Parameters
    ----------
    history : torch.FloatTensor
        The sequence of history. The shape is (L, B, F)
    index : torch.LongTensor
        The indexes of history to query. The shape is (B,)

    Returns
    -------
    torch.FloatTensor
        The hidden states of the specified indexes. The shape is (B, F)
    """

    device = history.device
    L = history.shape[0]
    B = history.shape[1]
    index_onehot = torch.eye(L, device=device)[index]  # (B, L)
    index_onehot = index_onehot.reshape((B, 1, L))  # (B, 1, L)
    h = torch.matmul(index_onehot, history.permute([1, 0, 2]))  # (B, 1, *)
    h = h.reshape((B, *history.shape[2:]))  # (B, *)
    return h


class LSTMTreeDecoder(nn.Module):
    def __init__(self,
                 inject_input: InjectInput,
                 input_feature_size: int, action_feature_size: int,
                 output_feature_size: int, dropout: float = 0.0):
        """
        Constructor

        Parameters
        ----------
        input_feature_size: int
            Size of each input vector
        action_feature_size: int
            Size of each ActionSequence vector
        output_feature_size: int
            The number of features in the hidden state
        dropout: float
            The probability of dropout
        """
        super().__init__()
        self.output_feature_size = output_feature_size
        self.dropout = dropout
        output_size, self.inject_input = inject_input(
            input_feature_size,
            action_feature_size + output_feature_size,
            output_feature_size,
            output_feature_size)
        self.lstm = nn.LSTMCell(output_size, output_feature_size)

    def forward(self,
                input_feature: Any,
                actions: PaddedSequenceWithMask,
                action_features: PaddedSequenceWithMask,
                history: Optional[torch.Tensor],
                hidden_state: Optional[torch.Tensor],
                state: Optional[torch.Tensor]
                ) -> Tuple[PaddedSequenceWithMask, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_feature: Any
        actions: rnn.PackedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the node types, ID of the parent-action's
            rule, the index of the parent action).
            The padding value should be -1.
        action_featuress: rnn.PackedSequenceWithMask
            The feature tensor of ActionSequence
        history: torch.FloatTensor
            The list of LSTM states. The shape is (B, L_h, hidden_size)
        hidden_state: torch.Tensor
            The LSTM initial hidden state. The shape is (B, hidden_size)
        state: torch.Tensor
            The LSTM initial state. The shape is (B, hidden_size)

        Returns
        -------
        action_features: PaddedSequenceWithMask
            Packed sequence containing the output hidden states.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        hidden_state: torch.Tensor
            The tuple of the next hidden state. The shape is (B, hidden_size)
        state: torch.Tensor
            The tuple of the next state. The shape is (B, hidden_size)
        """
        h_n = hidden_state
        c_n = state

        L_a, B, _ = action_features.data.shape
        _, _, parent_indexes = torch.split(actions.data, 1, dim=2)  # (L_a, B, 1)

        if history is None:
            history = torch.zeros(1, B, self.output_feature_size,
                                  device=action_features.data.device)
        if h_n is None:
            h_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        if c_n is None:
            c_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        s = (h_n, c_n)
        hs = []
        for d, parent_index in zip(action_features.data, parent_indexes):
            x = nn.functional.dropout(d, p=self.dropout)
            h = nn.functional.dropout(s[0], p=self.dropout)

            # Parent_history
            h_parent = query_history(history, parent_index)
            x = torch.cat([x, h_parent], dim=1)

            input = self.inject_input(input_feature, x, s[0], s[1])
            h1, c1 = self.lstm(input, (h, s[1]))
            hs.append(h1)
            s = (h1, c1)
            history = torch.cat([history,
                                 s[0].reshape(1, *s[0].shape)], dim=0)
        hs = torch.stack(hs)

        return rnn.PaddedSequenceWithMask(hs, action_features.mask), history, h1, c1
