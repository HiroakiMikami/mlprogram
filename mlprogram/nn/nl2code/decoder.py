from typing import Optional, Tuple

import torch
import torch.nn as nn

from mlprogram.nn.action_sequence import AttentionInput
from mlprogram.nn.action_sequence.lstm_tree_decoder import query_history
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Decoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 query_size: int, hidden_size: int,
                 att_hidden_size: int, dropout: float = 0.0):
        """
        Constructor

        Parameters
        ----------
        input_size: int
            Size of each input vector
        query_size: int
            Size of each query vector
        hidden_size: int
            The number of features in the hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        dropout: float
            The probability of dropout
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        inject_input = AttentionInput(att_hidden_size)
        output_size, self.inject_input = inject_input(
            query_size, input_size + hidden_size, hidden_size, hidden_size
        )
        self.lstm = nn.LSTMCell(output_size, hidden_size)

        nn.init.xavier_uniform_(self.lstm.weight_hh)
        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.xavier_uniform_(self.inject_input.attn[0].weight)
        nn.init.zeros_(self.inject_input.attn[0].bias)
        nn.init.xavier_uniform_(self.inject_input.attn[-1].weight)
        nn.init.zeros_(self.inject_input.attn[-1].bias)

    def forward(self,
                nl_query_features: PaddedSequenceWithMask,
                actions: PaddedSequenceWithMask,
                action_features: PaddedSequenceWithMask,
                history: Optional[torch.Tensor],
                hidden_state: Optional[torch.Tensor],
                state: Optional[torch.Tensor]
                ) -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        nl_query_features: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).

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
        action_contexts: PaddedSequenceWithMask
            Packed sequence containing the context vectors.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        hidden_state: torch.Tensor
            The tuple of the next hidden state. The shape is (B, hidden_size)
        state: torch.Tensor
            The tuple of the next state. The shape is (B, hidden_size)
        """
        _, _, query_size = nl_query_features.data.shape
        L_a, B, _ = actions.data.shape
        h_n = hidden_state
        c_n = state

        _, _, parent_indexes = torch.split(actions.data, 1, dim=2)  # (L_a, B, 1)

        if history is None:
            history = torch.zeros(1, B, self.hidden_size,
                                  device=nl_query_features.data.device)
        if h_n is None:
            h_n = torch.zeros(B, self.hidden_size,
                              device=nl_query_features.data.device)
        if c_n is None:
            c_n = torch.zeros(B, self.hidden_size,
                              device=nl_query_features.data.device)
        s = (h_n, c_n)
        hs = []
        cs = []
        for d, parent_index in zip(action_features.data, parent_indexes):
            x = nn.functional.dropout(d, p=self.dropout)
            h = nn.functional.dropout(s[0], p=self.dropout)

            # Parent_history
            h_parent = query_history(history, parent_index)
            x = torch.cat([x, h_parent], dim=1)

            input = self.inject_input(nl_query_features, x, s[0], s[1])
            ctx = input[:, :query_size]
            h1, c1 = self.lstm(input, (h, s[1]))
            hs.append(h1)
            cs.append(ctx)
            s = (h1, c1)
            history = torch.cat([history,
                                 s[0].reshape(1, *s[0].shape)], dim=0)
        hs = torch.stack(hs)
        cs = torch.stack(cs)
        h_n, c_n = s

        return (rnn.PaddedSequenceWithMask(hs, action_features.mask),
                rnn.PaddedSequenceWithMask(cs, action_features.mask),
                history,
                h_n,
                c_n)
