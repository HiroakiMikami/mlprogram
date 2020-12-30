from typing import Optional, Tuple

import torch
import torch.nn as nn

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


class DecoderCell(nn.Module):
    """
    One cell of the decoder

    Notes
    -----
    The LSTM cell initialization and dropout are slightly different from
    the original implementation.
    """

    def __init__(self, query_size: int, input_size: int, hidden_size: int,
                 att_hidden_size: int, dropout: float = 0.0):
        """
        Constructor

        Parameters
        ----------
        query_size: int
            Size of each query vector
        input_size: int
            Size of each input vector
        hidden_size: int
            The number of features in the hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        dropout: float
            The probability of dropout
        """
        super(DecoderCell, self).__init__()
        self._lstm_cell = nn.LSTMCell(
            input_size + query_size + hidden_size,
            hidden_size
        )
        self._dropout_in = nn.Dropout(dropout)
        self._dropout_h = nn.Dropout(dropout)
        self._attention_layer1 = nn.Linear(
            query_size + hidden_size, att_hidden_size)
        self._attention_layer2 = nn.Linear(att_hidden_size, 1)

        nn.init.xavier_uniform_(self._lstm_cell.weight_hh)
        nn.init.xavier_uniform_(self._lstm_cell.weight_ih)
        nn.init.zeros_(self._lstm_cell.bias_hh)
        nn.init.zeros_(self._lstm_cell.bias_ih)
        nn.init.xavier_uniform_(self._attention_layer1.weight)
        nn.init.zeros_(self._attention_layer1.bias)
        nn.init.xavier_uniform_(self._attention_layer2.weight)
        nn.init.zeros_(self._attention_layer2.bias)

    def forward(self,
                query: rnn.PaddedSequenceWithMask,
                input: torch.FloatTensor,
                parent_index: torch.LongTensor, history: torch.FloatTensor,
                state: Tuple[torch.FloatTensor, torch.FloatTensor]
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Parameters
        ----------
        query: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        input: torch.FloatTensor
            The input feature vector. The shape is (B, input_size)
        parent_index: torch.LongTensor
            The indexes of the parent actions. The shape is (B).
            If index is -1, it means that the action is root action.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        (h_0, c_0): Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the LSTM initial states. The shape of each tensor is
            (B, hidden_size)

        Returns
        -------
        ctx_vec : torch.FloatTensor
            The context vector of this cell. The shape is (B, query_size)
        (h_1, c_1) : Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the next states. The shape of each tensor is
            (B, hidden_size)
        """
        h_0, c_0 = state
        L_q, B, query_size = query.data.shape
        _, hidden_size = h_0.shape
        L_h, _, _ = history.shape

        # Context
        h_context = h_0.reshape([1, B, hidden_size])
        h_context = h_context.expand(L_q, B, hidden_size)
        # (L_q, B, query_size + hidden_size)
        att = torch.cat([h_context, query.data], dim=2)
        # (L_q, B, att_hidden_size)
        att_hidden = torch.tanh(self._attention_layer1(att))
        att_raw = self._attention_layer2(att_hidden)  # (L_q, B, 1)
        att_raw = att_raw.reshape([L_q, B])  # (L_q, B)
        ctx_att = torch.exp(att_raw -
                            torch.max(att_raw, dim=0, keepdim=True).values
                            )  # (L_q, B)
        ctx_att = ctx_att * query.mask.to(input.dtype)  # (L_q, B)
        ctx_att = ctx_att / torch.sum(ctx_att, dim=0, keepdim=True)  # (L_q, B)
        ctx_att = ctx_att.reshape([L_q, B, 1])  # (L_q, B, 1)
        ctx_vec = torch.sum(query.data * ctx_att, dim=0)  # (B, query_size)

        # Parent_history
        h_parent = query_history(history, parent_index)

        # dropout
        x = self._dropout_in(input)  # (B, input_size)
        h_0 = self._dropout_h(h_0)  # (B, hidden_size)

        # (B, input_size+query_size+input_size)
        x = torch.cat([x, ctx_vec, h_parent], dim=1)
        return ctx_vec, self._lstm_cell(x, (h_0, c_0))


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
        self._cell = DecoderCell(
            query_size, input_size, hidden_size, att_hidden_size,
            dropout=dropout)

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
        L_a, B, _ = actions.data.shape
        h_n = hidden_state
        c_n = state

        _, _, parent_index = torch.split(actions.data, 1, dim=2)  # (L_a, B, 1)

        parent_indexes = PaddedSequenceWithMask(parent_index, actions.mask)

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
        for d, i in zip(torch.split(action_features.data, 1, dim=0),
                        torch.split(parent_indexes.data, 1, dim=0)):
            ctx, s = self._cell(nl_query_features, d.reshape(
                d.shape[1:]), i, history, s)
            hs.append(s[0])
            cs.append(ctx)
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
