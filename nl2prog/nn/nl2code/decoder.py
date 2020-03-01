import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn.utils import rnn


def query_history(history: torch.FloatTensor, index: torch.LongTensor):
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
            input_size+query_size+hidden_size, hidden_size)
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
        super(Decoder, self).__init__()
        self._cell = DecoderCell(
            query_size, input_size, hidden_size, att_hidden_size,
            dropout=dropout)

    def forward(self,
                query: None,
                nl_feature: rnn.PaddedSequenceWithMask,
                other_feature: None,
                ast_feature: Tuple[rnn.PaddedSequenceWithMask,
                                   rnn.PaddedSequenceWithMask],
                state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> Tuple[Tuple[rnn.PaddedSequenceWithMask,
                                 rnn.PaddedSequenceWithMask],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        query
        nl_feature: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        other_feature
        ast_feature:
            input: rnn.PackedSequenceWithMask
                The input sequence of feature vectors.
                The shape of input is (L_a, B, input_size)
            parent_index: rnn.PackedSequenceWithMask
                The sequence of parent action indexes.
                If index is 0, it means that the action is root action.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (B, L_h, hidden_size)
        (h_0, c_0): Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the LSTM initial states. The shape of each tensor is
            (B, hidden_size)

        Returns
        -------
        feature:
            output: PaddedSequenceWithMask
                Packed sequence containing the output hidden states.
            contexts: PaddedSequenceWithMask
                Packed sequence containing the context vectors.
        state:
            history: torch.FloatTensor
                The list of LSTM states. The shape is (L_h, B, hidden_size)
            (h_n, c_n) : Tuple[torch.FloatTensor, torch.FloatTensor]
                The tuple of the next states. The shape of each tensor is
                (B, hidden_size)
        """
        input, parent_index = ast_feature
        history, h_n, c_n = state
        state = (h_n, c_n)
        hs = []
        cs = []
        for d, i in zip(torch.split(input.data, 1, dim=0),
                        torch.split(parent_index.data, 1, dim=0)):
            ctx, state = self._cell(nl_feature, d.reshape(
                d.shape[1:]), i, history, state)
            hs.append(state[0])
            cs.append(ctx)
            history = torch.cat([history,
                                 state[0].reshape(1, *state[0].shape)], dim=0)
        hs = torch.stack(hs)
        cs = torch.stack(cs)
        h_n, c_n = state

        return ((rnn.PaddedSequenceWithMask(hs, input.mask),
                 rnn.PaddedSequenceWithMask(cs, input.mask)),
                (history, h_n, c_n))
