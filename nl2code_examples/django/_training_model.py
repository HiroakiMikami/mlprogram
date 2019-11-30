import torch
import torch.nn as nn
from typing import Tuple
import nl2code.nn as nnn
import nl2code.nn.utils.rnn as rnn
from nl2code_examples.django import Encoder, DatasetEncoder


class TrainingModel(nn.Module):
    def __init__(self, encoder: DatasetEncoder,
                 embedding_dim: int, node_type_embedding_dim: int,
                 lstm_state_size: int, hidden_state_size: int,
                 dropout: float):
        super(TrainingModel, self).__init__()
        self.lstm_state_size = lstm_state_size
        self.encoder = Encoder(encoder.annotation_encoder.vocab_size,
                               embedding_dim, lstm_state_size,
                               dropout=dropout)
        self.predictor = nnn.Predictor(
            encoder.action_sequence_encoder._rule_encoder.vocab_size,
            encoder.action_sequence_encoder._token_encoder.vocab_size,
            encoder.action_sequence_encoder._node_type_encoder.vocab_size,
            node_type_embedding_dim, embedding_dim,
            lstm_state_size, lstm_state_size, hidden_state_size, dropout
        )

    def forward(self, query: rnn.PaddedSequenceWithMask,
                action: rnn.PaddedSequenceWithMask,
                previous_action: rnn.PaddedSequenceWithMask
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                           torch.FloatTensor, torch.FloatTensor,
                           Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Parameters
        ----------
        query: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length).
        aciton: rnn.PaddedSequenceWithMask
            The action sequence.
            The encoded tensor with the shape of
            (len(action_sequence) + 1, 3). Each action will be encoded by
            the tuple of (ID of the node types, ID of the parent-action's rule,
            the index of the parent action).
        previous_action: rnn.PaddedSequenceWithMask
            The previous action sequence.
            The encoded tensor with the shape of
            (len(action_sequence) + 1, 3). Each action will be encoded by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).

        Returns
        -------
        rule_pred: rnn.PackedSequenceWithMask
            The probabilities of apply-rule
        token_pred: PaddedSequenceWithMask
            The probabilities of gen-token
        copy_pred: PaddedSequenceWithMask
            The probabilities of copy-token
        history: torch.FloatTensor
            The list of LSTM states.
            The shape is (L_h + 1, B, lstm_hidden_size)
        (h_n, c_n) : Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the next states. The shape of each tensor is
            (B, lstm_state_size)
        """

        # Encode query
        query_embed = self.encoder(query)  # PackedSequence
        B = query_embed.data.shape[1]

        # Decode action sequence
        history = torch.zeros(0, B, self.lstm_state_size,
                              device=query_embed.data.device)
        h_0 = torch.zeros(B, self.lstm_state_size,
                          device=query_embed.data.device)
        c_0 = torch.zeros(B, self.lstm_state_size,
                          device=query_embed.data.device)
        return self.predictor(query_embed, action, previous_action,
                              history, (h_0, c_0))
