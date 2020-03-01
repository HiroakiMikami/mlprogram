import torch
import torch.nn as nn
from torchnlp.encoders import LabelEncoder
from typing import Tuple
from nl2prog.nn.utils import rnn
from nl2prog.nn.nl2code \
    import NLReader, ActionSequenceReader, Decoder, Predictor
from nl2prog.encoders import ActionSequenceEncoder


class TrainModel(nn.Module):
    def __init__(self, query_encoder: LabelEncoder,
                 action_sequence_encoder: ActionSequenceEncoder,
                 embedding_dim: int, node_type_embedding_dim: int,
                 lstm_state_size: int, hidden_state_size: int,
                 dropout: float):
        super(TrainModel, self).__init__()
        self.lstm_state_size = lstm_state_size
        self.nl_reader = NLReader(query_encoder.vocab_size,
                                  embedding_dim, lstm_state_size,
                                  dropout=dropout)
        self.ast_reader = \
            ActionSequenceReader(
                action_sequence_encoder._rule_encoder.vocab_size,
                action_sequence_encoder._token_encoder.vocab_size,
                action_sequence_encoder._node_type_encoder.vocab_size,
                node_type_embedding_dim, embedding_dim)
        self.decoder = Decoder(lstm_state_size,
                               2 * embedding_dim + node_type_embedding_dim,
                               lstm_state_size, hidden_state_size, dropout)
        self.predictor = Predictor(
            self.ast_reader, embedding_dim, lstm_state_size, lstm_state_size,
            hidden_state_size)

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
        """

        # Encode query
        nl_feature, _ = self.nl_reader(query)  # PackedSequence
        B = nl_feature.data.shape[1]

        # Decode action sequence
        feature = self.ast_reader(action, previous_action)
        history = torch.zeros(1, B, self.lstm_state_size,
                              device=nl_feature.data.device)
        h_0 = torch.zeros(B, self.lstm_state_size,
                          device=nl_feature.data.device)
        c_0 = torch.zeros(B, self.lstm_state_size,
                          device=nl_feature.data.device)
        feature, _ = self.decoder(None, nl_feature, None, feature,
                                  (history, h_0, c_0))
        return self.predictor(nl_feature, feature)
