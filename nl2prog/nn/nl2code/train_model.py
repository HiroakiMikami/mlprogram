from torchnlp.encoders import LabelEncoder
from nl2prog.nn import TrainModel as BaseTrainModel
from nl2prog.nn.nl2code \
    import NLReader, ActionSequenceReader, Decoder, Predictor
from nl2prog.encoders import ActionSequenceEncoder


class TrainModel(BaseTrainModel):
    def __init__(self, query_encoder: LabelEncoder,
                 action_sequence_encoder: ActionSequenceEncoder,
                 embedding_dim: int, node_type_embedding_dim: int,
                 lstm_state_size: int, hidden_state_size: int,
                 dropout: float):
        reader = ActionSequenceReader(
            action_sequence_encoder._rule_encoder.vocab_size,
            action_sequence_encoder._token_encoder.vocab_size,
            action_sequence_encoder._node_type_encoder.vocab_size,
            node_type_embedding_dim, embedding_dim)

        super(TrainModel, self).__init__(
            NLReader(query_encoder.vocab_size, embedding_dim, lstm_state_size,
                     dropout=dropout),
            reader,
            Decoder(lstm_state_size,
                    2 * embedding_dim + node_type_embedding_dim,
                    lstm_state_size, hidden_state_size, dropout),
            Predictor(
                reader, embedding_dim, lstm_state_size, lstm_state_size,
                hidden_state_size)
        )
        self.lstm_state_size = lstm_state_size
