from torchnlp.encoders import LabelEncoder
from nl2prog.nn import TrainModel as BaseTrainModel
from nl2prog.nn.treegen\
    import NLReader, ActionSequenceReader, Decoder, Predictor
from nl2prog.encoders import ActionSequenceEncoder


class TrainModel(BaseTrainModel):
    def __init__(self, query_encoder: LabelEncoder,
                 char_encoder: LabelEncoder,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_token_len: int, max_arity: int,
                 num_heads: int,
                 num_nl_reader_blocks: int, num_ast_reader_blocks: int,
                 num_decoder_blocks: int, hidden_size: int,
                 feature_size: int, dropout: float):
        rule_num = \
            action_sequence_encoder._rule_encoder.vocab_size
        token_num = \
            action_sequence_encoder._token_encoder.vocab_size
        node_type_num = \
            action_sequence_encoder._node_type_encoder.vocab_size
        token_num = \
            action_sequence_encoder._token_encoder. vocab_size
        super(TrainModel, self).__init__(
            NLReader(
                query_encoder.vocab_size, char_encoder.vocab_size,
                max_token_len, hidden_size, hidden_size, num_heads, dropout,
                num_nl_reader_blocks),
            ActionSequenceReader(
                rule_num, token_num, node_type_num, max_arity, hidden_size,
                hidden_size, 3, num_heads, dropout, num_ast_reader_blocks),
            Decoder(rule_num, token_num, hidden_size, feature_size,
                    hidden_size, num_heads, dropout, num_decoder_blocks),
            Predictor(hidden_size, hidden_size, rule_num, token_num,
                      hidden_size)
        )
