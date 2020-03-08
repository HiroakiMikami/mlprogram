import unittest
from torchnlp.encoders import LabelEncoder
import io
import pickle

from nl2prog.language.action import NodeConstraint, NodeType, ExpandTreeRule
from nl2prog.encoders import ActionSequenceEncoder, Samples
from nl2prog.nn.nl2code import TrainModel
from nl2prog.utils import Query
from nl2prog.utils.nl2code import BeamSearchSynthesizer
from nl2prog.utils.transform.nl2code import TransformQuery


def mock_tokenizer(query):
    return Query([query], [query])


X = NodeType("X", NodeConstraint.Node)
Str = NodeType("Str", NodeConstraint.Token)


def is_subtype(arg0, arg1):
    if arg0 == arg1:
        return True
    return False


class TestBeamSearchSynthesizer(unittest.TestCase):
    def test_shape(self):
        qencoder = LabelEncoder(["abc"], 0)
        aencoder = ActionSequenceEncoder(Samples(
            [ExpandTreeRule(X, [("value", Str)])],
            [X, Str],
            ["xyz"]
        ), 0)
        transform_input = TransformQuery(mock_tokenizer, qencoder)
        train_model = TrainModel(qencoder, aencoder, 1, 2, 6, 5, 0.0)
        synthesizer = BeamSearchSynthesizer(
            2, transform_input, train_model.input_reader,
            train_model.action_sequence_reader, train_model.decoder,
            train_model.predictor, aencoder, 3, 2, is_subtype,
            max_steps=2)
        synthesizer.synthesize("abc")

    def test_pickable(self):
        x = io.BytesIO()
        qencoder = LabelEncoder(["abc"], 0)
        aencoder = ActionSequenceEncoder(Samples(
            [ExpandTreeRule(X, [("value", Str)])],
            [X, Str],
            ["xyz"]
        ), 0)
        transform_input = TransformQuery(mock_tokenizer, qencoder)
        train_model = TrainModel(qencoder, aencoder, 1, 2, 6, 5, 0.0)
        synthesizer = BeamSearchSynthesizer(
            2, transform_input, train_model.input_reader,
            train_model.action_sequence_reader, train_model.decoder,
            train_model.predictor, aencoder, 3, 2, is_subtype,
            max_steps=2)
        pickle.dump(synthesizer, x)


if __name__ == "__main__":
    unittest.main()
