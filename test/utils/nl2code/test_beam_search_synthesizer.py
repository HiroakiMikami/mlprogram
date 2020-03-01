import unittest
from torchnlp.encoders import LabelEncoder

from nl2prog.utils import Query
from nl2prog.utils.nl2code import BeamSearchSynthesizer
from nl2prog.language.action import NodeConstraint, NodeType, ExpandTreeRule
from nl2prog.encoders import ActionSequenceEncoder, Samples
from nl2prog.nn.nl2code import TrainModel


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
        train_model = TrainModel(qencoder, aencoder, 1, 2, 6, 5, 0.0)
        synthesizer = BeamSearchSynthesizer(
            2, mock_tokenizer, train_model.nl_reader, train_model.ast_reader,
            train_model.decoder, train_model.predictor, qencoder,
            aencoder, 3, 2, is_subtype, max_steps=2)
        synthesizer.synthesize("abc")


if __name__ == "__main__":
    unittest.main()
