import torch
import unittest
from torchnlp.encoders import LabelEncoder
import io
import pickle

from nl2prog.language.action import NodeConstraint, NodeType, ExpandTreeRule
from nl2prog.encoders import ActionSequenceEncoder, Samples
from nl2prog.nn.treegen import TrainModel
from nl2prog.utils import Query
from nl2prog.utils.treegen import BeamSearchSynthesizer
from nl2prog.utils.data.nl2code \
    import CollateInput, CollateQuery, CollateActionSequence
from nl2prog.utils.transform.treegen import TransformQuery, TransformEvaluator


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
        cencoder = LabelEncoder(["a", "b", "c"], 0)
        aencoder = ActionSequenceEncoder(Samples(
            [ExpandTreeRule(X, [("value", Str)])],
            [X, Str],
            ["xyz"]
        ), 0)
        transform_input = TransformQuery(mock_tokenizer, qencoder, cencoder, 3)
        transform_evaluator = TransformEvaluator(aencoder, 2)
        train_model = TrainModel(qencoder, cencoder, aencoder,
                                 3, 2, 1, 1, 1, 1, 64, 64, 0.0)
        synthesizer = BeamSearchSynthesizer(
            2, transform_input, transform_evaluator,
            CollateInput(torch.device("cpu")),
            CollateActionSequence(torch.device("cpu")),
            CollateQuery(torch.device("cpu")),
            train_model.input_reader,
            train_model.action_sequence_reader, train_model.decoder,
            train_model.predictor, aencoder, is_subtype, max_steps=2)
        synthesizer.synthesize("abc")

    def test_pickable(self):
        qencoder = LabelEncoder(["abc"], 0)
        cencoder = LabelEncoder(["a", "b", "c"], 0)
        aencoder = ActionSequenceEncoder(Samples(
            [ExpandTreeRule(X, [("value", Str)])],
            [X, Str],
            ["xyz"]
        ), 0)
        transform_input = TransformQuery(mock_tokenizer, qencoder, cencoder, 3)
        transform_evaluator = TransformEvaluator(aencoder, 2)
        train_model = TrainModel(qencoder, cencoder, aencoder,
                                 3, 2, 1, 1, 1, 1, 64, 64, 0.0)
        synthesizer = BeamSearchSynthesizer(
            2, transform_input, transform_evaluator,
            CollateInput(torch.device("cpu")),
            CollateActionSequence(torch.device("cpu")),
            CollateQuery(torch.device("cpu")),
            train_model.input_reader,
            train_model.action_sequence_reader, train_model.decoder,
            train_model.predictor, aencoder, is_subtype, max_steps=2)
        x = io.BytesIO()
        pickle.dump(synthesizer, x)


if __name__ == "__main__":
    unittest.main()
