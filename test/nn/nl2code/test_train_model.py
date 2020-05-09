import torch
import unittest
from torchnlp.encoders import LabelEncoder
import nl2prog.nn.utils.rnn as rnn
from nl2prog.ast.action import NodeConstraint, NodeType, ActionOptions
from nl2prog.encoders import Samples, ActionSequenceEncoder
from nl2prog.nn.nl2code import TrainModel


class TestTrain(unittest.TestCase):
    def test_parameters(self):
        samples = Samples(["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"],
                          ActionOptions(True, True))
        qencoder = LabelEncoder(["foo"], 0)
        aencoder = ActionSequenceEncoder(samples, 0)
        model = TrainModel(qencoder, aencoder, 1, 2, 6, 5, 0.0)
        self.assertEqual(34, len(list(model.named_parameters())))

    def test_shape(self):
        samples = Samples(["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"],
                          ActionOptions(True, True))
        qencoder = LabelEncoder(["foo"], 0)
        aencoder = ActionSequenceEncoder(samples, 0)
        model = TrainModel(qencoder, aencoder, 1, 2, 6, 5, 0.0)
        q0 = torch.LongTensor([1, 1])
        q1 = torch.LongTensor([1, 1, 1])
        action0 = torch.LongTensor([[-1, -1, -1]])
        action1 = torch.LongTensor([[-1, -1, -1], [1, -1, -1]])
        prev_action0 = torch.LongTensor([[-1, -1, -1]])
        prev_action1 = torch.LongTensor([[-1, -1, -1], [1, -1, -1]])
        query = rnn.pad_sequence([q0, q1])

        action = rnn.pad_sequence([action0, action1], -1)
        prev_action = rnn.pad_sequence([prev_action0, prev_action1], -1)
        results = model(query, (action, prev_action), None)
        rule_prob = results[0]
        token_prob = results[1]
        copy_prob = results[2]

        self.assertEqual(3, len(results))
        self.assertEqual((2, 2, 3), rule_prob.data.shape)
        self.assertEqual((2, 2, 3), token_prob.data.shape)
        self.assertEqual((2, 2, 3), copy_prob.data.shape)


if __name__ == "__main__":
    unittest.main()
