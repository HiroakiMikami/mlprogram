import torch
import unittest
import nl2prog.nn.utils.rnn as rnn
from nl2prog.language.action import NodeConstraint, NodeType, ActionOptions
from nl2prog.encoders import Samples, Encoder
from nl2prog.nn.treegen import TrainModel


class TestTrain(unittest.TestCase):
    def test_parameters(self):
        samples = Samples(["foo"], ["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"])
        encoder = Encoder(samples, 0, 0)
        model = TrainModel(encoder, 3, 3, 3, 1, 3, 3, 3, 3, 3, 0.0)
        self.assertEqual(190, len(list(model.named_parameters())))

    def test_shape(self):
        samples = Samples(["foo"], ["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"])
        encoder = Encoder(samples, 0, 0, options=ActionOptions(False, False))
        model = TrainModel(encoder, 3, 3, 3, 1, 3, 3, 3, 3, 3, 0.0)
        q0 = torch.randint(1, [2])
        qc0 = torch.randint(256, [2, 3])
        q1 = torch.randint(1, [3])
        qc1 = torch.randint(256, [3, 3])
        a0 = torch.randint(1, [1])
        at0 = torch.randint(1, [1, 4])
        a1 = torch.randint(1, [2])
        at1 = torch.randint(1, [2, 4])
        d0 = torch.LongTensor([[0]])
        d1 = torch.LongTensor([[0], [1]])
        m0 = torch.LongTensor([[0, 0], [0, 0]])
        m1 = torch.LongTensor([[0, 0], [1, 0]])

        q = rnn.pad_sequence([q0, q1])
        qc = rnn.pad_sequence([qc0, qc1])
        a = rnn.pad_sequence([a0, a1])
        at = rnn.pad_sequence([at0, at1])
        d = rnn.pad_sequence([d0, d1]).data.view(2, 2)
        m = torch.cat([m0.view(1, 2, 2), m1.view(1, 2, 2)], dim=0).float()

        results = model(q, qc, a, at, d, m)
        rule_prob = results[0]
        token_prob = results[1]
        copy_prob = results[2]

        self.assertEqual((2, 2, 2), rule_prob.data.shape)
        self.assertEqual((2, 2, 2), token_prob.data.shape)
        self.assertEqual((2, 2, 3), copy_prob.data.shape)


if __name__ == "__main__":
    unittest.main()
