import torch
import unittest
from torchnlp.encoders import LabelEncoder
import mlprogram.nn.utils.rnn as rnn
from mlprogram.action import NodeConstraint, NodeType, ActionOptions
from mlprogram.encoders import Samples, ActionSequenceEncoder
from mlprogram.nn.treegen import TrainModel


class TestTrain(unittest.TestCase):
    def test_parameters(self):
        samples = Samples(["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"],
                          ActionOptions(False, False))
        qencoder = LabelEncoder(["foo"], 0)
        cencoder = LabelEncoder([str(i) for i in range(255)], 0)
        aencoder = ActionSequenceEncoder(samples, 0)
        model = TrainModel(qencoder, cencoder, aencoder,
                           3, 3, 3, 1, 3, 3, 3, 3, 3, 0.0)
        self.assertEqual(196, len(list(model.named_parameters())))

    def test_shape(self):
        samples = Samples(["mock-rule"],
                          [NodeType("mock", NodeConstraint.Node)],
                          ["token"],
                          ActionOptions(False, False))
        qencoder = LabelEncoder(["foo"], 0)
        cencoder = LabelEncoder([str(i) for i in range(255)], 0)
        aencoder = ActionSequenceEncoder(samples, 0)
        model = TrainModel(qencoder, cencoder, aencoder,
                           3, 3, 3, 1, 3, 3, 3, 3, 3, 0.0)
        q0 = torch.randint(1, [2])
        qc0 = torch.randint(256, [2, 3])
        q1 = torch.randint(1, [3])
        qc1 = torch.randint(256, [3, 3])
        a0 = torch.randint(1, [1, 3])
        at0 = torch.randint(1, [1, 4, 3])
        a1 = torch.randint(1, [2, 3])
        at1 = torch.randint(1, [2, 4, 3])
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

        results = model((q, qc), (a, at, d, m), a)
        rule_prob = results[0]
        token_prob = results[1]
        copy_prob = results[2]

        self.assertEqual((2, 2, 2), rule_prob.data.shape)
        self.assertEqual((2, 2, 2), token_prob.data.shape)
        self.assertEqual((2, 2, 3), copy_prob.data.shape)


if __name__ == "__main__":
    unittest.main()
