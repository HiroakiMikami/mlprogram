import unittest
import numpy as np

from nl2code_examples.nl2bash import Entry, DatasetEncoder
from nl2code_examples.nl2bash import RawDataset, TrainDataset, EvalDataset
from nl2code_examples.nl2bash._dataset import tokenize_annotation, \
    tokenize_token, get_subtokens


class TestGetSubtokens(unittest.TestCase):
    def test_words(self):
        self.assertEqual(["foo"], get_subtokens("foo"))

    def test_numbers(self):
        self.assertEqual(["foo", "10"], get_subtokens("foo10"))

    def test_others(self):
        self.assertEqual(["/", "foo"], get_subtokens("/foo"))
        self.assertEqual(["$", "{", "foo", "10", "}"],
                         get_subtokens("${" + "foo10" + "}"))


class TestTokenizeAnnotation(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_annotation("foo bar")
        self.assertEqual(["foo", "bar"], query)

    def test_subtokens(self):
        query = tokenize_annotation('foo.bar')
        self.assertEqual(["SUB_START", "foo", ".", "bar", "SUB_END"], query)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], tokenize_token("test"))

    def test_subtokens(self):
        query = tokenize_token('foo.bar')
        self.assertEqual(["foo", ".", "bar"], query)


class TestRawDataset(unittest.TestCase):
    def test_without_transform(self):
        groups = [[Entry("annotation0", "code0")],
                  [Entry("annotation1", "code1")]]
        dataset = RawDataset(groups)
        self.assertEqual(2, len(dataset))
        self.assertEqual([Entry("annotation0", "code0")], dataset[0])
        self.assertEqual([Entry("annotation1", "code1")], dataset[1])

    def test_transform(self):
        def transform(group):
            return len(group)
        groups = [[Entry("annotation0", "code0")],
                  [Entry("annotation1", "code1")]]
        dataset = RawDataset(groups, transform=transform)
        self.assertEqual(1, dataset[0])

    def test_samples(self):
        groups = [[Entry("foo bar", "y=x1")],
                  [Entry("test foo", "f x")]]
        dataset = RawDataset(groups)
        d = dataset.samples
        self.assertEqual(["foo", "bar", "test", "foo"], d.words)
        self.assertEqual(["y", "=", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(8, len(d.rules))
        self.assertEqual(16, len(d.node_types))


class TestTrainDataset(unittest.TestCase):
    def test_simple_case(self):
        groups = [[Entry("foo bar", "y=x1")]]
        dataset = RawDataset(groups)
        d = dataset.samples
        d.words = ["foo", "bar"]
        encoder = DatasetEncoder(d, 0, 0)
        tdataset = TrainDataset(dataset, encoder)
        query_tensor, action_tensor, prev_action_tensor = tdataset[0]
        self.assertTrue(np.array_equal([1, 2], query_tensor.numpy()))
        self.assertEqual((10, 3), action_tensor.shape)
        self.assertEqual((11, 3), prev_action_tensor.shape)

    def test_impossible_case(self):
        groups = [[Entry("foo bar", "y = (")]]
        dataset = RawDataset(groups)
        d = dataset.samples
        d.words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        encoder = DatasetEncoder(d, 0, 0)
        tdataset = TrainDataset(dataset, encoder)
        self.assertEqual(0, len(tdataset))


class TestEvalDataset(unittest.TestCase):
    def test_simple_case(self):
        groups = [[Entry("foo bar", "y=x1")]]
        dataset = RawDataset(groups)
        d = dataset.samples
        d.words = ["foo", "bar"]
        encoder = DatasetEncoder(d, 0, 0)
        vdataset = EvalDataset(dataset, encoder, 100, 100)
        query, ref, nref = vdataset[0]
        self.assertEqual(["foo", "bar"], query)
        self.assertEqual(["y=x1"], ref)
        self.assertEqual(["y=x1"], nref)

    def test_impossible_case(self):
        groups = [[Entry("foo bar", "y = (")]]
        dataset = RawDataset(groups)
        d = dataset.samples
        d.words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        encoder = DatasetEncoder(d, 0, 0)
        vdataset = EvalDataset(dataset, encoder, 1)
        self.assertEqual(0, len(vdataset))

        vdataset = EvalDataset(dataset, encoder, 100, False)
        self.assertEqual(1, len(vdataset))
        query, ref, nref = vdataset[0]
        self.assertEqual(["foo", "bar"], query)
        self.assertEqual(["y = ("], ref)
        self.assertEqual([], nref)


if __name__ == "__main__":
    unittest.main()
