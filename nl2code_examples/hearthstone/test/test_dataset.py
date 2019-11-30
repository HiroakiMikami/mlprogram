import unittest
import numpy as np

from nl2code_examples.hearthstone import Entry, DatasetEncoder
from nl2code_examples.hearthstone import RawDataset, TrainDataset, EvalDataset
from nl2code_examples.hearthstone._dataset import tokenize_annotation, \
    tokenize_token


class TestTokenizeAnnotation(unittest.TestCase):
    def test_simple_case(self):
        query = tokenize_annotation("foo bar")
        self.assertEqual(["foo", "bar"], query)

    def test_quoted_string_without_placeholder(self):
        query = tokenize_annotation('"quoted string" test')
        self.assertEqual(['"quoted', 'string"', "test"], query)

    def test_package_name_like_string(self):
        query = tokenize_annotation('foo.bar')
        self.assertEqual(["foo.bar", "foo", "bar"], query)


class TestTokenizeToken(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(["test"], tokenize_token("test"))

    def test_camel_case(self):
        self.assertEqual(["Foo", "Bar"], tokenize_token("FooBar", True))


class TestRawDataset(unittest.TestCase):
    def test_without_transform(self):
        entries = [Entry("annotation0", "code0"),
                   Entry("annotation1", "code1")]
        dataset = RawDataset(entries)
        self.assertEqual(2, len(dataset))
        self.assertEqual(Entry("annotation0", "code0"), dataset[0])
        self.assertEqual(Entry("annotation1", "code1"), dataset[1])

    def test_transform(self):
        def transform(entry: Entry):
            return "{}:{}".format(entry.annotation, entry.code)
        entries = [Entry("annotation0", "code0"),
                   Entry("annotation1", "code1")]
        dataset = RawDataset(entries, transform=transform)
        self.assertEqual(("annotation0:code0"), dataset[0])

    def test_samples(self):
        entries = [Entry("foo bar", "y = x + 1"),
                   Entry("test foo", "f(x)")]
        dataset = RawDataset(entries)
        d = dataset.samples
        self.assertEqual(["foo", "bar", "test", "foo"], d.words)
        self.assertEqual(["y", "x", "1", "f", "x"], d.tokens)
        self.assertEqual(10, len(d.rules))
        self.assertEqual(24, len(d.node_types))


class TestTrainDataset(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = RawDataset(entries)
        d = dataset.samples
        d.words = ["foo", "bar"]
        encoder = DatasetEncoder(d, 0, 0)
        tdataset = TrainDataset(dataset, encoder)
        query_tensor, action_tensor, prev_action_tensor = tdataset[0]
        self.assertTrue(np.array_equal([1, 2], query_tensor.numpy()))
        self.assertEqual((13, 3), action_tensor.shape)
        self.assertEqual((14, 3), prev_action_tensor.shape)

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = RawDataset(entries)
        d = dataset.samples
        d.words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        encoder = DatasetEncoder(d, 0, 0)
        tdataset = TrainDataset(dataset, encoder)
        self.assertEqual(0, len(tdataset))


class TestEvalDataset(unittest.TestCase):
    def test_simple_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = RawDataset(entries)
        d = dataset.samples
        d.words = ["foo", "bar"]
        encoder = DatasetEncoder(d, 0, 0)
        vdataset = EvalDataset(dataset, encoder, 100, 100)
        query, code = vdataset[0]
        self.assertEqual(["foo", "bar"], query)
        self.assertEqual("\ny = (x + 1)\n", code)

    def test_impossible_case(self):
        entries = [Entry("foo bar", "y = x + 1")]
        dataset = RawDataset(entries)
        d = dataset.samples
        d.words = ["foo", "bar"]
        d.tokens = ["y", "1"]
        encoder = DatasetEncoder(d, 0, 0)
        vdataset = EvalDataset(dataset, encoder, 1)
        self.assertEqual(0, len(vdataset))

        vdataset = EvalDataset(dataset, encoder, 100, False)
        self.assertEqual(1, len(vdataset))


if __name__ == "__main__":
    unittest.main()
