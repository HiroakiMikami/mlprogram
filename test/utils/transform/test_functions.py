import unittest
from torchnlp.encoders import LabelEncoder

from nl2prog.utils import Query
from nl2prog.utils.transform import TransformQuery


class TestTransformQuery(unittest.TestCase):
    def test_happy_path(self):
        def tokenize_query(value: str):
            return Query([value], [value + "dnn"])

        transform = TransformQuery(tokenize_query, LabelEncoder(["dnn"]))
        query_for_synth, query_tensor = transform("")
        self.assertEqual([""], query_for_synth)
        self.assertEqual([1], query_tensor.numpy().tolist())

    def test_tokenize_list_of_str(self):
        def tokenize_query(value: str):
            return Query([value], [value])

        transform = TransformQuery(tokenize_query, LabelEncoder(["0", "1"]))
        query_for_synth, query_tensor = transform(["0", "1"])
        self.assertEqual(["0", "1"], query_for_synth)
        self.assertEqual([1, 2], query_tensor.numpy().tolist())


if __name__ == "__main__":
    unittest.main()
