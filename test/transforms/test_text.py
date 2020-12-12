import numpy as np
from torchnlp.encoders import LabelEncoder

from mlprogram import Environment
from mlprogram.languages import Token
from mlprogram.transforms.text import (
    EncodeCharacterQuery,
    EncodeTokenQuery,
    EncodeWordQuery,
    ExtractReference,
)


class TestExtractReference(object):
    def test_happy_path(self):
        def tokenize(value: str):
            return [Token(None, value + "dnn", value)]

        transform = ExtractReference(tokenize)
        result = transform(Environment({"text_query": ""}))
        assert [Token(None, "dnn", "")] == result["reference"]


class TestEncodeWordQuery(object):
    def test_happy_path(self):
        transform = EncodeWordQuery(LabelEncoder(["dnn"]))
        result = transform(Environment(
            {"reference": [Token(None, "dnn", "")]}
        ))
        assert [1] == result["word_nl_query"].numpy().tolist()


class TestEncodeTokenQuery(object):
    def test_happy_path(self):
        transform = EncodeTokenQuery(LabelEncoder([Token(None, "dnn", "")]))
        result = transform(Environment(
            {"reference": [Token(None, "dnn", "")]}
        ))
        assert [1] == result["token_nl_query"].numpy().tolist()


class TestEncodeCharacterQuery(object):
    def test_simple_case(self):
        cencoder = LabelEncoder(["a", "b", "t", "e"], 0)
        transform = EncodeCharacterQuery(cencoder, 3)
        result = transform(Environment(
            {"reference": [
                Token(None, "ab", "ab"),
                Token(None, "test", "test")
            ]}
        ))
        char_query = result["char_nl_query"]
        assert np.array_equal([[1, 2, -1], [3, 4, 0]],
                              char_query.numpy())
