import numpy as np
from torchnlp.encoders import LabelEncoder

from mlprogram.languages import Token
from mlprogram.transforms.text import (
    EncodeCharacterQuery,
    EncodeTokenQuery,
    EncodeWordQuery,
)


class TestEncodeWordQuery(object):
    def test_happy_path(self):
        transform = EncodeWordQuery(LabelEncoder(["dnn"]))
        result = transform([Token(None, "dnn", "")])
        assert [1] == result.numpy().tolist()


class TestEncodeTokenQuery(object):
    def test_happy_path(self):
        transform = EncodeTokenQuery(LabelEncoder([Token(None, "dnn", "")]))
        result = transform([Token(None, "dnn", "")])
        assert [1] == result.numpy().tolist()


class TestEncodeCharacterQuery(object):
    def test_simple_case(self):
        cencoder = LabelEncoder(["a", "b", "t", "e"], 0)
        transform = EncodeCharacterQuery(cencoder, 3)
        char_query = transform(
            [Token(None, "ab", "ab"), Token(None, "test", "test")]
        )
        assert np.array_equal([[1, 2, -1], [3, 4, 0]],
                              char_query.numpy())
