from typing import Generic, List, TypeVar

import torch
from torch import nn
from torchnlp.encoders import LabelEncoder

from mlprogram.languages import Token

Kind = TypeVar("Kind")
Value = TypeVar("Value")


class EncodeTokenQuery(nn.Module, Generic[Kind, Value]):
    def __init__(self, token_encoder: LabelEncoder):
        super().__init__()
        self.token_encoder = token_encoder

    def __call__(self, reference: List[Token[Kind, Value]]) -> torch.Tensor:
        return self.token_encoder.batch_encode(reference)


class EncodeWordQuery(nn.Module, Generic[Kind, Value]):
    def __init__(self, word_encoder: LabelEncoder):
        super().__init__()
        self.word_encoder = word_encoder

    def __call__(self, reference: List[Token[Kind, Value]]) -> torch.Tensor:
        return self.word_encoder.batch_encode([
            token.value for token in reference
        ])


class EncodeCharacterQuery(nn.Module, Generic[Kind, Value]):
    def __init__(self, char_encoder: LabelEncoder, max_word_length: int):
        super().__init__()
        self.char_encoder = char_encoder
        self.max_word_length = max_word_length

    def forward(self, reference: List[Token[Kind, Value]]) -> torch.Tensor:
        char_query = \
            torch.ones(len(reference), self.max_word_length).long() \
            * -1
        for i, token in enumerate(reference):
            chars = self.char_encoder.batch_encode(token.value)
            length = min(self.max_word_length, len(chars))
            char_query[i, :length] = chars[:length]
        return char_query
