from typing import Callable, List, cast

import torch
from torchnlp.encoders import LabelEncoder

from mlprogram import Environment
from mlprogram.languages import Token


class ExtractReference(object):
    def __init__(self, extract_reference: Callable[[str], List[Token]]):
        self.extract_reference = extract_reference

    def __call__(self, entry: Environment) -> Environment:
        text_query = cast(str, entry.inputs["text_query"])
        entry.states["reference"] = self.extract_reference(text_query)

        return entry


class EncodeWordQuery(object):
    def __init__(self, word_encoder: LabelEncoder):
        self.word_encoder = word_encoder

    def __call__(self, entry: Environment) -> Environment:
        reference = entry.states["reference"]

        entry.states["word_nl_query"] = self.word_encoder.batch_encode([
            token.value for token in reference
        ])

        return entry


class EncodeCharacterQuery(object):
    def __init__(self, char_encoder: LabelEncoder, max_word_length: int):
        self.char_encoder = char_encoder
        self.max_word_length = max_word_length

    def __call__(self, entry: Environment) -> Environment:
        reference = entry.states["reference"]

        char_query = \
            torch.ones(len(reference), self.max_word_length).long() \
            * -1
        for i, token in enumerate(reference):
            chars = self.char_encoder.batch_encode(token.value)
            length = min(self.max_word_length, len(chars))
            char_query[i, :length] = chars[:length]
        entry.states["char_nl_query"] = char_query

        return entry
