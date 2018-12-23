from typing import Dict, List, NamedTuple
import numpy as np


class Annotation(NamedTuple):
    query: List[str]
    mappings: Dict[str, str]

    def __str__(self):
        return "Annotation(query={}, mappings={})".format(
            self.query, self.mappings)


class EncoderInput(NamedTuple):
    query: np.ndarray

    def __str__(self):
        return "EncoderInput({})".format(self.query)


UNKNOWN = "<unknown>"


def to_encoder_input(annotation: Annotation,
                     word_to_id: Dict[str, int]) -> EncoderInput:
    assert (UNKNOWN in word_to_id)

    length = len(annotation.query)

    def to_id(word):
        if word in word_to_id:
            return word_to_id[word]
        else:
            return word_to_id[UNKNOWN]

    return EncoderInput(np.array(list(map(to_id, annotation.query))))
