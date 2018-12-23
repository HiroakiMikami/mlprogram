import json
import os
import glob
import numpy as np
from nltk import tokenize
from typing import List, NamedTuple, Union, Dict

from .annotation import Annotation, EncoderInput, to_encoder_input
from .grammar import Sequence, DecoderInput, to_decoder_input, Rule, NodeType, CLOSE_NODE, Grammar
from .python.grammar import to_sequence

tokenizer = tokenize.WhitespaceTokenizer()


class Sample(NamedTuple):
    annotation: Annotation
    encoder_input: Union[None, EncoderInput]
    sequence: Sequence
    decoder_input: Union[None, DecoderInput]


class Dataset:
    def __init__(self,
                 directory: str,
                 shuffle: bool = False,
                 rng: Union[None, np.random.RandomState] = None):
        self._shuffle = shuffle
        self._rng = rng

        # Get all annotation files
        annotation_files = glob.glob(os.path.join(directory, "*.anno"))

        ids = list(map(lambda x: os.path.splitext(x)[0], annotation_files))
        self._raw_data = []
        for id in ids:
            with open("{}.anno".format(id)) as f:
                w = "\n".join(f.readlines())
                if w.endswith(".") or w.endswith(","):
                    w = w[0:len(w) - 1]
                query = tokenizer.tokenize(w)
            with open("{}.reference_seq.json".format(id)) as f:
                sequence = json.load(f)

                def convert(x):
                    if isinstance(x, str):
                        return x
                    else:
                        return Rule.from_json(x)

                sequence = list(map(convert, sequence))
            with open("{}.mapping.json".format(id)) as f:
                mappings = json.load(f)
            annotation = Annotation(query, mappings)
            self._raw_data.append(Sample(annotation, None, sequence, None))

        self._rules = None
        self._node_types = None
        self._data = None
        self.size = None

    def next(self):
        retval = self._data[self._index]
        self._index += 1
        if self._index >= len(self._data):
            self.reset()
        return retval

    def reset(self):
        self._index = 0
        if self._shuffle:
            if self._rng is None:
                np.random.shuffle(self._data)
            else:
                self._rng.shuffle(self._data)

    @property
    def rules(self) -> List[Rule]:
        if self._rules is None:
            rs = set()
            for sample in self._raw_data:
                for action in sample.sequence:
                    if isinstance(action, str):
                        continue
                    rs.add(action)
            self._rules = list(rs)
        return self._rules

    @property
    def node_types(self) -> List[NodeType]:
        if self._node_types is None:
            ns = set()
            for sample in self._raw_data:
                for action in sample.sequence:
                    if isinstance(action, str):
                        continue
                    ns.add(action.parent)
                    for child in action.children:
                        ns.add(child.node_type)
            self._node_types = list(ns)
        return self._node_types

    def words(self, threshold: int):
        ws = {}
        for sample in self._raw_data:
            for word in sample.annotation.query:
                if not (word in ws):
                    ws[word] = 0
                ws[word] += 1
        ws = {k: v for k, v in ws.items() if v >= threshold}
        return list(ws.keys())

    def tokens(self, threshold: int):
        ts = {}
        for sample in self._raw_data:
            for action in sample.sequence:
                if not isinstance(action, str):
                    continue
                if action != CLOSE_NODE:
                    if not (action in ts):
                        ts[action] = 0
                    ts[action] += 1
        ts = {k: v for k, v in ts.items() if v >= threshold}
        return list(ts.keys())

    def prepare(self, word_to_id: Dict[str, int], grammar: Grammar):
        import nnabla.logger as logger
        data = []
        for i, sample in enumerate(self._raw_data):
            if i % 100 == 0:
                logger.info("Process {} sample".format(i))
            try:
                tmp = to_decoder_input(sample.sequence, sample.annotation,
                                       grammar)
                if (tmp is not None) and (tmp[0] is None):
                    data.append(
                        Sample(sample.annotation,
                               to_encoder_input(sample.annotation, word_to_id),
                               sample.sequence, tmp[1]))
            except KeyError as e:
                logger.info(e)
        self._data = data
        self.size = len(data)
        self.reset()
