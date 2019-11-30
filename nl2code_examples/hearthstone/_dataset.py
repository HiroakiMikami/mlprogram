import torch
from torch.utils.data import Dataset
from torchnlp.encoders import LabelEncoder
from dataclasses import dataclass
import numpy as np
import re
from nltk import tokenize
from typing import List, Tuple, Union
from nl2code.language.encoder import Encoder
from nl2code.language.python import to_ast
from nl2code.language.action import ActionSequence, Rule, NodeType
from nl2code.language.action import ApplyRule
from nl2code.language.action import CloseNode, CloseVariadicFieldRule
from nl2code.language.evaluator import Evaluator
from nl2code_examples.hearthstone import parse, unparse

tokenizer = tokenize.WhitespaceTokenizer()


def tokenize_annotation(annotation: str) -> List[str]:
    """
    Tokenize annotation

    Parameters
    ----------
    annotation: str

    Returns
    -------
    query: List[str]
        query is the list of words.
    """

    query = []
    for word in tokenizer.tokenize(annotation):
        query.append(word)

        vars = list(filter(lambda x: len(x) > 0,
                           word.split('.')))  # split by '.'
        if len(vars) > 1:
            for v in vars:
                query.append(v)
    return query


def tokenize_token(value: str, split_camel_case: bool = False) -> List[str]:
    if split_camel_case and re.search(
            r"^[A-Z].*", value) and (" " not in value):
        # Camel Case
        words = re.findall(r"[A-Z][a-z]+", value)
        if "".join(words) == value:
            return words
        else:
            return [value]
    else:
        # Divide by space
        words = re.split(r"( +)", value)
        return [word for word in words if word != ""]


def to_action_sequence(code: str) -> ActionSequence:
    pyast = parse(code)
    ast = to_ast(pyast)
    return ast.to_action_sequence(tokenize_token)


@dataclass
class Entry:
    """
    Entry in Hearthstone dataset

    Attributes
    ----------
    annotation: str
        The description of the code
    code: str
        The python source code
    """
    annotation: str
    code: str


@dataclass
class Samples:
    words: List[str]
    rules: List[Rule]
    node_types: List[NodeType]
    tokens: List[Union[str, CloseNode]]


class RawDataset(Dataset):
    def __init__(self, entries: List[Entry], transform=None):
        self._entries = entries
        self._transform = transform

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        if self._transform is not None:
            return self._transform(entry)
        return entry

    @property
    def samples(self, split_camel_case: bool = False) -> Samples:
        words = []
        rules = []
        node_types = []
        tokens = []

        for entry in self._entries:
            query = \
                tokenize_annotation(entry.annotation)
            action_sequence = to_action_sequence(entry.code)
            words.extend(query)
            for action in action_sequence:
                if isinstance(action, ApplyRule):
                    rule: Rule = action.rule
                    if rule != CloseVariadicFieldRule():
                        rules.append(rule)
                        node_types.append(rule.parent)
                        for _, child in rule.children:
                            node_types.append(child)
                else:
                    token = action.token
                    if token != CloseNode():
                        ts = tokenize_token(token,
                                            split_camel_case=split_camel_case)
                        tokens.extend(ts)

        return Samples(words, rules, node_types, tokens)


class DatasetEncoder:
    annotation_encoder: LabelEncoder
    action_sequence_encoder: Encoder

    def __init__(self,
                 samples: Samples,
                 word_threshold: int,
                 token_threshold: int):
        """
        Parameters
        ----------
        samples: Samples
            The list of words, tokens, rules, and node_types
        word_threshold: int
        token_threshold: int
        """
        self.annotation_encoder = LabelEncoder(samples.words,
                                               min_occurrences=word_threshold)
        self.action_sequence_encoder = Encoder(
            samples.rules, samples.node_types,
            samples.tokens, token_threshold)


class TrainDataset(Dataset):
    def __init__(self, raw_dataset: RawDataset,
                 encoder: DatasetEncoder,
                 transform=None):
        """
        Parameters
        ----------
        raw_dataset: RawDataset
            The original dataset
        encoder: DatasetEncoder
            The encoder for annotation and code
        """
        self._entries = []
        for entry in raw_dataset:
            annotation = entry.annotation
            code = entry.code
            query = tokenize_annotation(annotation)
            query_tensor = \
                encoder.annotation_encoder.batch_encode(query)
            action_sequence = to_action_sequence(code)
            evaluator = Evaluator()
            for action in action_sequence:
                evaluator.eval(action)
            action_sequence_tensor = \
                encoder.action_sequence_encoder.encode(evaluator, query)
            if action_sequence_tensor is None:
                continue
            if np.any(action_sequence_tensor.action[-1, :].numpy() != -1):
                continue
            self._entries.append((query_tensor,
                                  action_sequence_tensor.action[:-1],
                                  action_sequence_tensor.previous_action))
        self._transform = transform

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        if self._transform is not None:
            return self._transform(entry)
        return entry

    @staticmethod
    def collate(data: List[Tuple[torch.LongTensor, torch.LongTensor,
                                 torch.LongTensor]]) \
        -> Tuple[List[torch.LongTensor], List[torch.LongTensor],
                 List[torch.LongTensor]]:
        xs = []
        ys = []
        zs = []
        for x, y, z in data:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return xs, ys, zs


class EvalDataset(Dataset):
    def __init__(self, raw_dataset: RawDataset,
                 encoder: DatasetEncoder,
                 max_steps: int,
                 skip_impossible_entry: bool = True,
                 transform=None):
        """
        Parameters
        ----------
        raw_dataset: RawDataset
            The original dataset
        encoder: DatasetEncoder
            The encoder for annotation and code
        max_steps: int
        skip_impossible_entry: bool
        """
        self._entries = []
        for entry in raw_dataset:
            annotation = entry.annotation
            code = entry.code
            query = tokenize_annotation(annotation)
            action_sequence = to_action_sequence(code)
            evaluator = Evaluator()
            for action in action_sequence:
                evaluator.eval(action)
            action_sequence_tensor = \
                encoder.action_sequence_encoder.encode(evaluator, query)
            if skip_impossible_entry:
                if len(action_sequence) > max_steps:
                    continue
                if action_sequence_tensor is None:
                    continue
                if np.any(action_sequence_tensor.action[-1].numpy() != -1):
                    continue
            self._entries.append((query,
                                  unparse(parse(code))))
        self._transform = transform

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        if self._transform is not None:
            return self._transform(entry)
        return entry
