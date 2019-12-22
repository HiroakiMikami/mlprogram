import torch
from torch.utils.data import Dataset
from torchnlp.encoders import LabelEncoder
from dataclasses import dataclass
import numpy as np
import re
from nltk import tokenize
from typing import List, Tuple, Union
from nl2prog.language.nl2code.encoder import Encoder
from nl2prog.language.nl2code.action \
    import ActionSequence, Rule, NodeType, ApplyRule, \
    CloseNode, CloseVariadicFieldRule, ast_to_action_sequence
from nl2prog.language.nl2code.evaluator import Evaluator
from nl2code_examples.nl2bash import parse, unparse

tokenizer = tokenize.WhitespaceTokenizer()


def get_subtokens(token: str) -> List[str]:
    return list(re.findall(r"[A-Za-z]+|\d+|\s+|.", token))


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
        subtokens = get_subtokens(word)
        assert(word == "".join(subtokens))

        if len(subtokens) == 1:
            query.append(word)
        else:
            query.append("SUB_START")
            query.extend(subtokens)
            query.append("SUB_END")
    return query


def tokenize_token(value: str) -> List[str]:
    tokens = []
    tokens = get_subtokens(value)
    assert(value == "".join(tokens))

    return tokens


def to_action_sequence(code: str) -> ActionSequence:
    try:
        ast = parse(code)
        return ast_to_action_sequence(ast, tokenize_token)
    except:  # noqa
        return None


@dataclass
class Entry:
    """
    Entry in NL2Bash dataset

    Attributes
    ----------
    annotation: str
        The description of the code
    code: str
        The python source code
    """
    annotation: str
    code: str


Group = List[Entry]


@dataclass
class Samples:
    words: List[str]
    rules: List[Rule]
    node_types: List[NodeType]
    tokens: List[Union[str, CloseNode]]


class RawDataset(Dataset):
    def __init__(self, groups: List[Group], transform=None):
        self._groups = groups
        self._transform = transform

    def __len__(self):
        return len(self._groups)

    def __getitem__(self, idx):
        group = self._groups[idx]
        if self._transform is not None:
            return self._transform(group)
        return group

    @property
    def samples(self) -> Samples:
        words = []
        rules = []
        node_types = []
        tokens = []

        for group in self._groups:
            for entry in group:
                query = \
                    tokenize_annotation(entry.annotation)
                action_sequence = to_action_sequence(entry.code)
                words.extend(query)
                if action_sequence is not None:
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
                                ts = tokenize_token(token)
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
        for group in raw_dataset:
            for entry in group:
                annotation = entry.annotation
                code = entry.code
                query = tokenize_annotation(annotation)
                query_tensor = \
                    encoder.annotation_encoder.batch_encode(query)
                action_sequence = to_action_sequence(code)
                if action_sequence is None:
                    continue
                evaluator = Evaluator()
                if action_sequence is not None:
                    for action in action_sequence:
                        evaluator.eval(action)
                    action_sequence_tensor = \
                        encoder.action_sequence_encoder.encode(evaluator,
                                                               query)
                else:
                    action_sequence_tensor = None
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
        for group in raw_dataset:
            references = [entry.code for entry in group]
            normalized_references = []
            for entry in group:
                code = entry.code
                x = to_action_sequence(code)
                if x is None:
                    continue
                normalized_references.append(unparse(parse(code)))
            for entry in group:
                annotation = entry.annotation
                code = entry.code
                query = tokenize_annotation(annotation)
                action_sequence = to_action_sequence(code)
                if action_sequence is None and skip_impossible_entry:
                    continue
                evaluator = Evaluator()
                if action_sequence is not None:
                    for action in action_sequence:
                        evaluator.eval(action)
                    action_sequence_tensor = \
                        encoder.action_sequence_encoder.encode(evaluator,
                                                               query)
                else:
                    action_sequence_tensor = None
                if skip_impossible_entry:
                    if len(action_sequence) > max_steps:
                        continue
                    if action_sequence_tensor is None:
                        continue
                    if np.any(action_sequence_tensor.action[-1].numpy() != -1):
                        continue
                self._entries.append((query,
                                      references,
                                      normalized_references))
        self._transform = transform

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        if self._transform is not None:
            return self._transform(entry)
        return entry
