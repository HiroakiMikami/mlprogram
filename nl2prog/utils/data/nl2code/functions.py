import torch
from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Any, Tuple, Union
from nl2prog.language.action \
    import Rule, CloseNode, ApplyRule, CloseVariadicFieldRule, \
    ActionSequence
from nl2prog.language.evaluator import Evaluator
from nl2prog.encoders import Encoder, Samples
from nl2prog.utils.data import ListDataset


@dataclass
class Query:
    query_for_synth: List[str]
    query_for_dnn: List[str]


def get_samples(dataset: torch.utils.data.Dataset,
                tokenize_query: Callable[[str], Query],
                tokenize_token: Callable[[str], List[str]],
                to_action_sequence: Callable[[Any],
                                             Union[ActionSequence, None]]
                ) -> Samples:
    words = []
    rules = []
    node_types = []
    tokens = []

    for group in dataset:
        for entry in group:
            query = tokenize_query(entry.query)
            action_sequence = to_action_sequence(entry.ground_truth)
            words.extend(query.query_for_dnn)
            if action_sequence is None:
                continue
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


def to_train_dataset(dataset: torch.utils.data.Dataset,
                     tokenize_query: Callable[[str], Query],
                     tokenize_token: Callable[[str], List[str]],
                     to_action_sequence: Callable[[Any],
                                                  Union[ActionSequence, None]],
                     encoder: Encoder) \
        -> torch.utils.data.Dataset:
    entries = []
    for group in dataset:
        for entry in group:
            annotation = entry.query
            code = entry.ground_truth
            query = tokenize_query(annotation)
            query_tensor = \
                encoder.annotation_encoder.batch_encode(query.query_for_dnn)
            action_sequence = to_action_sequence(code)
            if action_sequence is None:
                continue
            evaluator = Evaluator()
            for action in action_sequence:
                evaluator.eval(action)
            a = \
                encoder.action_sequence_encoder.encode_action(
                    evaluator, query.query_for_synth)
            p = \
                encoder.action_sequence_encoder.encode_parent(
                    evaluator)
            if a is None:
                continue
            if np.any(a[-1, :].numpy() != -1):
                continue
            action_tensor = torch.cat(
                [a[:-1, 0].view(-1, 1), p[:-1, 1:3].view(-1, 2)],
                dim=1)
            dummy = torch.ones([1, 3]).to(a.dtype).to(a.device)
            prev_action = torch.cat([dummy, a[:-1, 1:]], dim=0)
            entries.append((query_tensor, action_tensor, prev_action))
    return ListDataset(entries)


def collate_train_dataset(data: List[Tuple[torch.LongTensor, torch.LongTensor,
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


def to_eval_dataset(dataset: torch.utils.data.Dataset,
                    tokenize_query: Callable[[str], Query],
                    tokenize_token: Callable[[str], List[str]],
                    to_action_sequence: Callable[[Any],
                                                 Union[ActionSequence, None]],
                    encoder: Encoder) \
        -> torch.utils.data.Dataset:
    entries = []
    for group in dataset:
        gts = []
        for entry in group:
            gts.append(entry.ground_truth)
        for entry in group:
            query = entry.query
            query = tokenize_query(query)
            entries.append((query, gts))
    return ListDataset(entries)
