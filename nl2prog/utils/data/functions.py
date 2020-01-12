import torch
from typing import Callable, List, Any, Union
from nl2prog.language.action \
    import Rule, CloseNode, ApplyRule, CloseVariadicFieldRule, \
    ActionSequence
from nl2prog.encoders import Samples
from nl2prog.utils.data import ListDataset
from nl2prog.utils import Query


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


def to_eval_dataset(dataset: torch.utils.data.Dataset) \
        -> torch.utils.data.Dataset:
    entries = []
    for group in dataset:
        gts = []
        for entry in group:
            gts.append(entry.ground_truth)
        for entry in group:
            query = entry.query
            entries.append((query, gts))
    return ListDataset(entries)
