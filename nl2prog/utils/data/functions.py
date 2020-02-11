import torch
from typing import Callable, List, Any, Optional
from nl2prog.language.action \
    import Rule, CloseNode, ApplyRule, CloseVariadicFieldRule, \
    ActionSequence
from nl2prog.encoders import Samples
from nl2prog.utils.data import ListDataset
from nl2prog.utils import Query


def get_words(dataset: torch.utils.data.Dataset,
              tokenize_query: Callable[[str], Query],
              ) -> List[str]:
    words = []

    for group in dataset:
        for entry in group:
            query = tokenize_query(entry.query)
            words.extend(query.query_for_dnn)

    return words


def get_samples(dataset: torch.utils.data.Dataset,
                tokenize_token: Callable[[str], List[str]],
                to_action_sequence: Callable[[Any],
                                             Optional[ActionSequence]]
                ) -> Samples:
    rules = []
    node_types = []
    tokens = []

    for group in dataset:
        for entry in group:
            action_sequence = to_action_sequence(entry.ground_truth)
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

    return Samples(rules, node_types, tokens)


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
