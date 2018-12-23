from collections import namedtuple
from typing import List, NamedTuple, Union, Tuple
import numpy as np

from .annotation import Annotation


class NodeType(NamedTuple):
    type_name: str
    is_list: bool

    def __hash__(self):
        return hash(self.type_name) * hash(self.is_list)

    def __str__(self):
        if self.is_list:
            return "{}*".format(self.type_name)
        else:
            return "{}".format(self.type_name)

    @staticmethod
    def from_json(value):
        return NodeType(value[0], value[1])


class Node(NamedTuple):
    name: str
    node_type: NodeType

    def __hash__(self):
        return hash(self.name) * hash(self.node_type)

    def __str__(self):
        return "{}:{}".format(self.name, self.node_type)

    @staticmethod
    def from_json(value):
        return Node(value[0], NodeType.from_json(value[1]))


class Rule(NamedTuple):
    parent: NodeType
    children: Tuple[Node]

    def __hash__(self):
        return hash(self.parent) * hash(self.children)

    def __str__(self):
        return "Rule({} -> {})".format(
            self.parent, ",".join(map(lambda x: str(x), self.children)))

    @staticmethod
    def from_json(value):
        parent = NodeType.from_json(value[0])
        children = []
        for child in value[1]:
            children.append(Node.from_json(child))
        return Rule(parent, tuple(children))


Action = Union[Rule, str]
Sequence = List[Action]


class DecoderInput(NamedTuple):
    action: np.array
    action_type: np.array
    node_type: np.array
    parent_action: np.array
    parent_index: np.array


class Grammar:
    def __init__(self, node_types: List[NodeType], rules: List[Rule],
                 tokens: List[str]):
        assert (CLOSE_NODE in tokens)

        self.id_to_node_type = node_types
        self.id_to_rule = rules
        self.id_to_token = tokens
        self.node_type_to_id = {}
        for i, n in enumerate(self.id_to_node_type):
            self.node_type_to_id[n] = i
        self.rule_to_id = {}
        for i, r in enumerate(self.id_to_rule):
            self.rule_to_id[r] = i
        self.token_to_id = {}
        for i, t in enumerate(self.id_to_token):
            self.token_to_id[t] = i


CLOSE_NODE = "CLOSE_NODE"
ROOT = NodeType("root", False)


def to_decoder_input(
        sequence: Sequence, annotation: Annotation, grammar: Grammar
) -> Union[None, Tuple[DecoderInput, Union[None, NodeType]]]:
    length = len(sequence)

    action = np.zeros((length, 3))
    action_type = np.zeros((length, 3))
    parent_action = np.zeros((length, ))
    parent_index = np.zeros((length, ))
    node_types = np.zeros((length, ))

    index = 0
    is_invalid = False

    def visit(pindex, node_type, index):
        next_info = None
        current = index
        index += 1

        if current >= length:
            return current, (node_type, pindex)
        node_types[current] = grammar.node_type_to_id[node_type]
        a = sequence[current]
        parent_index[current] = pindex
        if pindex >= 0 and (not isinstance(sequence[pindex], str)):
            parent_action[current] = grammar.rule_to_id[sequence[pindex]]

        if isinstance(a, str):
            # GenToken
            token = a
            if token in grammar.token_to_id:
                # pick token from vocabulary
                action[current, 1] = grammar.token_to_id[token]
                action_type[current, 1] = 1

            # copy word from annotation
            for j, word in enumerate(annotation.query):
                if (word == token) or ((word in annotation.mappings)
                                       and annotation.mappings[word] == token):
                    action[current, 2] = j
                    action_type[current, 2] = 1

            # cannot generate
            if action_type[current, :].sum() == 0:
                raise RuntimeError("cannot generate")

            if token != CLOSE_NODE:
                index, tmp = visit(pindex, node_type, index)
                if tmp is not None:
                    next_info = tmp
        else:
            # ApplyRule
            if a.parent != node_type:
                raise RuntimeError("invalid rule")
            action[current, 0] = grammar.rule_to_id[a]
            action_type[current, 0] = 1
            for child in a.children:
                index, tmp = visit(current, child.node_type, index)
                if tmp is not None and next_info is None:
                    next_info = tmp
        return index, next_info

    try:
        index, next_info = visit(-1, ROOT, 0)
        if index < length:
            is_invalid = True
    except RuntimeError:
        is_invalid = True

    if is_invalid:
        return None
    else:
        return next_info, DecoderInput(action, action_type, node_types,
                                       parent_action, parent_index)


def to_sequence(input: DecoderInput, annotation: Annotation,
                grammar: Grammar) -> Sequence:
    sequence = []
    for action, action_type in zip(
            input.action.astype(np.int32).tolist(),
            input.action_type.astype(np.int32).tolist()):
        if action_type[0] == 1:
            # ApplyRule
            sequence.append(grammar.id_to_rule[action[0]])
        elif action_type[1] == 1:
            # GenToken (from vocabulary)
            sequence.append(grammar.id_to_token[action[1]])
        elif action_type[2] == 1:
            # GenToken (from annotation)
            w = annotation.query[action[2]]
            if w in annotation.mappings:
                w = annotation.mappings[w]
            sequence.append(w)
    return sequence
