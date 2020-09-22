from dataclasses import dataclass
from typing import Dict, Optional, List, cast, Any
from copy import deepcopy

from mlprogram.actions.action \
    import Action, ApplyRule, Rule, ExpandTreeRule, NodeConstraint, NodeType, \
    GenerateToken, CloseVariadicFieldRule
from mlprogram.languages.ast import AST, Node, Leaf, Field, Root
from mlprogram import logging


logger = logging.Logger(__name__)


class InvalidActionException(Exception):
    """
    Exception occurred if the action is invalid
    """

    def __init__(self, expected: str, actual: Action):
        """
        Parameters
        ----------
        expected: str
            The expected type or attributes
        actual: Action
            The actual action
        """
        super(InvalidActionException, self).__init__(
            f"Invalid action: {actual} (expected: {expected})")


@dataclass
class Parent:
    action: int
    field: int


@dataclass
class Tree:
    """
    Attributes
    ----------
    children: Dict[int, Tuple[int, EdgeLabel]]
        The adjacency dictionary. Each key represents the action generating
        the parent node. Each value is the list of the children actions.
    parent: Dict[int, Optional[int]]
        The adjacency dictionary. Each key represents the action generating
        the child node, and each value represents the action generating
        the parent node.
    """
    children: Dict[int, List[List[int]]]
    parent: Dict[int, Optional[Parent]]


class ActionSequence:
    """
    The action sequence.
    This receives a sequence of actions and generate a corresponding AST.

    Attributes
    ----------
    _tree: Tree
        The intermidiate AST.
    _action_sequence: List[Action]
        The sequence of actions to be evaluated.
    _head_action_index: Optional[Int]
        The index of the head AST node.
    _head_children_index: Dict[Int, Int]
        The relation between actions and their head indexes of fields.
    """

    def __init__(self):
        self._tree = Tree(dict(), dict())
        self._action_sequence: List[Action] = []
        self._head_action_index: Optional[int] = None
        self._head_children_index: Dict[int, int] = dict()

    @property
    def head(self) -> Optional[Parent]:
        """
        Return the index of the head (it will be the parent of
        the next action).
        """
        if self._head_action_index is None:
            return None
        return Parent(self._head_action_index,
                      self._head_children_index[self._head_action_index])

    def eval(self, action: Action) -> None:
        def append_action() -> None:
            index = len(self._action_sequence)
            self._action_sequence.append(action)
            self._tree.children[index] = []

        def update_head(close_variadic_field: bool = False) -> None:
            head = self.head
            if head is None:
                return

            # The action that have children should be ApplyRule
            head_action = cast(ApplyRule, self._action_sequence[head.action])
            # The action that have children should apply ExpandTreeRule
            head_rule = cast(ExpandTreeRule, head_action.rule)

            n_fields = len(head_rule.children)
            if n_fields <= head.field:
                # Return to the parent becase the rule does not create children
                tmp = self._tree.parent[head.action]
                if tmp is not None:
                    self._head_action_index = tmp.action
                else:
                    self._head_action_index = None
                update_head()
                return

            if close_variadic_field or \
               not head_rule.children[head.field][1].is_variadic:
                self._head_children_index[head.action] += 1

            if self._head_children_index[head.action] < n_fields:
                return
            tmp = self._tree.parent[head.action]
            if tmp is not None:
                self._head_action_index = tmp.action
            else:
                self._head_action_index = None
            update_head()

        index = len(self._action_sequence)
        head = self.head
        if head is not None:
            head_action = cast(ApplyRule, self._action_sequence[head.action])
            head_rule = cast(ExpandTreeRule, head_action.rule)
            head_field: Optional[NodeType] = \
                head_rule.children[head.field][1]
        else:
            head_field = None

        if isinstance(action, ApplyRule):
            # ApplyRule
            rule: Rule = action.rule
            if isinstance(rule, ExpandTreeRule):
                # ExpandTree
                if head_field is not None and \
                        head_field.constraint == NodeConstraint.Token:
                    raise InvalidActionException("GenerateToken", action)

                append_action()
                # 1. Add the action to the head
                if head is not None:
                    self._tree.children[head.action][head.field].append(index)
                    self._tree.parent[index] = head
                else:
                    self._tree.parent[index] = None
                # 2. Update children
                for _ in range(len(rule.children)):
                    self._tree.children[index].append([])
                # 3. Update head
                self._head_children_index[index] = 0
                self._head_action_index = index

                if len(rule.children) == 0:
                    update_head()
            else:
                # CloseVariadicField
                # Check whether head is variadic field
                if head is None:
                    raise InvalidActionException(
                        "Applying ExpandTreeRule", action)
                assert head_field is not None
                if not head_field.is_variadic:
                    raise InvalidActionException(
                        "Variadic Fields", action)

                append_action()
                # 2. Append the action to the head
                self._tree.children[head.action][head.field].append(index)
                self._tree.parent[index] = head

                # 3. Update head
                update_head(close_variadic_field=True)
        else:
            # GenerateToken
            if head is None:
                raise InvalidActionException(
                    "Applying ExpandTreeRule", action)
            assert head_field is not None
            if head_field.constraint != NodeConstraint.Token:
                raise InvalidActionException(
                    "ApplyRule", action)

            append_action()
            # 1. Append the action to the head
            self._tree.children[head.action][head.field].append(index)
            self._tree.parent[index] = head

            # 2. Update head if the token is closed.
            if not head_rule.children[head.field][1].is_variadic:
                update_head()

    def generate(self) -> AST:
        """
        Generate AST from the action sequence

        Returns
        -------
        AST
            The AST corresponding to the action sequence
        """
        def generate(head: int, node_type: Optional[NodeType] = None) -> AST:
            action = self._action_sequence[head]
            if isinstance(action, GenerateToken):
                assert node_type is not None
                assert node_type.type_name is not None
                return Leaf(node_type.type_name, action.token)
            elif isinstance(action, ApplyRule):
                # The head action should apply ExpandTreeRule
                rule = cast(ExpandTreeRule, action.rule)

                ast = Node(rule.parent.type_name, [])
                for (name, node_type), actions in zip(
                        rule.children,
                        self._tree.children[head]):
                    assert node_type.type_name is not None
                    if node_type.is_variadic:
                        # Variadic field
                        ast.fields.append(
                            Field(name, node_type.type_name, []))
                        for act in actions:
                            if isinstance(self._action_sequence[act],
                                          ApplyRule):
                                a = cast(ApplyRule, self._action_sequence[act])
                                if isinstance(a.rule, CloseVariadicFieldRule):
                                    break
                            assert isinstance(ast.fields[-1].value, list)
                            ast.fields[-1].value.append(
                                generate(act, node_type))
                    else:
                        ast.fields.append(
                            Field(name, node_type.type_name,
                                  generate(actions[0], node_type)))
                return ast
            else:
                logger.critical(f"Invalid type of action: {type(action)}")
                raise InvalidActionException("Action", action)

        if len(self.action_sequence) == 0:
            return generate(0)
        begin = self.action_sequence[0]
        if isinstance(begin, ApplyRule) and \
                isinstance(begin.rule, ExpandTreeRule):
            if begin.rule.parent.type_name == Root():
                # Ignore Root -> ???
                return generate(1)
            return generate(0)
        return generate(0)

    def clone(self):
        """
        Generate and return the clone of this action_sequence

        Returns
        -------
        action_sequence
            The cloned action_sequence
        """
        action_sequence = ActionSequence()
        for key, value in self._tree.children.items():
            v = []
            for src in value:
                v.append(deepcopy(src))
            action_sequence._tree.children[key] = v
        action_sequence._tree.parent = deepcopy(self._tree.parent)
        action_sequence._action_sequence = deepcopy(self._action_sequence)
        action_sequence._head_action_index = self._head_action_index
        action_sequence._head_children_index = \
            deepcopy(self._head_children_index)

        return action_sequence

    def parent(self, index: int) -> Optional[Parent]:
        return self._tree.parent[index]

    @property
    def action_sequence(self) -> List[Action]:
        return self._action_sequence

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ActionSequence):
            return False
        return self.action_sequence == other.action_sequence

    def __hash__(self) -> int:
        return hash(tuple(self.action_sequence))

    def __str__(self) -> str:
        return f"{self.action_sequence}"

    def __repr__(self) -> str:
        return str(self.action_sequence)
