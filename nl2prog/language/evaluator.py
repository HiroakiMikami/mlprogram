from dataclasses import dataclass
from typing import Dict, Optional, List, cast
from copy import deepcopy
import itertools

import nl2prog.language.action as A
from nl2prog.language.action \
    import Action, ActionSequence, ApplyRule, ExpandTreeRule
from nl2prog.language.ast import AST, Node, Leaf, Field, Root


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


class Evaluator:
    """
    Evaluator of action sequence.
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
    _options: A.ActionOptions
        The action sequence options.
    """

    def __init__(self, options: A.ActionOptions = A.ActionOptions(True, True)):
        self._tree = Tree(dict(), dict())
        self._action_sequence: List[Action] = []
        self._head_action_index: Optional[int] = None
        self._head_children_index: Dict[int, int] = dict()
        self._options = options

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

    def eval(self, action: Action):
        def append_action():
            index = len(self._action_sequence)
            self._action_sequence.append(action)
            self._tree.children[index] = []

        def update_head(close_variadic_field=False):
            head = self.head
            if head is None:
                return

            # The action that have children should be ApplyRule
            head_action: A.ApplyRule = self._action_sequence[head.action]
            # The action that have children should apply ExpandTreeRule
            head_rule: A.ExpandTreeRule = head_action.rule

            n_fields = len(head_rule.children)
            if n_fields <= head.field:
                # Return to the parent becase the rule does not create children
                self._head_action_index = \
                    self._tree.parent[head.action].action \
                    if self._tree.parent[head.action] is not None \
                    else None
                update_head()
                return

            if not self._options.retain_variadic_fields or \
               close_variadic_field or \
               head_rule.children[head.field][1].constraint != \
                    A.NodeConstraint.Variadic:
                self._head_children_index[head.action] += 1

            if self._head_children_index[head.action] < n_fields:
                return
            self._head_action_index = \
                self._tree.parent[head.action].action \
                if self._tree.parent[head.action] is not None \
                else None
            update_head()

        index = len(self._action_sequence)
        head = self.head
        if head is not None:
            head_action = cast(A.ApplyRule, self._action_sequence[head.action])
            head_rule = cast(A.ExpandTreeRule, head_action.rule)
            head_field: Optional[A.NodeType] = \
                head_rule.children[head.field][1]
        else:
            head_field = None

        if isinstance(action, A.ApplyRule):
            # ApplyRule
            rule: A.Rule = action.rule
            if isinstance(rule, A.ExpandTreeRule):
                # ExpandTree
                if head_field is not None and \
                        head_field.constraint == A.NodeConstraint.Token:
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
                if head_field.constraint == A.NodeConstraint.Node:
                    raise InvalidActionException(
                        "Applying ExpandTreeRule", action)
                if head_field.constraint == A.NodeConstraint.Token:
                    raise InvalidActionException(
                        "GenerateToken", action)
                if not self._options.retain_variadic_fields:
                    raise InvalidActionException(
                        "CloseVariadicField is invalid "
                        "(retain_variadic_fields=False)", action)

                append_action()
                # 2. Append the action to the head
                self._tree.children[head.action][head.field].append(index)
                self._tree.parent[index] = head

                # 3. Update head
                update_head(close_variadic_field=True)
        else:
            # GenerateToken
            token = action.token
            if head is None:
                raise InvalidActionException(
                    "Applying ExpandTreeRule", action)
            assert head_field is not None
            if head_field.constraint != A.NodeConstraint.Token:
                raise InvalidActionException(
                    "ApplyRule", action)
            if not self._options.split_non_terminal and token == A.CloseNode():
                raise InvalidActionException(
                    "GenerateToken", action)

            append_action()
            # 1. Append the action to the head
            self._tree.children[head.action][head.field].append(index)
            self._tree.parent[index] = head

            # 2. Update head if the token is closed.
            if token == A.CloseNode() or not self._options.split_non_terminal:
                update_head()

    def generate_ast(self) -> AST:
        """
        Generate AST from the action sequence

        Returns
        -------
        AST
            The AST corresponding to the action sequence
        """
        def generate_ast(head: int) -> AST:
            # The head action should be ApplyRule
            action = cast(A.ApplyRule, self._action_sequence[head])
            # The head action should apply ExpandTreeRule
            rule = cast(A.ExpandTreeRule, action.rule)

            ast = Node(rule.parent.type_name, [])
            for (name, node_type), actions in zip(
                    rule.children,
                    self._tree.children[head]):
                if node_type.constraint == A.NodeConstraint.Node:
                    # ApplyRule
                    ast.fields.append(
                        Field(name, node_type.type_name,
                              generate_ast(actions[0])))
                elif node_type.constraint == A.NodeConstraint.Variadic:
                    # Variadic
                    if self._options.retain_variadic_fields:
                        ast.fields.append(Field(name, node_type.type_name, []))
                        for act in actions:
                            a = cast(A.ApplyRule, self._action_sequence[act])
                            if isinstance(a.rule, A.CloseVariadicFieldRule):
                                break
                            assert isinstance(ast.fields[-1].value, list)
                            ast.fields[-1].value.append(generate_ast(act))
                    else:
                        childrens = cast(Node, generate_ast(actions[0]))
                        ast.fields.append(Field(
                            name, node_type.type_name,
                            list(itertools.chain.from_iterable(
                                [field.value if isinstance(field.value, list)
                                 else [field.value]
                                 for field in childrens.fields
                                 ]
                            ))
                        ))
                else:
                    # GenerateToken
                    value = ""
                    for action_idx in actions:
                        token = cast(A.GenerateToken,
                                     self._action_sequence[action_idx]).token
                        if isinstance(token, str):
                            value += token
                    ast.fields.append(Field(name, node_type.type_name,
                                            Leaf(node_type.type_name, value)
                                            ))

            return ast
        if len(self.action_sequence.sequence) == 0:
            return generate_ast(0)
        begin = self.action_sequence.sequence[0]
        if isinstance(begin, ApplyRule) and \
                isinstance(begin.rule, ExpandTreeRule):
            if begin.rule.parent.type_name == Root():
                # Ignore Root -> ???
                return generate_ast(1)
            return generate_ast(0)
        return generate_ast(0)

    def clone(self):
        """
        Generate and return the clone of this evaluator

        Returns
        -------
        Evaluator
            The cloned evaluator
        """
        evaluator = Evaluator(self._options)
        for key, value in self._tree.children.items():
            v = []
            for src in value:
                v.append(deepcopy(src))
            evaluator._tree.children[key] = v
        evaluator._tree.parent = deepcopy(self._tree.parent)
        evaluator._action_sequence = deepcopy(self._action_sequence)
        evaluator._head_action_index = self._head_action_index
        evaluator._head_children_index = deepcopy(self._head_children_index)

        return evaluator

    def parent(self, index: int) -> Optional[Parent]:
        return self._tree.parent[index]

    @property
    def action_sequence(self) -> ActionSequence:
        return ActionSequence(self._action_sequence, self._options)
