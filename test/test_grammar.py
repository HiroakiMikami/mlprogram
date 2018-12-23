import unittest
from src.annotation import Annotation
from src.grammar import *

import ast
import numpy as np

# node_types
call = NodeType("Call", False)
expr = NodeType("expr", False)
expr_ = NodeType("expr", True)
name = NodeType("Name", False)
identifier = NodeType("identifier", False)
str_ = NodeType("str", False)

# rules
Root = Rule(ROOT, (Node("body", call), ))
Call = Rule(call, (Node("name", expr), Node("args", expr_)))
Expr = Rule(expr, (Node("name", name), ))
Name = Rule(name, (Node("value", identifier), ))
Str = Rule(identifier, (Node("Str", str_), ))
Expand2 = Rule(expr_, (Node("arg0", expr), Node("arg1", expr)))

grammar = Grammar([ROOT, call, expr, expr_, name, identifier, str_],
                  [Root, Call, Name, Str, Expand2, Expr],
                  ["func", "x", CLOSE_NODE])


class TestGrammar(unittest.TestCase):
    def test_to_decoder_input(self):
        annotation = Annotation(["x", "y"], {})

        "func(x, y)"
        sequence = [
            Root, Call, Expr, Name, Str, "func", CLOSE_NODE, Expand2, Expr,
            Name, Str, "x", CLOSE_NODE, Expr, Name, Str, "y", CLOSE_NODE
        ]
        next_node_type, input = to_decoder_input(sequence, annotation, grammar)
        self.assertEqual(next_node_type, None)
        self.assertEqual(
            input.action.astype(np.int32).tolist(),
            [[0, 0, 0], [1, 0, 0], [5, 0, 0], [2, 0, 0], [3, 0, 0], [0, 0, 0],
             [0, 2, 0], [4, 0, 0], [5, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0],
             [0, 2, 0], [5, 0, 0], [2, 0, 0], [3, 0, 0], [0, 0, 1], [0, 2, 0]])
        self.assertEqual(
            input.action_type[:, 0].astype(np.int32).tolist(),
            [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(
            input.action_type[:, 1].astype(np.int32).tolist(),
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        self.assertEqual(
            input.action_type[:, 2].astype(np.int32).tolist(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        self.assertEqual(
            input.node_type.astype(np.int32).tolist(),
            [0, 1, 2, 4, 5, 6, 6, 3, 2, 4, 5, 6, 6, 2, 4, 5, 6, 6])
        self.assertEqual(
            input.parent_action.astype(np.int32).tolist(),
            [0, 0, 1, 5, 2, 3, 3, 1, 4, 5, 2, 3, 3, 4, 5, 2, 3, 3])
        self.assertEqual(
            input.parent_index.astype(np.int32).tolist(),
            [-1, 0, 1, 2, 3, 4, 4, 1, 7, 8, 9, 10, 10, 7, 13, 14, 15, 15])

        "func(x,"
        sequence = [
            Root, Call, Expr, Name, Str, "func", CLOSE_NODE, Expand2, Expr,
            Name, Str, "x", CLOSE_NODE
        ]
        next_info, input = to_decoder_input(sequence, annotation, grammar)
        self.assertEqual(next_info, (expr, 7))

        "func(x, y)foo"
        sequence = [
            Root, Call, Expr, Name, Str, "func", CLOSE_NODE, Expand2, Expr,
            Name, Str, "x", CLOSE_NODE, Expr, Name, Str, "y", CLOSE_NODE, Name
        ]
        result = to_decoder_input(sequence, annotation, grammar)
        self.assertEqual(result, None)

        "func2(x, y)"
        sequence = [
            Root, Call, Expr, Name, Str, "func2", CLOSE_NODE, Expand2, Expr,
            Name, Str, "x", CLOSE_NODE, Expr, Name, Str, "y", CLOSE_NODE
        ]
        result = to_decoder_input(sequence, annotation, grammar)
        self.assertEqual(result, None)

        "func(--, y)"
        sequence = [
            Root, Call, Expr, Name, Str, "func", CLOSE_NODE, Expand2, Name,
            Str, "x", CLOSE_NODE, Expr, Name, Str, "y", CLOSE_NODE
        ]
        result = to_decoder_input(sequence, annotation, grammar)
        self.assertEqual(result, None)

        ""
        result = to_decoder_input([], annotation, grammar)
        self.assertEqual(result[0], (ROOT, -1))

        "func("
        result = to_decoder_input(
            [Root, Call, Expr, Name, Str, "func", CLOSE_NODE], annotation,
            grammar)
        self.assertEqual(result[0], (expr_, 1))
        ""
        result = to_decoder_input([Root, Call], annotation, grammar)
        self.assertEqual(result[0], (expr, 1))

    def test_to_sequence(self):
        annotation = Annotation(["x", "y"], {})
        "func(x, y)"
        sequence = [
            Root, Call, Expr, Name, Str, "func", CLOSE_NODE, Expand2, Expr,
            Name, Str, "x", CLOSE_NODE, Expr, Name, Str, "y", CLOSE_NODE
        ]
        _, input = to_decoder_input(sequence, annotation, grammar)
        sequence2 = to_sequence(input, annotation, grammar)
        self.assertEqual(sequence2, sequence)


if __name__ == "__main__":
    unittest.main()
