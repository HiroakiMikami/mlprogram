import ast as python_ast

import mlprogram.languages as ast
from mlprogram.languages.python.ast_to_python_ast import to_builtin_type, to_python_ast
from mlprogram.languages.python.python_ast_to_ast import to_ast


class TestToBuiltinType(object):
    def test_int(self):
        assert 10 == to_builtin_type("10", "int")

    def test_bool(self):
        assert to_builtin_type("True", "bool")
        assert not to_builtin_type("False", "bool")

    def test_bytes(self):
        assert bytes([10, 10]) == \
            to_builtin_type(bytes([10, 10]).decode(), "bytes")

    def test_None(self):
        assert to_builtin_type("None", "NoneType") is None


class TestToPythonAST(object):
    def test_node(self):
        node = python_ast.Expr()
        name = python_ast.Name()
        setattr(name, "id", None)
        setattr(name, "ctx", None)
        setattr(node, "value", name)
        node2 = to_python_ast(to_ast(node, lambda x: [x]))
        assert python_ast.dump(node) == python_ast.dump(node2)

    def test_builtin_type(self):
        def split_value(x):
            return list(x)

        node = python_ast.List()
        ten = python_ast.Constant()
        setattr(ten, "value", 10)
        setattr(ten, "kind", None)
        setattr(node, "elts", [ten])
        setattr(node, "ctx", None)
        assert python_ast.dump(node) == \
            python_ast.dump(to_python_ast(
                to_ast(node,
                       split_value=split_value)))

    def test_variadic_args(self):
        node = python_ast.List()
        n = python_ast.Constant()
        s = python_ast.Constant()
        setattr(n, "n", 10)
        setattr(n, "kind", None)
        setattr(s, "s", "foo")
        setattr(s, "kind", None)
        setattr(node, "elts", [n, s])
        setattr(node, "ctx", None)
        assert python_ast.dump(node) == \
            python_ast.dump(to_python_ast(to_ast(node, lambda x: [x])))

    def test_optional_arg(self):
        node = python_ast.Yield()
        setattr(node, "value", None)
        assert python_ast.dump(node) == \
            python_ast.dump(to_python_ast(to_ast(node,
                                                 lambda x: [x])))

    def test_empty_list(self):
        node = python_ast.List()
        setattr(node, "ctx", None)
        setattr(node, "elts", [])
        assert python_ast.dump(node) == \
            python_ast.dump(to_python_ast(to_ast(node, lambda x: [x])))

    def test_token_list(self):
        node = python_ast.Global()
        setattr(node, "names", ["v1", "v2"])
        assert python_ast.dump(node) == \
            python_ast.dump(to_python_ast(to_ast(node, lambda x: [x])))
