import ast as python_ast

from mlprogram.languages.python import IsSubtype
from mlprogram.languages.python.utils import is_builtin_type


class TestIsBuiltinType(object):
    def test_ast(self):
        assert not is_builtin_type(python_ast.Expr())

    def test_builtin(self):
        assert is_builtin_type(10)
        assert is_builtin_type(None)
        assert is_builtin_type(True)


class TestIsSubtype(object):
    def test_ast(self):
        assert IsSubtype()("Expr", "AST")
        assert not IsSubtype()("Expr", "Name")

    def test_builtin(self):
        assert IsSubtype()("int", "int")
        assert not IsSubtype()("str", "int")

    def test_special_type(self):
        assert IsSubtype()("str__proxy", "str__proxy")
        assert not IsSubtype()("str__proxy", "str")
