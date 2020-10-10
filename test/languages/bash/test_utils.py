from mlprogram.languages.bash import IsSubtype


class TestIsSubtype(object):
    def test_ast(self):
        assert IsSubtype()("Assign", "Node")
        assert not IsSubtype()("str", "Node")

    def test_builtin(self):
        assert IsSubtype()("str", "str")
        assert not IsSubtype()("Node", "str")
