from mlprogram.languages.c import TypoMutator
from mlprogram.languages.linediff import Diff, Interpreter


class TestTypoMutator(object):
    def test_empty_string(self):
        mutator = TypoMutator(5)
        assert mutator.mutate("") == ("", [("", "")], Diff([]))

    def test_inverse_diff(self):
        mutator = TypoMutator(3, seed=0)
        interpreter = Interpreter()
        orig = """int a = 0;
printf(\"%d\\n\", a);
int *b = {0, 1, 2};
int y = x.a;
"""
        for _ in range(100):
            mutated, test_cases, diff = mutator.mutate(orig)
            assert mutated == test_cases[0][0]
            assert interpreter.eval(diff, [mutated])[0] == orig
