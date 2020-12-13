from mlprogram.builtins import Environment
from mlprogram.languages.python.metrics import Bleu


class TestBleu(object):
    def test_bleu(self):
        bleu = Bleu()
        assert bleu(
            Environment({"ground_truth": "def f():\n  pass\n"}, set(["ground_truth"])),
            "def f():\n  pass\n") == 1
        assert bleu(
            Environment({"ground_truth": "def f():\n  pass\n"}, set(["ground_truth"])),
            "def f(arg):\n  pass\n") > 0.9
