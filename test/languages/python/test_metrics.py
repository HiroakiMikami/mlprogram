from mlprogram.languages.python.metrics import Bleu


class TestBleu(object):
    def test_bleu(self):
        bleu = Bleu()
        assert bleu({"ground_truth": "def f():\n  pass\n"},
                    "def f():\n  pass\n") == 1
        assert bleu({"ground_truth": "def f():\n  pass\n"},
                    "def f(arg):\n  pass\n") > 0.9
