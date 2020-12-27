from mlprogram.languages.python.metrics import Bleu


class TestBleu(object):
    def test_bleu(self):
        bleu = Bleu()
        assert bleu(
            expected="def f():\n  pass\n",
            actual="def f():\n  pass\n"
        ) == 1
        assert bleu(
            expected="def f():\n  pass\n",
            actual="def f(arg):\n  pass\n"
        ) > 0.9
