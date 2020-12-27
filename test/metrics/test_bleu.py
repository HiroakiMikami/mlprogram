from mlprogram.metrics import Bleu


class TestBleu(object):
    def test_simple_case(self):
        m = Bleu()
        assert m(expected="int", actual="xxx") < 0.1
