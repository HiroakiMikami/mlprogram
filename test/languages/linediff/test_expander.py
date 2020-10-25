from mlprogram.languages.linediff import Expander
from mlprogram.languages.linediff import Diff
from mlprogram.languages.linediff import Remove


class TestExpander(object):
    def test_expand(self):
        expander = Expander()
        assert expander.expand(Remove(0)) == [Remove(0)]
        assert expander.expand(Diff([Remove(0)])) == [Remove(0)]

    def test_unexpand(self):
        expander = Expander()
        assert expander.unexpand([Remove(1)]) == Diff([Remove(1)])
        assert expander.unexpand([Diff([Remove(1)]), Remove(0)]) == \
            Diff([Remove(1), Remove(0)])
