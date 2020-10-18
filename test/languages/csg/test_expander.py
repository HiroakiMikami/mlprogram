from mlprogram.languages.csg import Expander
from mlprogram.languages.csg import Circle
from mlprogram.languages.csg import Rotation
from mlprogram.languages.csg import Difference
from mlprogram.languages.csg import Reference


class TestExpander(object):
    def test_expand(self):
        expander = Expander()
        assert expander.expand(Circle(1)) == [Circle(1)]
        assert expander.expand(Rotation(1, Circle(1))) == \
            [Rotation(1, Circle(1))]

        assert expander.expand(Difference(Circle(1),
                                          Reference(Circle(1)))) == \
            [Circle(1), Difference(Circle(1), Reference(Circle(1)))]

        assert expander.expand(Difference(
            Circle(1),
            Reference(Rotation(1, Reference(Circle(1)))))) == \
            [Circle(1), Rotation(1, Reference(Circle(1))),
             Difference(Circle(1),
                        Reference(Rotation(1, Reference(Circle(1)))))]

    def test_unexpand(self):
        expander = Expander()
        assert expander.unexpand([Circle(1)]) == Circle(1)
        assert expander.unexpand([Rotation(1, Circle(1))]) == \
            Rotation(1, Circle(1))

        assert expander.unexpand(
            [Circle(1), Difference(Circle(1), Reference(Circle(1)))]
        ) == Difference(Circle(1), Reference(Circle(1)))

        assert expander.unexpand(
            [Circle(1), Rotation(1, Reference(Circle(1))),
             Difference(Circle(1),
                        Reference(Rotation(1, Reference(Circle(1)))))]
        ) == Difference(Circle(1),
                        Reference(Rotation(1, Reference(Circle(1)))))
