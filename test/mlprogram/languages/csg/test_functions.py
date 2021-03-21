from mlprogram.languages.csg import Dataset, IsSubtype, Parser, get_samples


class TestIsSubtype(object):
    def test_happy_path(self):
        assert IsSubtype()("CSG", "CSG")
        assert IsSubtype()("Circle", "CSG")

    def test_integer(self):
        assert IsSubtype()("int", "size")
        assert not IsSubtype()("int", "CSG")


class TestGetSamples(object):
    def test_not_reference(self):
        dataset = Dataset(1, 1, 1, 1, 45)

        samples = get_samples(dataset, Parser())
        assert 7 == len(samples.rules)
        assert 12 == len(samples.node_types)
        assert 9 == len(samples.tokens)

    def test_reference(self):
        dataset = Dataset(1, 1, 1, 1, 45)

        samples = get_samples(dataset, Parser(), reference=True)
        assert 7 == len(samples.rules)
        assert 12 == len(samples.node_types)
        assert 9 == len(samples.tokens)
