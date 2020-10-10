from mlprogram.collections import TopKElement


class TestTopKElement(object):
    def test_simple_case(self):
        topk = TopKElement(2)
        topk.add(1.0, "1")
        assert [(1.0, "1")] == topk.elements
        topk.add(2.0, "2")
        assert [(2.0, "2"), (1.0, "1")] == topk.elements
        topk.add(3.0, "3")
        assert [(3.0, "3"), (2.0, "2")] == topk.elements
        topk.add(0.0, "0")
        assert [(3.0, "3"), (2.0, "2")] == topk.elements

    def test_callback(self):
        class Callback:
            def __init__(self):
                self.elems = []

            def __call__(self, elem):
                self.elems.append(elem)

        callback = Callback()
        topk = TopKElement(2, callback)
        topk.add(1.0, "1")
        topk.add(2.0, "2")
        topk.add(3.0, "3")
        topk.add(0.0, "0")
        assert ["1", "0"] == callback.elems
