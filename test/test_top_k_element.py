import unittest

from nl2prog._utils import TopKElement


class TestTopKElement(unittest.TestCase):
    def test_simple_case(self):
        topk = TopKElement(2)
        topk.add(1.0, "1")
        self.assertEqual([(1.0, "1")], topk.elements)
        topk.add(2.0, "2")
        self.assertEqual([(2.0, "2"), (1.0, "1")], topk.elements)
        topk.add(3.0, "3")
        self.assertEqual([(3.0, "3"), (2.0, "2")], topk.elements)
        topk.add(0.0, "0")
        self.assertEqual([(3.0, "3"), (2.0, "2")], topk.elements)


if __name__ == "__main__":
    unittest.main()
