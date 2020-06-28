import unittest
from mlprogram.utils.torch import StateDict


class TestStateDict(unittest.TestCase):
    def test_getitem(self):
        state_dict = {
            "x.x.w": 0,
            "x.x.b": 1,
            "x.y": 2,
            "z.w": 3
        }
        self.assertEqual(
            {
                "x.w": 0, "x.b": 1, "y": 2
            },
            StateDict(state_dict)["x"]
        )


if __name__ == "__main__":
    unittest.main()
