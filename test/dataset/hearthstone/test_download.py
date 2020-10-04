import unittest

from mlprogram.datasets.hearthstone import download


class TestDownload(unittest.TestCase):
    def test_download(self):
        values = [
            "line0\n", "x = 10\n",
            "line1\n", "if True:§  pass\n",
            "line2\n", "if True and \\True:§  pass\n",
        ]

        def get(path):
            return values.pop(0)
        dataset = download(get=get)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        valid_dataset = dataset["valid"]

        self.assertEqual(1, len(train_dataset))
        self.assertEqual({"input": "line0", "ground_truth": "x = 10"},
                         train_dataset[0])

        self.assertEqual(1, len(test_dataset))
        self.assertEqual({"input": "line1",
                          "ground_truth": "if True:\n  pass"},
                         test_dataset[0])

        self.assertEqual(1, len(valid_dataset))
        self.assertEqual({"input": "line2",
                          "ground_truth": "if True and True:\n  pass"},
                         valid_dataset[0])


if __name__ == "__main__":
    unittest.main()
