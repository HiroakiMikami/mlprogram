import unittest

from mlprogram.utils.data import Entry
from mlprogram.datasets.django import download


class TestDownload(unittest.TestCase):
    def test_download(self):
        values = ["line0\nline1\nline2\n", "x = 10\nif True:\nelse:\n"]

        def get(path):
            return values.pop(0)
        dataset = download(get=get, num_train=1, num_test=1)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        valid_dataset = dataset["valid"]

        self.assertEqual(1, len(train_dataset))
        self.assertEqual([Entry("line0", "x = 10")],
                         train_dataset[0])

        self.assertEqual(1, len(test_dataset))
        self.assertEqual([Entry("line1", "if True:")],
                         test_dataset[0])

        self.assertEqual(1, len(valid_dataset))
        self.assertEqual([Entry("line2", "else:")],
                         valid_dataset[0])


if __name__ == "__main__":
    unittest.main()
