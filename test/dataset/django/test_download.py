from mlprogram import Environment
from mlprogram.datasets.django import download


class TestDownload(object):
    def test_download(self):
        values = ["line0\nline1\nline2\n", "x = 10\nif True:\nelse:\n"]

        def get(path):
            return values.pop(0)
        dataset = download(get=get, num_train=1, num_test=1)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        valid_dataset = dataset["valid"]

        assert 1 == len(train_dataset)
        assert train_dataset[0] == Environment(
            inputs={"input": "line0"},
            supervisions={"ground_truth": "x = 10"}
        )

        assert 1 == len(test_dataset)
        assert test_dataset[0] == Environment(
            inputs={"input": "line1"},
            supervisions={"ground_truth": "if True:"}
        )

        assert 1 == len(valid_dataset)
        assert valid_dataset[0] == Environment(
            inputs={"input": "line2"},
            supervisions={"ground_truth": "else:"}
        )
