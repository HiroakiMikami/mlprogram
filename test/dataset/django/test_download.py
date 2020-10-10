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
        assert {"input": "line0", "ground_truth": "x = 10"} == train_dataset[0]

        assert 1 == len(test_dataset)
        assert \
            {"input": "line1", "ground_truth": "if True:"} == test_dataset[0]

        assert 1 == len(valid_dataset)
        assert {"input": "line2", "ground_truth": "else:"} == valid_dataset[0]
