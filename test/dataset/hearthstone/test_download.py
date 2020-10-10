from mlprogram.datasets.hearthstone import download


class TestDownload(object):
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

        assert 1 == len(train_dataset)
        assert {"input": "line0", "ground_truth": "x = 10"} == train_dataset[0]

        assert 1 == len(test_dataset)
        assert {"input": "line1",
                "ground_truth": "if True:\n  pass"} == test_dataset[0]

        assert 1 == len(valid_dataset)
        assert {"input": "line2",
                "ground_truth": "if True and True:\n  pass"} == \
            valid_dataset[0]
