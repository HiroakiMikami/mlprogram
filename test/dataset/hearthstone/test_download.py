import os
import tempfile

from mlprogram.builtins import Environment
from mlprogram.datasets.hearthstone import download


class TestDownload(object):
    def test_download(self):
        values = [
            "line0\n", "x = 10\n",
            "line1\n", "if True:§  pass\n",
            "line2\n", "if True and \\True:§  pass\n",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.pt")

            def get(path):
                return values.pop(0)
            dataset0 = download(get=get, cache_path=cache_path)

            def get2(path):
                raise NotImplementedError
            dataset1 = download(get=get2, cache_path=cache_path)
        train_dataset = dataset0["train"]
        test_dataset = dataset0["test"]
        valid_dataset = dataset0["valid"]

        assert 1 == len(train_dataset)
        assert list(train_dataset) == list(dataset1["train"])
        assert train_dataset[0] == Environment(
            {"text_query": "line0", "ground_truth": "x = 10"},
            set(["ground_truth"])
        )

        assert 1 == len(test_dataset)
        assert list(test_dataset) == list(dataset1["test"])
        assert test_dataset[0] == Environment(
            {"text_query": "line1", "ground_truth": "if True:\n  pass"},
            set(["ground_truth"])
        )

        assert 1 == len(valid_dataset)
        assert list(valid_dataset) == list(dataset1["valid"])
        assert valid_dataset[0] == Environment(
            {"text_query": "line2", "ground_truth": "if True and True:\n  pass"},
            set(["ground_truth"])
        )
