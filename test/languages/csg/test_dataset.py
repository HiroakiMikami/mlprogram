import unittest
import torch
import numpy as np
from mlprogram.interpreters import Reference as R
from mlprogram.languages.csg import Dataset


class TestDataset(unittest.TestCase):
    def test_iterator(self):
        dataset = Dataset(2, 1, 1, 1, 45)
        for x in dataset:
            break

    def test_multiprocess_loader(self):
        torch.manual_seed(0)
        np.random.seed(0)
        dataset = Dataset(2, 1, 2, 1, 45)
        loader = torch.utils.data.DataLoader(dataset, 2, num_workers=2,
                                             collate_fn=lambda x: x)
        samples = []
        for i, xs in enumerate(loader):
            samples.append(xs)
            if i == 1:
                break
        self.assertNotEqual(samples[0][0]["ground_truth"],
                            samples[1][0]["ground_truth"])

    def test_reference(self):
        torch.manual_seed(0)
        dataset = Dataset(2, 1, 3, 1, 45, reference=True)
        cnt = 0
        for x in dataset:
            cnt += 1
            if cnt == 10:
                sample = x
                break
        n_ref = len(sample["ground_truth"].statements)
        self.assertEqual(
            [R(str(i)) for i in range(n_ref)],
            [stmt.reference for stmt in sample["ground_truth"].statements]
        )


if __name__ == "__main__":
    unittest.main()
