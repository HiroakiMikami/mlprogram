import unittest
import torch
from mlprogram.languages.csg import Dataset


class TestDataset(unittest.TestCase):
    def test_iterator(self):
        dataset = Dataset(2, 2, 1, 1, 45)
        for x in dataset:
            break

    def test_multiprocess_loader(self):
        dataset = Dataset(2, 2, 1, 1, 45)
        loader = torch.utils.data.DataLoader(dataset, 2, num_workers=2,
                                             collate_fn=lambda x: x)
        samples = []
        for i, xs in enumerate(loader):
            samples.append(xs)
            if i == 1:
                break
        self.assertNotEqual(samples[0][0][0], samples[1][0][0])
        self.assertNotEqual(samples[0][1][0], samples[1][1][0])


if __name__ == "__main__":
    unittest.main()
