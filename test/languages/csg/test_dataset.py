import unittest
import torch
from mlprogram.utils import Reference as R, Token
from mlprogram.languages.csg import Dataset


class TestDataset(unittest.TestCase):
    def test_iterator(self):
        dataset = Dataset(2, 2, 1, 1, 45)
        for x in dataset:
            break

    def test_multiprocess_loader(self):
        torch.manual_seed(0)
        dataset = Dataset(2, 2, 1, 1, 45)
        loader = torch.utils.data.DataLoader(dataset, 2, num_workers=2,
                                             collate_fn=lambda x: x)
        samples = []
        for i, xs in enumerate(loader):
            samples.append(xs)
            if i == 1:
                break
        self.assertNotEqual(samples[0][0]["ground_truth"],
                            samples[1][0]["ground_truth"])

    def test_transform(self):
        dataset = Dataset(2, 2, 1, 1, 45, transform=lambda x: 0)
        for x in dataset:
            self.assertEqual(0, x)
            break

    def test_reference(self):
        torch.manual_seed(0)
        dataset = Dataset(2, 2, 3, 1, 45, reference=True)
        cnt = 0
        for x in dataset:
            cnt += 1
            if cnt == 10:
                sample = x
                break
        n_ref = len(sample["references"])
        self.assertEqual(
            [Token(None, R(str(i))) for i in range(n_ref)],
            sample["references"]
        )
        self.assertEqual(R(str(n_ref - 1)), sample["output_reference"])


if __name__ == "__main__":
    unittest.main()
