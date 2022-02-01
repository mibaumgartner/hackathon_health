import unittest
from typing import List

from torch.utils.data import WeightedRandomSampler
import numpy as np


class CorrectSamplingTest(unittest.TestCase):
    def test_something(self):
        n_sampling = 10000

        n_negatives = 1000
        n_positives = 100
        negatives = [0 for _ in range(n_negatives)]
        positives = [1 for _ in range(n_positives)]

        all = negatives + positives

        positives = sum(all)
        n_all = len(all)
        prob_positive = 1 - positives / n_all
        prob_negatives = positives / n_all

        weights = [prob_negatives if sample == 0 else prob_positive for sample in all]

        training_data_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
        )

        all_samples: List[int] = []
        for n, sample_id in enumerate(training_data_sampler):
            if n > n_sampling:
                break
            all_samples.append(all[sample_id])

        mean = float(np.mean(all_samples))
        print(f"Mean {mean}")
        self.assertAlmostEqual(0.5, mean, delta=0.05)

if __name__ == "__main__":
    unittest.main()
