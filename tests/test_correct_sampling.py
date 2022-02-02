import unittest
from typing import List

import numpy as np

from medhack.distributed_sampler import WeightedDistributedRandomSampler


class CorrectSamplingTest(unittest.TestCase):
    def test_is_balanced(self):
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

        sampler_a = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=0
        )
        
        all_samples: List[int] = []
        for n, sample_id in enumerate(sampler_a):
            if n > n_sampling:
                break
            all_samples.append(all[sample_id])

        mean = float(np.mean(all_samples))
        print(f"Mean {mean}")
        self.assertAlmostEqual(0.5, mean, delta=0.05)

    def test_is_balanced_other_rank(self):
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
    
        sampler_a = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=10
            )
    
        all_samples: List[int] = []
        for n, sample_id in enumerate(sampler_a):
            if n > n_sampling:
                break
            all_samples.append(all[sample_id])
    
        mean = float(np.mean(all_samples))
        print(f"Mean {mean}")
        self.assertAlmostEqual(0.5, mean, delta=0.05)

    def test_is_different(self):
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
    
        sampler_a = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=0
            )
        
        sampler_b = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=1
            )
    
        all_sample_ids_a: List[int] = []
        for n, sample_id in enumerate(sampler_a):
            if n > n_sampling:
                break
            all_sample_ids_a.append(sample_id)
            
        all_sample_ids_b: List[int] = []
        for n, sample_id in enumerate(sampler_b):
            if n > n_sampling:
                break
            all_sample_ids_b.append(sample_id)
    
        self.assertNotEqual(all_sample_ids_a, all_sample_ids_b)

    def test_is_equal(self):
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
    
        sampler_a = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=0
            )
    
        sampler_b = WeightedDistributedRandomSampler(
            weights,
            num_samples=len(all),
            replacement=True,
            rank=0
            )
    
        all_sample_ids_a: List[int] = []
        for n, sample_id in enumerate(sampler_a):
            if n > n_sampling:
                break
            all_sample_ids_a.append(sample_id)
    
        all_sample_ids_b: List[int] = []
        for n, sample_id in enumerate(sampler_b):
            if n > n_sampling:
                break
            all_sample_ids_b.append(sample_id)
    
        self.assertEqual(all_sample_ids_a, all_sample_ids_b)


if __name__ == "__main__":
    unittest.main()
