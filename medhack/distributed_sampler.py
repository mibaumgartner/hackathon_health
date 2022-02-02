from typing import Sequence, Iterator, Union

from torch.utils.data import Sampler
import torch.distributed as dist
import torch
from torch import Tensor


class WeightedDistributedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]``
    with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedDistributedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6],
            5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedDistributedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1],
            5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self, weights: Sequence[float], num_samples: int, replacement: bool = True,
            rank: Union[int, None] = None
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(num_samples)
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(replacement)
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        rank = dist.get_rank() if self.rank is None else self.rank
        g = torch.Generator()
        g.manual_seed(rank)
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
