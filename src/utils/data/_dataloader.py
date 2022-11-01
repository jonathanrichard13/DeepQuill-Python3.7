from random import shuffle

from . import Dataset
from ...nn.functional import stack

class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

    def __iter__(self):
        dataset_size: int = len(self.dataset)
        start_indices: list[int] = list(range(0, dataset_size, self.batch_size))
        if self.shuffle:
            shuffle(start_indices)
        for start_idx in start_indices:
            yield tuple(stack(column) for column in zip(*(self.dataset[idx] for idx in range(start_idx, min(start_idx + self.batch_size, dataset_size)))))