"""Shuffle chronological blocks for I/O locality, include every train sample."""
import math
import random


class BlockDistributedBatchSampler:
    def __init__(self, size, batch_size, rank, world_size, seed=42):
        if size <= 0 or batch_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed sampler dimensions")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.size / (self.batch_size * self.world_size))

    @property
    def padded_presentations(self):
        return len(self) * self.batch_size * self.world_size - self.size

    def __iter__(self):
        blocks = [list(range(i, min(i + self.batch_size, self.size)))
                  for i in range(0, self.size, self.batch_size)]
        random.Random(self.seed + self.epoch).shuffle(blocks)
        order = [i for block in blocks for i in block]
        total = len(self) * self.batch_size * self.world_size
        order.extend((order * math.ceil((total - len(order)) / len(order)))[:total - len(order)])
        for step in range(getattr(self, "start_step", 0), len(self)):
            lo = (step * self.world_size + self.rank) * self.batch_size
            yield order[lo:lo + self.batch_size]


class StridedValidationSampler:
    def __init__(self, size, rank, world_size):
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
