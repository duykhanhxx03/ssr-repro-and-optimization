"""Batch sampler that keeps all triplets of one query in the same batch.

The authors' training setup (GitHub issue #1, not described in the paper): 32 BM25 hard
negatives per query give 32 triplets per query, and the triplets of a query are grouped
into one batch (e.g. 2 queries x 32 triplets for batch size 64). Groups are the runs of
consecutive rows with the same ``query`` text, as produced by the triplet loader; their
order is shuffled every epoch, and groups are packed into batches without being split
unless a single group is larger than the batch.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from sentence_transformers.sampler import DefaultBatchSampler


class QueryGroupedBatchSampler(DefaultBatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        queries = dataset["query"]
        groups: list[list[int]] = []
        for i, q in enumerate(queries):
            if i > 0 and q == queries[i - 1]:
                groups[-1].append(i)
            else:
                groups.append([i])
        self.groups = groups

    def _batches(self) -> list[list[int]]:
        g = torch.Generator()
        g.manual_seed(int(self.seed) + int(self.epoch))
        order = torch.randperm(len(self.groups), generator=g).tolist()
        batches: list[list[int]] = []
        current: list[int] = []
        for gi in order:
            group = self.groups[gi]
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            while len(group) > self.batch_size:  # oversized group: split it
                batches.append(group[: self.batch_size])
                group = group[self.batch_size :]
            current = current + group
        if current and (not self.drop_last or len(current) == self.batch_size):
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
