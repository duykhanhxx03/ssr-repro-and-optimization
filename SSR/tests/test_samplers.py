"""QueryGroupedBatchSampler: whole query groups per batch, every row once per epoch.

Run: python tests/test_samplers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset  # noqa: E402

from ssr.samplers import QueryGroupedBatchSampler  # noqa: E402


def _dataset(sizes):
    rows = []
    for qi, n in enumerate(sizes):
        rows += [{"query": f"q{qi}", "positive": "p", "negative": f"n{j}"} for j in range(n)]
    return Dataset.from_list(rows)


def test_groups_stay_together_and_cover_all_rows():
    ds = _dataset([32] * 7 + [20, 5])
    s = QueryGroupedBatchSampler(ds, batch_size=64, drop_last=False, seed=3)
    for epoch in range(3):
        s.set_epoch(epoch)
        batches = list(iter(s))
        assert sorted(i for b in batches for i in b) == list(range(len(ds)))
        for b in batches:
            assert len(b) <= 64
            qs = [ds[i]["query"] for i in b]
            for q in set(qs):
                assert qs.count(q) == sum(1 for x in ds["query"] if x == q)


def test_epochs_reshuffle_and_oversized_groups_split():
    ds = _dataset([100, 10, 10])
    s = QueryGroupedBatchSampler(ds, batch_size=64, drop_last=False, seed=0)
    s.set_epoch(0)
    a = list(iter(s))
    s.set_epoch(1)
    b = list(iter(s))
    assert all(len(x) <= 64 for x in a + b)
    assert sorted(i for x in a for i in x) == list(range(len(ds)))
    assert [x[0] for x in a] != [x[0] for x in b] or len(a) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
