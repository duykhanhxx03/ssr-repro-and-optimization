"""Exactness tests for the inverted-index SSR scorer (paper Eq. 4 / Eq. 12).

Run: python -m pytest tests/test_exact_index_maxsim.py   (or: python tests/test_exact_index_maxsim.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssr.retrieval.exact_index_maxsim import (  # noqa: E402
    coarse_doc_scores,
    exact_doc_scores,
    reference_maxsim_dense,
    ssr_plus_plus_topk,
)
from ssr.retrieval.inverted_index import build_chunk_index, exact_maxsim_via_index  # noqa: E402
from ssr.retrieval.sparse_repr import SparseTokenEmbeddings  # noqa: E402


def _random_sparse(rng, n_tokens, n_latents, k, active_latents=None) -> SparseTokenEmbeddings:
    idx = np.full((n_tokens, k), -1, dtype=np.int32)
    val = np.zeros((n_tokens, k), dtype=np.float32)
    pool = np.arange(n_latents) if active_latents is None else active_latents
    for t in range(n_tokens):
        kk = int(rng.integers(1, k + 1))
        u = rng.choice(pool, size=kk, replace=False)
        v = rng.random(kk).astype(np.float32) + 0.01
        v /= np.linalg.norm(v)
        idx[t, :kk] = u
        val[t, :kk] = v
    return SparseTokenEmbeddings(indices=idx, values=val, n_latents=n_latents)


def _corpus(seed=0, n_docs=60, n_latents=64, k=6):
    rng = np.random.default_rng(seed)
    active = np.arange(n_latents // 2)  # small vocabulary => many overlaps
    docs = [_random_sparse(rng, int(rng.integers(1, 12)), n_latents, k, active) for _ in range(n_docs)]
    queries = [_random_sparse(rng, int(rng.integers(1, 8)), n_latents, k, active) for _ in range(15)]
    return docs, queries


def test_exact_matches_dense_reference():
    docs, queries = _corpus()
    index = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    for q in queries:
        got = exact_doc_scores(q, index)
        want = np.array([reference_maxsim_dense(q, d) for d in docs])
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_repo_entry_point_is_exact():
    docs, queries = _corpus(seed=1)
    index = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    for q in queries:
        ids, scores = exact_maxsim_via_index(q, index, top_docs=10)
        want = np.array([reference_maxsim_dense(q, d) for d in docs])
        top_want = np.sort(want)[::-1][: len(ids)]
        np.testing.assert_allclose(np.sort(scores)[::-1], top_want, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scores, want[ids], rtol=1e-5, atol=1e-5)


def test_coarse_is_eq12():
    docs, queries = _corpus(seed=2)
    index = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    for q in queries:
        got = coarse_doc_scores(q, index, coarse_k=2)
        want = np.zeros(len(docs))
        for t in range(q.n_tokens):
            valid = q.indices[t] >= 0
            u, v = q.indices[t][valid], q.values[t][valid]
            top = np.argsort(-v)[:2]
            for uu, vv in zip(u[top], v[top]):
                for di, d in enumerate(docs):
                    m = d.indices == uu
                    if m.any():
                        want[di] += vv * d.values[m].max()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_ssrpp_with_full_pool_equals_exact():
    docs, queries = _corpus(seed=3)
    index = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    for q in queries:
        ids, scores = ssr_plus_plus_topk(q, index, top_docs=10, coarse_k=100, n_candidates=len(docs))
        want = np.array([reference_maxsim_dense(q, d) for d in docs])
        np.testing.assert_allclose(scores, want[ids], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scores, np.sort(want)[::-1][: len(ids)], rtol=1e-5, atol=1e-5)


def test_python_reference_agrees():
    from ssr.retrieval.inverted_index import maxsim_index_results_equal

    docs, queries = _corpus(seed=4)
    index = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    for q in queries:
        assert maxsim_index_results_equal(q, index, top_docs=0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
