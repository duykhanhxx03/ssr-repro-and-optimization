"""Correctness tests for the optimised SSR kernels (opt/ssr_opt).

Run:  python opt/tests/test_fast_exact.py      (no pytest in the venv; also pytest-compatible)

Checks, on random small corpora (many latent overlaps, duplicate docs => exact ties,
padded/invalid entries, empty queries, latents absent from the corpus):
  * every exhaustive kernel == dense reference Eq. 4 (allclose) and == baseline
    ``exact_doc_scores`` bit-for-bit when rows hold ascending latent ids (as SAE output does)
  * mu upper bound >= exact score for every doc; UB-pruned top-k == exhaustive top-k
  * SSR++ port == baseline ``ssr_plus_plus_topk`` (ids and scores identical)
  * baseline-order query segments reproduced exactly (incl. argpartition tie handling)
  * GPU scorer within 1e-5 of the reference (if CUDA is available)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssr.retrieval import exact_index_maxsim as B  # noqa: E402
from ssr.retrieval.inverted_index import build_chunk_index  # noqa: E402
from ssr.retrieval.sparse_repr import SparseTokenEmbeddings  # noqa: E402
from ssr_opt import fast_exact as F  # noqa: E402


def _random_sparse(rng, n_tokens, n_latents, k, pool, *, sorted_rows=True, pad_prob=0.2,
                   quantize=False):
    idx = np.full((n_tokens, k), -1, dtype=np.int32)
    val = np.zeros((n_tokens, k), dtype=np.float32)
    for t in range(n_tokens):
        kk = k if rng.random() > pad_prob else int(rng.integers(0, k + 1))
        if kk == 0:
            continue
        u = rng.choice(pool, size=kk, replace=False)
        if sorted_rows:
            u = np.sort(u)
        v = rng.random(kk).astype(np.float32) + 0.01
        if quantize:  # many equal values -> argpartition ties in the coarse step
            v = np.round(v * 4) / 4 + 0.05
        v /= np.linalg.norm(v)
        idx[t, :kk] = u
        val[t, :kk] = v
    return SparseTokenEmbeddings(indices=idx, values=val, n_latents=n_latents)


def _corpus(seed=0, n_docs=80, n_latents=96, k=6, sorted_rows=True, quantize=False):
    rng = np.random.default_rng(seed)
    pool = np.arange(n_latents // 2)  # small vocabulary => many overlaps
    docs = [_random_sparse(rng, int(rng.integers(1, 14)), n_latents, k, pool,
                           sorted_rows=sorted_rows, quantize=quantize) for _ in range(n_docs)]
    # exact duplicates -> exactly tied scores
    for i in range(0, 10, 2):
        docs[i + 1] = SparseTokenEmbeddings(docs[i].indices.copy(), docs[i].values.copy(),
                                            n_latents)
    qpool = np.arange(n_latents // 2 + 4)  # a few query latents never used by the corpus
    queries = [_random_sparse(rng, int(rng.integers(1, 9)), n_latents, k, qpool,
                              sorted_rows=sorted_rows, quantize=quantize) for _ in range(20)]
    queries.append(SparseTokenEmbeddings(np.full((3, k), -1, np.int32),
                                         np.zeros((3, k), np.float32), n_latents))
    return docs, queries


def _indexes(docs, shard_docs):
    base = build_chunk_index(docs, n_latents=docs[0].n_latents, block_size=16)
    fast = F.FastIndex.build(docs, with_token_postings=True, shard_docs=shard_docs)
    return base, fast


def _all_exact(fast, q):
    pq = fast.prepare(q, with_pairs=True)
    cand = np.arange(fast.n_docs)
    return {
        "lat": fast.exact_all_lat(pq),
        "hyb1": fast.exact_all_hyb(pq, dense_min=1),
        "hyb2_pad8": fast.exact_all_hyb(pq, dense_min=2, pad=8),
        "hyb3": fast.exact_all_hyb(pq, dense_min=3),
        "hyb_sparse": fast.exact_all_hyb(pq, dense_min=10**6),
        "tok": fast.exact_all_tok(pq),
        "dense": fast.exact_all_dense(pq),
        "dense0": fast.exact_all_dense0(pq),
        "fwd_par": fast.score_docs(pq, cand, parallel=True),
        "fwd_ser": fast.score_docs(pq, cand, parallel=False),
    }


def test_exhaustive_kernels_match_reference_and_baseline_bitwise():
    for shard_docs in (1, 3, 7, 1000):
        docs, queries = _corpus(seed=shard_docs)
        base, fast = _indexes(docs, shard_docs)
        for q in queries:
            want = np.array([B.reference_maxsim_dense(q, d) for d in docs])
            base_scores = B.exact_doc_scores(q, base)
            for name, got in _all_exact(fast, q).items():
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)
                assert np.array_equal(got, base_scores), (name, shard_docs)


def test_unsorted_rows_still_exact():
    docs, queries = _corpus(seed=11, sorted_rows=False)
    _, fast = _indexes(docs, 5)
    for q in queries:
        want = np.array([B.reference_maxsim_dense(q, d) for d in docs])
        for name, got in _all_exact(fast, q).items():
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)


def test_topk_drivers_match_baseline_topk():
    docs, queries = _corpus(seed=9)
    base, fast = _indexes(docs, 2)
    for q in queries:
        want = B.exact_maxsim_topk(q, base, top_docs=10)
        for fn in (F.exact_topk_hyb, F.exact_topk_lat, F.exact_topk_tok):
            got = fn(fast, q, top_docs=10)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_batched_hybrid_matches_single():
    docs, queries = _corpus(seed=10)
    base, fast = _indexes(docs, 3)
    pqs = [fast.prepare(q) for q in queries]
    for dm in (0, 1, 2, 10**6):
        S = fast.exact_batch_hyb(pqs, dense_min=dm)
        for b, q in enumerate(queries):
            assert np.array_equal(S[b], B.exact_doc_scores(q, base)), dm
    got = F.exact_topk_hyb_batch(fast, queries[:7], top_docs=10)
    for q, (ids, sc) in zip(queries[:7], got):
        want = B.exact_maxsim_topk(q, base, top_docs=10)
        assert np.array_equal(ids, want[0]) and np.array_equal(sc, want[1])


def test_upper_bound_and_pruned_topk():
    for seed in (2, 3):
        docs, queries = _corpus(seed=seed)
        base, fast = _indexes(docs, 4)
        for q in queries:
            pq = fast.prepare(q)
            exact = B.exact_doc_scores(q, base)
            if pq.qlat.size:
                ub, lin = fast.upper_bounds(pq, want_lin=True)
                assert np.all(ub >= exact) and np.all(lin >= exact)
                assert np.all(ub <= lin + 1e-6)
            for k in (1, 5, 10, 200):
                for fb, ch in ((0, 0), (2, 1), (k, 3)):
                    ids, sc = F.exact_topk_pruned(fast, q, top_docs=k, first_batch=fb, chunk=ch)
                    ref_ids, ref_sc = B._top(exact, k, 0.0)
                    assert np.array_equal(sc, ref_sc), (k, fb, ch)
                    # ids may differ only among exactly tied scores
                    for s in np.unique(sc):
                        if s == sc[-1]:
                            assert set(ids[sc == s]) <= set(np.flatnonzero(exact == s))
                        else:
                            assert set(ids[sc == s]) == set(ref_ids[ref_sc == s])


def test_ssrpp_port_identical_to_baseline():
    for seed, quant in ((4, False), (5, True)):
        docs, queries = _corpus(seed=seed, quantize=quant)
        base, fast = _indexes(docs, 3)
        for q in queries:
            for kc in (0, 1, 2, 4, 100):
                for n in (5, 17, len(docs)):
                    a = B.ssr_plus_plus_topk(q, base, top_docs=10, coarse_k=kc, n_candidates=n)
                    b = F.ssrpp_topk(fast, q, top_docs=10, coarse_k=kc, n_candidates=n)
                    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (kc, n)
                c_base = B.coarse_doc_scores(q, base, coarse_k=kc)
                c_fast = fast.coarse_scores(q, kc)
                assert np.array_equal(c_base, c_fast), kc


def test_query_segments_match_baseline_order():
    docs, queries = _corpus(seed=6, quantize=True)
    base, fast = _indexes(docs, 3)
    st = B._state(base)
    for q in queries:
        for kc in (0, 1, 3, 4, 6):
            qo, rows, v = B._query_segments(q, st, per_token_k=kc)
            qo2, lat2, v2 = F.pair_segments(np.asarray(q.indices), np.asarray(q.values),
                                            fast.lat_present, per_token_k=kc)
            assert np.array_equal(qo, qo2)
            assert np.array_equal(st.latent_ids[rows], lat2)
            assert np.array_equal(v, v2)


def test_lb_rerank_full_is_exact():
    docs, queries = _corpus(seed=7)
    base, fast = _indexes(docs, 4)
    for q in queries:
        exact = B.exact_doc_scores(q, base)
        ids, sc = F.lb_rerank_topk(fast, q, top_docs=10, t_latents=1000, n_candidates=len(docs))
        ref_ids, ref_sc = B._top(exact, 10, 0.0)
        assert np.array_equal(sc, ref_sc)
        # the truncated query score is a lower bound of the exact score
        pq = fast.prepare(q)
        if pq.qlat.size:
            ql, QT = F.truncate_query(pq.Qc, pq.qlat, 2)
            lb = fast.exact_all_lat(pq, qlat=ql, Qc=QT)
            assert np.array_equal(lb, fast.exact_all_hyb(pq, qlat=ql, Qc=QT))
            assert np.all(lb <= exact)


def test_gpu_matches_reference():
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if not torch.cuda.is_available():
        return
    from ssr_opt.gpu_exact import GpuExactIndex

    docs, queries = _corpus(seed=8)
    base, fast = _indexes(docs, 4)
    g = GpuExactIndex(fast)
    S = g.scores(queries).cpu().numpy()
    for qi, q in enumerate(queries):
        np.testing.assert_allclose(S[qi], B.exact_doc_scores(q, base), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
