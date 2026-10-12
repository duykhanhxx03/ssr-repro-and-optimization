"""Paper-faithful sparse late-interaction scoring on the token-level inverted index.

Exact SSR score (paper Eq. 4)::

    S(Q, D) = sum_i max_j sum_{u in A(q_i) ∩ A(d_j)} q_i^u * d_j^u

Every posting keeps ``(doc_idx, token_idx, value)``, so for each query token the
dot products are accumulated per (doc, doc-token) row and only *then* maximised over
the document's tokens. (The previous index accumulators maximised each single
``q_i^u * d_j^u`` product over latents, i.e. ``sum_i max_u q_i^u * max_j d_j^u``, which
is not MaxSim.)

SSR++ (paper Sec. 3.3):

* Step 1, coarse (Eq. 11-12): ``S_coarse = sum_i sum_{u in A_Kcoarse(q_i)} q_i^u * mu_{D,u}``
  with ``mu_{D,u} = max_j d_j^u``; keep the top ``n_candidates`` documents.
* Step 2, exact refinement: Eq. 4 with the full query on the candidates only.

All values are non-negative (TopK + ReLU, then L2-normalised), so doc tokens that share
no latent with a query token contribute 0 and can be skipped without changing the max.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .sparse_repr import SparseTokenEmbeddings

try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:  # pragma: no cover - numba is in the [eval] extra
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def wrap(fn):
            return fn

        return wrap if not (args and callable(args[0])) else args[0]


@njit(cache=True)
def _exact_totals_kernel(
    offsets: np.ndarray,
    doc_idx: np.ndarray,
    token_idx: np.ndarray,
    values: np.ndarray,
    row_start: np.ndarray,
    row_doc: np.ndarray,
    qt_offsets: np.ndarray,
    seg_rows: np.ndarray,
    seg_vq: np.ndarray,
    allowed: np.ndarray,
    use_allowed: bool,
    acc: np.ndarray,
    row_gen: np.ndarray,
    touched_rows: np.ndarray,
    doc_best: np.ndarray,
    doc_gen: np.ndarray,
    touched_docs: np.ndarray,
    gen_state: np.ndarray,
    doc_total: np.ndarray,
) -> None:
    n_qt = qt_offsets.shape[0] - 1
    for qt in range(n_qt):
        gen_state[0] += 1
        gen = gen_state[0]
        n_rows = 0
        for k in range(qt_offsets[qt], qt_offsets[qt + 1]):
            li = seg_rows[k]
            vq = seg_vq[k]
            for p in range(offsets[li], offsets[li + 1]):
                d = doc_idx[p]
                if use_allowed and not allowed[d]:
                    continue
                r = row_start[d] + token_idx[p]
                if row_gen[r] != gen:
                    row_gen[r] = gen
                    acc[r] = 0.0
                    touched_rows[n_rows] = r
                    n_rows += 1
                acc[r] += vq * values[p]
        n_docs = 0
        for i in range(n_rows):
            r = touched_rows[i]
            d = row_doc[r]
            if doc_gen[d] != gen:
                doc_gen[d] = gen
                doc_best[d] = acc[r]
                touched_docs[n_docs] = d
                n_docs += 1
            elif acc[r] > doc_best[d]:
                doc_best[d] = acc[r]
        for i in range(n_docs):
            d = touched_docs[i]
            doc_total[d] += doc_best[d]


@njit(cache=True)
def _coarse_totals_kernel(
    mu_offsets: np.ndarray,
    mu_doc: np.ndarray,
    mu_val: np.ndarray,
    seg_rows: np.ndarray,
    seg_vq: np.ndarray,
    doc_total: np.ndarray,
) -> None:
    for k in range(seg_rows.shape[0]):
        li = seg_rows[k]
        vq = seg_vq[k]
        for p in range(mu_offsets[li], mu_offsets[li + 1]):
            doc_total[mu_doc[p]] += vq * mu_val[p]


class _ExactState:
    """Per-index layout and scratch buffers (built once, reused across queries)."""

    def __init__(self, index) -> None:
        compact = index.compact
        n_docs = int(index.n_docs)
        doc = compact.doc_idx.astype(np.int64, copy=False)
        tok = compact.token_idx.astype(np.int64, copy=False)
        n_tok = np.zeros(n_docs, dtype=np.int64)
        if doc.size:
            np.maximum.at(n_tok, doc, tok + 1)
        self.row_start = np.zeros(n_docs + 1, dtype=np.int64)
        np.cumsum(n_tok, out=self.row_start[1:])
        n_rows = int(self.row_start[-1])
        self.row_doc = np.repeat(np.arange(n_docs, dtype=np.int32), n_tok)
        self.doc_idx = compact.doc_idx.astype(np.int32, copy=False)
        self.token_idx = compact.token_idx.astype(np.int32, copy=False)
        self.values = compact.values.astype(np.float32, copy=False)
        self.offsets = compact.offsets.astype(np.int64, copy=False)
        self.latent_ids = compact.latent_ids
        self.latent_row_index = index.latent_row_index
        self.n_docs = n_docs
        self.n_latents = int(index.n_latents)
        self.acc = np.zeros(max(n_rows, 1), dtype=np.float32)
        self.row_gen = np.zeros(max(n_rows, 1), dtype=np.int64)
        self.touched_rows = np.zeros(max(n_rows, 1), dtype=np.int64)
        self.doc_best = np.zeros(max(n_docs, 1), dtype=np.float32)
        self.doc_gen = np.zeros(max(n_docs, 1), dtype=np.int64)
        self.touched_docs = np.zeros(max(n_docs, 1), dtype=np.int32)
        self.gen_state = np.zeros(1, dtype=np.int64)
        self._mu: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def latent_row(self, latent: int) -> int:
        if self.latent_row_index is not None:
            if 0 <= latent < self.latent_row_index.shape[0]:
                return int(self.latent_row_index[latent])
            return -1
        i = int(np.searchsorted(self.latent_ids, latent))
        if i < self.latent_ids.shape[0] and int(self.latent_ids[i]) == latent:
            return i
        return -1

    def mu_postings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``mu_{D,u} = max_j d_j^u`` per (latent row, doc), paper Eq. 11."""
        if self._mu is None:
            n_rows_lat = self.offsets.shape[0] - 1
            lat_of_posting = np.repeat(
                np.arange(n_rows_lat, dtype=np.int64), np.diff(self.offsets)
            )
            key = lat_of_posting * np.int64(self.n_docs + 1) + self.doc_idx.astype(np.int64)
            order = np.lexsort((-self.values, key))
            key_sorted = key[order]
            first = np.ones(key_sorted.shape[0], dtype=bool)
            first[1:] = key_sorted[1:] != key_sorted[:-1]
            sel = order[first]
            lat_sel = lat_of_posting[sel]
            mu_offsets = np.zeros(n_rows_lat + 1, dtype=np.int64)
            np.add.at(mu_offsets, lat_sel + 1, 1)
            np.cumsum(mu_offsets, out=mu_offsets)
            self._mu = (
                mu_offsets,
                self.doc_idx[sel].astype(np.int32),
                self.values[sel].astype(np.float32),
            )
        return self._mu


def _state(index) -> _ExactState:
    st = getattr(index, "_ssr_exact_state", None)
    if st is None or st.n_docs != int(index.n_docs):
        st = _ExactState(index)
        index._ssr_exact_state = st  # type: ignore[attr-defined]
    return st


def _query_segments(
    query: SparseTokenEmbeddings,
    st: _ExactState,
    *,
    per_token_k: int = 0,
    query_token_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(qt_offsets, latent rows, query values), optionally top ``per_token_k`` per token."""
    rows: list[int] = []
    vqs: list[float] = []
    qt_offsets = [0]
    for t in range(query.n_tokens):
        idx = query.indices[t]
        val = query.values[t]
        valid = idx >= 0
        idx, val = idx[valid], val[valid]
        if per_token_k > 0 and idx.size > per_token_k:
            keep = np.argpartition(-val, per_token_k - 1)[:per_token_k]
            idx, val = idx[keep], val[keep]
        w = 1.0 if query_token_weights is None else float(query_token_weights[t])
        for u, v in zip(idx.tolist(), val.tolist()):
            if v <= 0.0:
                continue
            li = st.latent_row(int(u))
            if li >= 0:
                rows.append(li)
                vqs.append(float(v) * w)
        qt_offsets.append(len(rows))
    return (
        np.asarray(qt_offsets, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(vqs, dtype=np.float32),
    )


def exact_doc_scores(
    query: SparseTokenEmbeddings,
    index,
    *,
    allowed: np.ndarray | None = None,
    query_token_weights: np.ndarray | None = None,
    query_latent_top_k: int = 0,
) -> np.ndarray:
    """Exact Eq. 4 score for every doc in ``index`` (0 for docs without overlap)."""
    st = _state(index)
    doc_total = np.zeros(max(st.n_docs, 1), dtype=np.float32)[: st.n_docs]
    if st.n_docs == 0 or query.n_tokens == 0:
        return doc_total
    qt_offsets, rows, vqs = _query_segments(
        query, st, per_token_k=query_latent_top_k, query_token_weights=query_token_weights
    )
    if rows.size == 0:
        return doc_total
    use_allowed = allowed is not None
    allowed_arr = (
        np.asarray(allowed, dtype=np.bool_) if use_allowed else np.zeros(1, dtype=np.bool_)
    )
    _exact_totals_kernel(
        st.offsets, st.doc_idx, st.token_idx, st.values, st.row_start, st.row_doc,
        qt_offsets, rows, vqs, allowed_arr, use_allowed,
        st.acc, st.row_gen, st.touched_rows, st.doc_best, st.doc_gen, st.touched_docs,
        st.gen_state, doc_total,
    )
    return doc_total


def coarse_doc_scores(
    query: SparseTokenEmbeddings,
    index,
    *,
    coarse_k: int,
    query_token_weights: np.ndarray | None = None,
) -> np.ndarray:
    """SSR++ step 1 (Eq. 12): sum over each query token's top-``coarse_k`` latents of q * mu."""
    st = _state(index)
    doc_total = np.zeros(st.n_docs, dtype=np.float32)
    if st.n_docs == 0 or query.n_tokens == 0:
        return doc_total
    _qt, rows, vqs = _query_segments(
        query, st, per_token_k=coarse_k, query_token_weights=query_token_weights
    )
    if rows.size:
        mu_offsets, mu_doc, mu_val = st.mu_postings()
        _coarse_totals_kernel(mu_offsets, mu_doc, mu_val, rows, vqs, doc_total)
    return doc_total


def _top(total: np.ndarray, top_docs: int, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.nonzero(total > max(float(min_score), 0.0))[0]
    if keep.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if top_docs > 0 and keep.size > top_docs:
        part = np.argpartition(-total[keep], top_docs - 1)[:top_docs]
        keep = keep[part]
    order = np.argsort(-total[keep], kind="stable")
    keep = keep[order]
    return keep.astype(np.int64), total[keep].astype(np.float32)


def exact_maxsim_topk(
    query: SparseTokenEmbeddings,
    index,
    *,
    top_docs: int,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
    query_latent_top_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """SSR: exact Eq. 4 over all documents; returns (doc_local_ids, scores) sorted desc."""
    total = exact_doc_scores(
        query,
        index,
        query_token_weights=query_token_weights,
        query_latent_top_k=query_latent_top_k,
    )
    return _top(total, top_docs, min_score)


def ssr_plus_plus_topk(
    query: SparseTokenEmbeddings,
    index,
    *,
    top_docs: int,
    coarse_k: int = 4,
    n_candidates: int = 2000,
    min_score: float = 0.0,
    query_token_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """SSR++: Eq. 12 coarse top-``n_candidates`` then exact Eq. 4 refinement on them."""
    coarse = coarse_doc_scores(
        query, index, coarse_k=coarse_k, query_token_weights=query_token_weights
    )
    cand, _ = _top(coarse, n_candidates, 0.0)
    if cand.size == 0:
        return cand, np.array([], dtype=np.float32)
    allowed = np.zeros(coarse.shape[0], dtype=np.bool_)
    allowed[cand] = True
    total = exact_doc_scores(
        query, index, allowed=allowed, query_token_weights=query_token_weights
    )
    return _top(total, top_docs, min_score)


def reference_maxsim_dense(
    query: SparseTokenEmbeddings, doc: SparseTokenEmbeddings
) -> float:
    """Dense reference implementation of Eq. 4 (for tests)."""
    n_lat = max(int(query.n_latents), int(doc.n_latents))

    def dense(s: SparseTokenEmbeddings) -> np.ndarray:
        m = np.zeros((s.n_tokens, n_lat), dtype=np.float64)
        for t in range(s.n_tokens):
            for j in range(s.k):
                u = int(s.indices[t, j])
                if u >= 0:
                    m[t, u] = float(s.values[t, j])
        return m

    q, d = dense(query), dense(doc)
    if q.size == 0 or d.size == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).clip(min=0.0).sum())
