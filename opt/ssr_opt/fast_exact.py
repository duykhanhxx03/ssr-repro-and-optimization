"""Fast SSR retrieval: exact Eq. 4 top-k with safe upper-bound pruning, and SSR++.

Paper notation (Sec. 3.1, 3.3):

    S(Q, D)       = sum_i max_j sum_{u in A(q_i) & A(d_j)} q_i^u d_j^u          (Eq. 4)
    mu_{D,u}      = max_j d_j^u                                                (Eq. 11)
    S_coarse(Q,D) = sum_i sum_{u in A_Kc(q_i)} q_i^u mu_{D,u}                  (Eq. 12)

Safe upper bound used for exact top-k (all q, d >= 0, token vectors ~unit norm):

    <q_i, d_j> = sum_u q_i^u d_j^u <= sum_{u in A(q_i)} q_i^u mu_{D,u}   (0 <= d_j^u <= mu_{D,u})
    <q_i, d_j> <= ||q_i|| * ||d_j|| <= ||q_i|| * max_tok_norm =: c_i   (Cauchy-Schwarz)
    => S(Q, D) <= UB(Q, D) = sum_i min(c_i, sum_{u in A(q_i)} q_i^u mu_{D,u})

(the full-K Eq. 12 score is the same bound without the per-token cap). Documents are
scored exactly in decreasing UB order and the scan stops once the next UB is below the
current k-th best exact score, which yields the same top-k as exhaustive scoring.

Floating point: the forward kernel adds q_i^u d_j^u in ascending latent order, which is
also the order in which the baseline kernel adds them (SAE rows are stored with ascending
latent ids), so exact scores are bit-identical to ``ssr.retrieval.exact_index_maxsim``.
The UB kernel sums q_i^u mu_{D,u} in the same ascending order over a superset of
non-negative terms, so (by monotonicity of IEEE rounding) the float32 UB is >= the float32
exact score too; a tiny extra slack is added anyway.

Index layout (built once, ``FastIndex.build``):

* forward index: tok_off[n_docs+1], fw_lat int16[n_tok, K] (pad = n_latents),
  fw_val float32[n_tok, K]  -> score any given set of documents exactly.
* mu postings, split into contiguous doc-range shards, shard-major:
  key = shard * n_latents + latent -> run of (doc, mu_{D,u}) sorted by doc.
* optional token postings with the same sharding (exhaustive parallel kernel).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

F32 = np.float32


# --------------------------------------------------------------------------- build


@njit(parallel=True, cache=True)
def _count_by_shard(tok_off, fw_lat, n_latents, n_shards, shard_size, distinct, cnt):
    """cnt[s*L + u + 1] = #docs in shard s with latent u (distinct) or #token postings."""
    n_docs = tok_off.shape[0] - 1
    K = fw_lat.shape[1]
    for s in prange(n_shards):
        d0 = s * shard_size
        d1 = min(n_docs, d0 + shard_size)
        mark = np.full(n_latents, -1, np.int64)
        base = s * n_latents + 1
        for d in range(d0, d1):
            for j in range(tok_off[d], tok_off[d + 1]):
                for t in range(K):
                    u = fw_lat[j, t]
                    if u >= n_latents:
                        continue
                    if distinct:
                        if mark[u] != d:
                            mark[u] = d
                            cnt[base + u] += 1
                    else:
                        cnt[base + u] += 1


@njit(parallel=True, cache=True)
def _fill_mu(tok_off, fw_lat, fw_val, n_latents, n_shards, shard_size, off, mu_doc, mu_val):
    n_docs = tok_off.shape[0] - 1
    K = fw_lat.shape[1]
    for s in prange(n_shards):
        d0 = s * shard_size
        d1 = min(n_docs, d0 + shard_size)
        pos = off[s * n_latents : (s + 1) * n_latents].copy()
        maxv = np.zeros(n_latents, np.float32)
        seen = np.zeros(n_latents, np.bool_)
        touched = np.empty(n_latents, np.int32)
        for d in range(d0, d1):
            nt = 0
            for j in range(tok_off[d], tok_off[d + 1]):
                for t in range(K):
                    u = fw_lat[j, t]
                    if u >= n_latents:
                        continue
                    v = fw_val[j, t]
                    if not seen[u]:
                        seen[u] = True
                        maxv[u] = v
                        touched[nt] = u
                        nt += 1
                    elif v > maxv[u]:
                        maxv[u] = v
            for k in range(nt):
                u = touched[k]
                p = pos[u]
                mu_doc[p] = d
                mu_val[p] = maxv[u]
                pos[u] = p + 1
                seen[u] = False


@njit(parallel=True, cache=True)
def _fill_tok(tok_off, fw_lat, fw_val, n_latents, n_shards, shard_size, off, inv_row, inv_val):
    n_docs = tok_off.shape[0] - 1
    K = fw_lat.shape[1]
    for s in prange(n_shards):
        d0 = s * shard_size
        d1 = min(n_docs, d0 + shard_size)
        pos = off[s * n_latents : (s + 1) * n_latents].copy()
        for d in range(d0, d1):
            for j in range(tok_off[d], tok_off[d + 1]):
                for t in range(K):
                    u = fw_lat[j, t]
                    if u >= n_latents:
                        continue
                    p = pos[u]
                    inv_row[p] = j
                    inv_val[p] = fw_val[j, t]
                    pos[u] = p + 1


@njit(cache=True)
def _max_latent(fw_lat, pad):
    best = -1
    for j in range(fw_lat.shape[0]):
        for t in range(fw_lat.shape[1]):
            u = fw_lat[j, t]
            if u < pad and u > best:
                best = u
    return best


@njit(cache=True)
def _replace_pad(fw_lat, pad, new_pad):
    for j in range(fw_lat.shape[0]):
        for t in range(fw_lat.shape[1]):
            if fw_lat[j, t] == pad:
                fw_lat[j, t] = new_pad


@njit(cache=True)
def _row_norm_max(fw_val):
    best = 0.0
    for j in range(fw_val.shape[0]):
        s = 0.0
        for t in range(fw_val.shape[1]):
            x = np.float64(fw_val[j, t])
            s += x * x
        if s > best:
            best = s
    return math.sqrt(best)


# --------------------------------------------------------------------------- query prep


@njit(cache=True)
def _prep_dense(q_ind, q_val, lat_present, n_latents):
    """Distinct query latents (ascending), latent->code map and dense Qc[m, n_qt]."""
    nq, K = q_ind.shape
    qcode = np.full(n_latents + 1, -1, np.int32)
    for i in range(nq):
        for t in range(K):
            u = q_ind[i, t]
            if u < 0 or u >= n_latents or q_val[i, t] <= 0.0 or not lat_present[u]:
                continue
            qcode[u] = 0
    m = 0
    for u in range(n_latents):
        if qcode[u] == 0:
            m += 1
    qlat = np.empty(m, np.int32)
    c = 0
    for u in range(n_latents):
        if qcode[u] == 0:
            qcode[u] = c
            qlat[c] = u
            c += 1
    Qc = np.zeros((m, nq), np.float32)
    for i in range(nq):
        for t in range(K):
            u = q_ind[i, t]
            if u < 0 or u >= n_latents or q_val[i, t] <= 0.0 or not lat_present[u]:
                continue
            Qc[qcode[u], i] = q_val[i, t]
    return qlat, Qc, qcode


@dataclass
class PreparedQuery:
    n_qt: int
    qlat: np.ndarray  # int32[m] distinct query latents present in corpus, ascending
    Qc: np.ndarray  # float32[m, n_qt]
    qcode: np.ndarray  # int32[n_latents+1] latent -> row of Qc (or -1)
    cap: np.ndarray  # float32[n_qt]  c_i = ||q_i|| * max_tok_norm * (1 + slack)
    # baseline-order (qt, latent) pairs of the full query (exhaustive token kernel)
    qt_off: np.ndarray | None = None
    pair_lat: np.ndarray | None = None
    pair_v: np.ndarray | None = None


def pair_segments(q_ind, q_val, lat_present, per_token_k: int = 0):
    """(qt_offsets, latents, values) exactly as the baseline ``_query_segments`` builds them.

    Same filtering (idx >= 0, v > 0, latent present in corpus) and, for ``per_token_k``,
    the same ``np.argpartition`` call per token so ties are resolved identically.
    """
    n_qt, K = q_ind.shape
    L = lat_present.shape[0]
    if per_token_k <= 0 or per_token_k >= K:
        valid = (q_ind >= 0) & (q_ind < L) & (q_val > 0.0)
        safe = np.where(valid, q_ind, 0)
        valid &= lat_present[safe]
        counts = valid.sum(axis=1)
        qt_off = np.zeros(n_qt + 1, np.int64)
        np.cumsum(counts, out=qt_off[1:])
        return qt_off, q_ind[valid].astype(np.int64), q_val[valid].astype(np.float32)
    rows: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    qt_off = np.zeros(n_qt + 1, np.int64)
    full = (q_ind >= 0).all(axis=1)
    part_all = None
    if full.any():
        # identical to the per-row baseline call for rows without padding
        part_all = np.argpartition(-q_val[full], per_token_k - 1, axis=1)[:, :per_token_k]
    full_pos = np.cumsum(full) - 1
    for t in range(n_qt):
        idx = q_ind[t]
        val = q_val[t]
        if full[t]:
            keep = part_all[full_pos[t]]
            idx, val = idx[keep], val[keep]
        else:
            valid = idx >= 0
            idx, val = idx[valid], val[valid]
            if idx.size > per_token_k:
                keep = np.argpartition(-val, per_token_k - 1)[:per_token_k]
                idx, val = idx[keep], val[keep]
        ok = (val > 0.0) & (idx >= 0) & (idx < L)
        if idx.size:
            ok &= lat_present[np.where(ok, idx, 0)]
        rows.append(idx[ok].astype(np.int64))
        vals.append(val[ok].astype(np.float32))
        qt_off[t + 1] = qt_off[t] + int(ok.sum())
    if rows:
        return qt_off, np.concatenate(rows), np.concatenate(vals)
    return qt_off, np.zeros(0, np.int64), np.zeros(0, np.float32)


# --------------------------------------------------------------------------- kernels


@njit(parallel=True, cache=True, nogil=True)
def _ub_kernel(mu_off, mu_doc, mu_val, n_latents, n_shards, shard_size, n_docs,
               qlat, Qc, cap, ub_out, lin_out, want_lin):
    """UB[d] = sum_i min(c_i, sum_u q_i^u mu_{D,u}); optional uncapped (full-K Eq. 12)."""
    m = qlat.shape[0]
    nq = Qc.shape[1]
    for s in prange(n_shards):
        d0 = s * shard_size
        d1 = min(n_docs, d0 + shard_size)
        if d0 >= d1:
            continue
        nd = d1 - d0
        acc = np.zeros((nd, nq), np.float32)
        base = s * n_latents
        for r in range(m):
            k = base + qlat[r]
            for p in range(mu_off[k], mu_off[k + 1]):
                dl = mu_doc[p] - d0
                mv = mu_val[p]
                for i in range(nq):
                    acc[dl, i] += mv * Qc[r, i]
        for dl in range(nd):
            t = np.float32(0.0)
            for i in range(nq):
                a = acc[dl, i]
                t += a if a < cap[i] else cap[i]
            ub_out[d0 + dl] = t
            if want_lin:
                t2 = np.float32(0.0)
                for i in range(nq):
                    t2 += acc[dl, i]
                lin_out[d0 + dl] = t2


@njit(parallel=True, cache=True, nogil=True)
def _coarse_kernel(mu_off, mu_doc, mu_val, n_latents, n_shards, shard_size, n_docs,
                   pair_lat, pair_v, out):
    """Eq. 12 in the baseline's (qt, latent) addition order -> bit-identical coarse scores."""
    npairs = pair_lat.shape[0]
    for s in prange(n_shards):
        d0 = s * shard_size
        if d0 >= n_docs:
            continue
        base = s * n_latents
        for k in range(npairs):
            vq = pair_v[k]
            kk = base + pair_lat[k]
            for p in range(mu_off[kk], mu_off[kk + 1]):
                out[mu_doc[p]] += vq * mu_val[p]


@njit(cache=True, nogil=True)
def _score_docs_serial(cand, a, b, tok_off, fw_lat, fw_val, qcode, Qc, out):
    """Forward (doc-at-a-time) exact Eq. 4 for docs cand[a:b].  Per doc token, the
    latents hitting the query are first gathered branch-free, then accumulated in the
    token's (ascending) latent order -> bit-identical to the baseline kernel."""
    nq = Qc.shape[1]
    K = fw_lat.shape[1]
    acc = np.zeros(nq, np.float32)
    best = np.zeros(nq, np.float32)
    hidx = np.empty(K, np.int32)
    hval = np.empty(K, np.float32)
    for ci in range(a, b):
        d = cand[ci]
        for i in range(nq):
            best[i] = 0.0
        for j in range(tok_off[d], tok_off[d + 1]):
            nh = 0
            for t in range(K):
                r = qcode[fw_lat[j, t]]
                hidx[nh] = r
                hval[nh] = fw_val[j, t]
                nh += r >= 0
            if nh == 0:
                continue
            r = hidx[0]
            v = hval[0]
            for i in range(nq):
                acc[i] = v * Qc[r, i]
            for h in range(1, nh):
                r = hidx[h]
                v = hval[h]
                for i in range(nq):
                    acc[i] += v * Qc[r, i]
            for i in range(nq):
                if acc[i] > best[i]:
                    best[i] = acc[i]
        s = np.float32(0.0)
        for i in range(nq):
            s += best[i]
        out[ci] = s


@njit(parallel=True, cache=True, nogil=True)
def _score_docs_parallel(cand, tok_off, fw_lat, fw_val, qcode, Qc, out, n_chunks):
    n = cand.shape[0]
    step = (n + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        a = c * step
        b = min(n, a + step)
        if a < b:
            _score_docs_serial(cand, a, b, tok_off, fw_lat, fw_val, qcode, Qc, out)


@njit(parallel=True, cache=True, nogil=True)
def _exact_tok_kernel(inv_off, inv_row, inv_val, row_doc, tok_off, n_latents, n_shards,
                      shard_size, n_docs, qt_off, pair_lat, pair_v,
                      acc, stamp, touched, dbest, dstamp, dtouched, gen0, out):
    """Exhaustive Eq. 4 with the baseline's per-(doc, token) accumulation, sharded by doc
    range (parallel, cache-local).  Bit-identical to the baseline kernel."""
    nq = qt_off.shape[0] - 1
    for s in prange(n_shards):
        d0 = s * shard_size
        d1 = min(n_docs, d0 + shard_size)
        if d0 >= d1:
            continue
        r0 = tok_off[d0]
        base = s * n_latents
        for dl in range(d0, d1):
            out[dl] = 0.0
        for qt in range(nq):
            gen = gen0 + qt + 1
            nrows = r0
            for k in range(qt_off[qt], qt_off[qt + 1]):
                vq = pair_v[k]
                kk = base + pair_lat[k]
                for p in range(inv_off[kk], inv_off[kk + 1]):
                    r = inv_row[p]
                    if stamp[r] != gen:
                        stamp[r] = gen
                        acc[r] = 0.0
                        touched[nrows] = r
                        nrows += 1
                    acc[r] += vq * inv_val[p]
            nd = d0
            for ii in range(r0, nrows):
                r = touched[ii]
                d = row_doc[r]
                if dstamp[d] != gen:
                    dstamp[d] = gen
                    dbest[d] = acc[r]
                    dtouched[nd] = d
                    nd += 1
                elif acc[r] > dbest[d]:
                    dbest[d] = acc[r]
            for ii in range(d0, nd):
                d = dtouched[ii]
                out[d] += dbest[d]


@njit(parallel=True, cache=True, nogil=True)
def _exact_dense_kernel(inv_off, inv_row, inv_val, row_doc, tok_off, n_latents, n_shards,
                        shard_size, n_docs, max_rows, qlat, Qc, stamp, gen, n_workers, out):
    """Exhaustive Eq. 4, latent-major: every token posting of a query latent is visited
    once and updates all query tokens at once (acc[row, :] += d * Qc[latent, :]).

    Latents are processed in ascending order, so per (row, query token) the additions
    happen in the same order as in the baseline kernel -> bit-identical scores."""
    m = qlat.shape[0]
    nq = Qc.shape[1]
    for w in prange(n_workers):
        acc = np.empty((max_rows, nq), np.float32)
        tl = np.empty(max_rows, np.int32)
        best = np.empty((shard_size, nq), np.float32)
        for s in range(w, n_shards, n_workers):
            d0 = s * shard_size
            d1 = min(n_docs, d0 + shard_size)
            if d0 >= d1:
                continue
            r0 = tok_off[d0]
            base = s * n_latents
            nt = 0
            for r in range(m):
                k = base + qlat[r]
                for p in range(inv_off[k], inv_off[k + 1]):
                    row = inv_row[p]
                    rl = row - r0
                    v = inv_val[p]
                    if stamp[row] != gen:
                        stamp[row] = gen
                        tl[nt] = rl
                        nt += 1
                        for i in range(nq):
                            acc[rl, i] = v * Qc[r, i]
                    else:
                        for i in range(nq):
                            acc[rl, i] += v * Qc[r, i]
            nd = d1 - d0
            for dl in range(nd):
                for i in range(nq):
                    best[dl, i] = 0.0
            for x in range(nt):
                rl = tl[x]
                dl = row_doc[r0 + rl] - d0
                for i in range(nq):
                    a = acc[rl, i]
                    if a > best[dl, i]:
                        best[dl, i] = a
            for dl in range(nd):
                t = np.float32(0.0)
                for i in range(nq):
                    t += best[dl, i]
                out[d0 + dl] = t


@njit(parallel=True, cache=True, nogil=True)
def _exact_dense0_kernel(inv_off, inv_row, inv_val, tok_off, n_latents, n_shards,
                         shard_size, n_docs, max_rows, qlat, Qc, n_workers, out):
    """Like ``_exact_dense_kernel`` but without stamps: the shard accumulator is zeroed
    densely and the per-doc max runs over all rows (branch-free; best when most rows
    of a shard are touched). 0 + x == x, so results stay bit-identical."""
    m = qlat.shape[0]
    nq = Qc.shape[1]
    for w in prange(n_workers):
        acc = np.empty((max_rows, nq), np.float32)
        best = np.empty(nq, np.float32)
        for s in range(w, n_shards, n_workers):
            d0 = s * shard_size
            d1 = min(n_docs, d0 + shard_size)
            if d0 >= d1:
                continue
            r0 = tok_off[d0]
            nr = tok_off[d1] - r0
            for rl in range(nr):
                for i in range(nq):
                    acc[rl, i] = 0.0
            base = s * n_latents
            for r in range(m):
                k = base + qlat[r]
                for p in range(inv_off[k], inv_off[k + 1]):
                    rl = inv_row[p] - r0
                    v = inv_val[p]
                    for i in range(nq):
                        acc[rl, i] += v * Qc[r, i]
            for d in range(d0, d1):
                for i in range(nq):
                    best[i] = 0.0
                for rl in range(tok_off[d] - r0, tok_off[d + 1] - r0):
                    for i in range(nq):
                        a = acc[rl, i]
                        if a > best[i]:
                            best[i] = a
                t = np.float32(0.0)
                for i in range(nq):
                    t += best[i]
                out[d] = t


@njit(parallel=True, cache=True, nogil=True)
def _exact_lat_kernel(inv_off, inv_row, inv_val, tok_off, n_latents, n_shards,
                      shard_size, n_docs, max_rows, qlat, ql_off, ql_qt, ql_v, nq,
                      n_workers, out):
    """Latent-major with sparse query columns: each posting of a query latent u is
    visited once and updates only the query tokens containing u (ascending latent
    order per (row, token) -> bit-identical)."""
    m = qlat.shape[0]
    for w in prange(n_workers):
        acc = np.empty((max_rows, nq), np.float32)
        best = np.empty(nq, np.float32)
        for s in range(w, n_shards, n_workers):
            d0 = s * shard_size
            d1 = min(n_docs, d0 + shard_size)
            if d0 >= d1:
                continue
            r0 = tok_off[d0]
            nr = tok_off[d1] - r0
            for rl in range(nr):
                for i in range(nq):
                    acc[rl, i] = 0.0
            base = s * n_latents
            for r in range(m):
                k = base + qlat[r]
                a = ql_off[r]
                b = ql_off[r + 1]
                if b - a == 1:
                    i0 = ql_qt[a]
                    q0 = ql_v[a]
                    for p in range(inv_off[k], inv_off[k + 1]):
                        acc[inv_row[p] - r0, i0] += inv_val[p] * q0
                else:
                    for p in range(inv_off[k], inv_off[k + 1]):
                        rl = inv_row[p] - r0
                        v = inv_val[p]
                        for e in range(a, b):
                            acc[rl, ql_qt[e]] += v * ql_v[e]
            for d in range(d0, d1):
                for i in range(nq):
                    best[i] = 0.0
                for rl in range(tok_off[d] - r0, tok_off[d + 1] - r0):
                    for i in range(nq):
                        x = acc[rl, i]
                        if x > best[i]:
                            best[i] = x
                t = np.float32(0.0)
                for i in range(nq):
                    t += best[i]
                out[d] = t


@njit(parallel=True, cache=True, nogil=True)
def _exact_hyb_kernel(inv_off, inv_row, inv_val, tok_off, n_latents, n_shards,
                      shard_size, n_docs, max_rows, qlat, Qc, ql_off, ql_qt, ql_v, dense_min,
                      n_workers, out):
    """Hybrid of ``_exact_dense0_kernel`` and ``_exact_lat_kernel``: per query latent,
    a dense SIMD row update (acc[row, :] += d * Qc[latent, :]) when at least ``dense_min``
    query tokens contain the latent, otherwise scalar updates of just those tokens.
    Additions per (row, query token) stay in ascending latent order -> bit-identical."""
    m = qlat.shape[0]
    nq = Qc.shape[1]
    for w in prange(n_workers):
        acc = np.empty((max_rows, nq), np.float32)
        best = np.empty(nq, np.float32)
        for s in range(w, n_shards, n_workers):
            d0 = s * shard_size
            d1 = min(n_docs, d0 + shard_size)
            if d0 >= d1:
                continue
            r0 = tok_off[d0]
            nr = tok_off[d1] - r0
            for rl in range(nr):
                for i in range(nq):
                    acc[rl, i] = 0.0
            base = s * n_latents
            for r in range(m):
                k = base + qlat[r]
                a = ql_off[r]
                b = ql_off[r + 1]
                if b - a >= dense_min:
                    for p in range(inv_off[k], inv_off[k + 1]):
                        rl = inv_row[p] - r0
                        v = inv_val[p]
                        for i in range(nq):
                            acc[rl, i] += v * Qc[r, i]
                elif b - a == 1:
                    i0 = ql_qt[a]
                    q0 = ql_v[a]
                    for p in range(inv_off[k], inv_off[k + 1]):
                        acc[inv_row[p] - r0, i0] += inv_val[p] * q0
                else:
                    for p in range(inv_off[k], inv_off[k + 1]):
                        rl = inv_row[p] - r0
                        v = inv_val[p]
                        for e in range(a, b):
                            acc[rl, ql_qt[e]] += v * ql_v[e]
            for d in range(d0, d1):
                for i in range(nq):
                    best[i] = 0.0
                for rl in range(tok_off[d] - r0, tok_off[d + 1] - r0):
                    for i in range(nq):
                        x = acc[rl, i]
                        if x > best[i]:
                            best[i] = x
                t = np.float32(0.0)
                for i in range(nq):
                    t += best[i]
                out[d] = t


@njit(parallel=True, cache=True, nogil=True)
def _exact_hyb_batch_kernel(inv_off, inv_row, inv_val, tok_off, n_latents, n_shards,
                            shard_size, n_docs, max_rows, qlat, Qc, ql_off, ql_qt, ql_v,
                            dense_min, col_off, n_workers, out):
    """``_exact_hyb_kernel`` for a batch of queries: Qc holds the query tokens of all
    queries side by side (columns col_off[b]:col_off[b+1] belong to query b) over the
    union of their latents, so every posting is read once per batch.  out[b, d] is the
    sequential float32 sum of query b's token maxima -> bit-identical per query."""
    m = qlat.shape[0]
    nq = Qc.shape[1]
    nb = col_off.shape[0] - 1
    for w in prange(n_workers):
        acc = np.empty((max_rows, nq), np.float32)
        best = np.empty(nq, np.float32)
        for s in range(w, n_shards, n_workers):
            d0 = s * shard_size
            d1 = min(n_docs, d0 + shard_size)
            if d0 >= d1:
                continue
            r0 = tok_off[d0]
            nr = tok_off[d1] - r0
            for rl in range(nr):
                for i in range(nq):
                    acc[rl, i] = 0.0
            base = s * n_latents
            for r in range(m):
                k = base + qlat[r]
                a = ql_off[r]
                b = ql_off[r + 1]
                if b - a >= dense_min:
                    for p in range(inv_off[k], inv_off[k + 1]):
                        rl = inv_row[p] - r0
                        v = inv_val[p]
                        for i in range(nq):
                            acc[rl, i] += v * Qc[r, i]
                elif b - a == 1:
                    i0 = ql_qt[a]
                    q0 = ql_v[a]
                    for p in range(inv_off[k], inv_off[k + 1]):
                        acc[inv_row[p] - r0, i0] += inv_val[p] * q0
                else:
                    for p in range(inv_off[k], inv_off[k + 1]):
                        rl = inv_row[p] - r0
                        v = inv_val[p]
                        for e in range(a, b):
                            acc[rl, ql_qt[e]] += v * ql_v[e]
            for d in range(d0, d1):
                for i in range(nq):
                    best[i] = 0.0
                for rl in range(tok_off[d] - r0, tok_off[d + 1] - r0):
                    for i in range(nq):
                        x = acc[rl, i]
                        if x > best[i]:
                            best[i] = x
                for bq in range(nb):
                    t = np.float32(0.0)
                    for i in range(col_off[bq], col_off[bq + 1]):
                        t += best[i]
                    out[bq, d] = t


def _sparse_columns(Qc):
    """Per distinct latent row r of Qc: (qt, value) list in ascending qt order."""
    rr, ii = np.nonzero(Qc)
    off = np.zeros(Qc.shape[0] + 1, np.int64)
    np.cumsum(np.bincount(rr, minlength=Qc.shape[0]), out=off[1:])
    return off, ii.astype(np.int32), Qc[rr, ii].astype(np.float32)


# --------------------------------------------------------------------------- index


@dataclass
class FastIndex:
    n_docs: int
    n_latents: int
    K: int
    tok_off: np.ndarray
    fw_lat: np.ndarray
    fw_val: np.ndarray
    max_tok_norm: float
    n_shards: int
    shard_size: int
    mu_off: np.ndarray
    mu_doc: np.ndarray
    mu_val: np.ndarray
    lat_present: np.ndarray
    build_seconds: dict = field(default_factory=dict)
    # optional token postings (exhaustive sharded kernel)
    inv_off: np.ndarray | None = None
    inv_row: np.ndarray | None = None
    inv_val: np.ndarray | None = None
    row_doc: np.ndarray | None = None
    _scratch: tuple | None = None
    _gen: int = 0
    _stamp32: np.ndarray | None = None
    _max_rows: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, docs, *, n_latents: int | None = None, n_shards: int | None = None,
              with_token_postings: bool = False, shard_docs: int = 0) -> "FastIndex":
        t0 = time.perf_counter()
        n_docs = len(docs)
        n_latents = int(n_latents or docs[0].n_latents)
        if n_latents > np.iinfo(np.int16).max:
            raise ValueError("int16 forward latents need n_latents <= 32767")
        K = max(int(d.k) for d in docs)
        n_tok = np.array([d.n_tokens for d in docs], dtype=np.int64)
        tok_off = np.zeros(n_docs + 1, np.int64)
        np.cumsum(n_tok, out=tok_off[1:])
        tot = int(tok_off[-1])
        fw_lat = np.full((tot, K), n_latents, np.int16)
        fw_val = np.zeros((tot, K), np.float32)
        for d, e in enumerate(docs):
            a, b = tok_off[d], tok_off[d + 1]
            if b == a:
                continue
            ind = np.asarray(e.indices)
            val = np.asarray(e.values, dtype=np.float32)
            ok = (ind >= 0) & (val > 0.0)
            fw_lat[a:b, : e.k] = np.where(ok, ind, n_latents)
            fw_val[a:b, : e.k] = np.where(ok, val, 0.0)
        # latents never used by the corpus (the released SAE only uses 0..8191) are
        # trimmed so that the (shard, latent) offset tables stay small
        l_eff = max(int(_max_latent(fw_lat, n_latents)) + 1, 1)
        if l_eff < n_latents:
            _replace_pad(fw_lat, n_latents, l_eff)
            n_latents = l_eff
        t1 = time.perf_counter()
        idx = cls._finish(tok_off, fw_lat, fw_val, n_latents, n_shards, shard_docs,
                          with_token_postings)
        idx.build_seconds["forward"] = t1 - t0
        return idx

    @classmethod
    def _finish(cls, tok_off, fw_lat, fw_val, n_latents, n_shards, shard_docs,
                with_token_postings):
        from numba import get_num_threads

        n_docs = tok_off.shape[0] - 1
        if n_shards is None:
            if shard_docs > 0:
                n_shards = max(1, -(-n_docs // shard_docs))
            else:
                n_shards = max(1, min(n_docs, 8 * get_num_threads()))
        shard_size = max(1, -(-n_docs // n_shards))
        n_shards = max(1, -(-n_docs // shard_size))
        t0 = time.perf_counter()
        cnt = np.zeros(n_shards * n_latents + 1, np.int64)
        _count_by_shard(tok_off, fw_lat, n_latents, n_shards, shard_size, True, cnt)
        mu_off = np.cumsum(cnt)
        n_mu = int(mu_off[-1])
        mu_doc = np.empty(n_mu, np.int32)
        mu_val = np.empty(n_mu, np.float32)
        _fill_mu(tok_off, fw_lat, fw_val, n_latents, n_shards, shard_size, mu_off, mu_doc, mu_val)
        per_lat = cnt[1:].reshape(n_shards, n_latents).sum(axis=0)
        lat_present = per_lat > 0
        t1 = time.perf_counter()
        max_norm = float(_row_norm_max(fw_val))
        idx = cls(
            n_docs=n_docs, n_latents=n_latents, K=int(fw_lat.shape[1]), tok_off=tok_off,
            fw_lat=fw_lat, fw_val=fw_val, max_tok_norm=max_norm, n_shards=n_shards,
            shard_size=shard_size, mu_off=mu_off, mu_doc=mu_doc, mu_val=mu_val,
            lat_present=lat_present,
        )
        idx.build_seconds["mu"] = t1 - t0
        if with_token_postings:
            idx.add_token_postings()
        return idx

    def add_token_postings(self) -> None:
        t0 = time.perf_counter()
        L, S = self.n_latents, self.n_shards
        cnt = np.zeros(S * L + 1, np.int64)
        _count_by_shard(self.tok_off, self.fw_lat, L, S, self.shard_size, False, cnt)
        off = np.cumsum(cnt)
        n = int(off[-1])
        self.inv_row = np.empty(n, np.int32)
        self.inv_val = np.empty(n, np.float32)
        _fill_tok(self.tok_off, self.fw_lat, self.fw_val, L, S, self.shard_size, off,
                  self.inv_row, self.inv_val)
        self.inv_off = off
        n_tok = np.diff(self.tok_off)
        self.row_doc = np.repeat(np.arange(self.n_docs, dtype=np.int32), n_tok)
        self.build_seconds["token_postings"] = time.perf_counter() - t0

    def nbytes(self) -> dict:
        out = {
            "forward": self.tok_off.nbytes + self.fw_lat.nbytes + self.fw_val.nbytes,
            "mu": self.mu_off.nbytes + self.mu_doc.nbytes + self.mu_val.nbytes,
        }
        if self.inv_off is not None:
            out["token_postings"] = (self.inv_off.nbytes + self.inv_row.nbytes
                                     + self.inv_val.nbytes + self.row_doc.nbytes)
        return out

    # ------------------------------------------------------------------ queries
    def prepare(self, query, *, with_pairs: bool = False) -> PreparedQuery:
        q_ind = np.ascontiguousarray(query.indices, dtype=np.int32)
        q_val = np.ascontiguousarray(query.values, dtype=np.float32)
        qlat, Qc, qcode = _prep_dense(q_ind, q_val, self.lat_present, self.n_latents)
        qn = np.sqrt((Qc.astype(np.float64) ** 2).sum(axis=0))
        cap = (qn * self.max_tok_norm * (1.0 + 1e-5) + 1e-7).astype(np.float32)
        pq = PreparedQuery(n_qt=q_ind.shape[0], qlat=qlat, Qc=Qc, qcode=qcode, cap=cap)
        if with_pairs:
            pq.qt_off, pq.pair_lat, pq.pair_v = pair_segments(q_ind, q_val, self.lat_present)
        return pq

    # ------------------------------------------------------------------ scoring
    def upper_bounds(self, pq: PreparedQuery, *, want_lin: bool = False):
        ub = np.empty(self.n_docs, np.float32)
        lin = np.empty(self.n_docs if want_lin else 1, np.float32)
        _ub_kernel(self.mu_off, self.mu_doc, self.mu_val, self.n_latents, self.n_shards,
                   self.shard_size, self.n_docs, pq.qlat, pq.Qc, pq.cap, ub, lin, want_lin)
        return (ub, lin) if want_lin else ub

    def score_docs(self, pq: PreparedQuery, cand: np.ndarray, *, parallel: bool = True,
                   n_chunks: int = 0) -> np.ndarray:
        cand = np.ascontiguousarray(cand, dtype=np.int64)
        out = np.zeros(cand.shape[0], np.float32)
        if cand.size == 0 or pq.qlat.size == 0:
            return out
        if parallel and cand.size > 8:
            from numba import get_num_threads

            nc = n_chunks or min(cand.size, 8 * get_num_threads())
            _score_docs_parallel(cand, self.tok_off, self.fw_lat, self.fw_val, pq.qcode,
                                 pq.Qc, out, nc)
        else:
            _score_docs_serial(cand, 0, cand.size, self.tok_off, self.fw_lat, self.fw_val,
                               pq.qcode, pq.Qc, out)
        return out

    def coarse_scores(self, query, coarse_k: int) -> np.ndarray:
        q_ind = np.asarray(query.indices)
        q_val = np.asarray(query.values, dtype=np.float32)
        _, lat, v = pair_segments(q_ind, q_val, self.lat_present, per_token_k=coarse_k)
        out = np.zeros(self.n_docs, np.float32)
        if lat.size:
            _coarse_kernel(self.mu_off, self.mu_doc, self.mu_val, self.n_latents,
                           self.n_shards, self.shard_size, self.n_docs, lat, v, out)
        return out

    def exact_all_tok(self, pq: PreparedQuery) -> np.ndarray:
        """Exhaustive Eq. 4 for all docs via sharded token postings (parallel)."""
        if self.inv_off is None:
            self.add_token_postings()
        if self._scratch is None:
            n_rows = int(self.tok_off[-1])
            self._scratch = (
                np.zeros(max(n_rows, 1), np.float32),
                np.zeros(max(n_rows, 1), np.int64),
                np.zeros(max(n_rows, 1), np.int64),
                np.zeros(max(self.n_docs, 1), np.float32),
                np.zeros(max(self.n_docs, 1), np.int64),
                np.zeros(max(self.n_docs, 1), np.int64),
            )
        if pq.qt_off is None:
            raise ValueError("prepare(query, with_pairs=True) required")
        acc, stamp, touched, dbest, dstamp, dtouched = self._scratch
        out = np.zeros(self.n_docs, np.float32)
        _exact_tok_kernel(self.inv_off, self.inv_row, self.inv_val, self.row_doc, self.tok_off,
                          self.n_latents, self.n_shards, self.shard_size, self.n_docs,
                          pq.qt_off, pq.pair_lat, pq.pair_v, acc, stamp, touched, dbest,
                          dstamp, dtouched, self._gen, out)
        self._gen += pq.n_qt + 1
        return out

    def _ensure_rows(self):
        if self._max_rows == 0:
            starts = self.tok_off[: -1 : self.shard_size]
            ends = self.tok_off[np.minimum(np.arange(1, self.n_shards + 1) * self.shard_size,
                                           self.n_docs)]
            self._max_rows = max(int((ends - starts).max()) if starts.size else 1, 1)

    def exact_all_dense0(self, pq: PreparedQuery, n_workers: int = 0) -> np.ndarray:
        from numba import get_num_threads

        if self.inv_off is None:
            self.add_token_postings()
        self._ensure_rows()
        out = np.zeros(self.n_docs, np.float32)
        if pq.qlat.size == 0:
            return out
        _exact_dense0_kernel(self.inv_off, self.inv_row, self.inv_val, self.tok_off,
                             self.n_latents, self.n_shards, self.shard_size, self.n_docs,
                             self._max_rows, pq.qlat, pq.Qc, n_workers or get_num_threads(),
                             out)
        return out

    def exact_all_lat(self, pq: PreparedQuery, n_workers: int = 0, *, qlat=None,
                      Qc=None) -> np.ndarray:
        """Exhaustive Eq. 4 for all docs (latent-major, sparse query columns, parallel).
        ``qlat``/``Qc`` override the query (e.g. a truncated query for candidate LBs)."""
        from numba import get_num_threads

        if self.inv_off is None:
            self.add_token_postings()
        self._ensure_rows()
        qlat = pq.qlat if qlat is None else qlat
        Qc = pq.Qc if Qc is None else Qc
        out = np.zeros(self.n_docs, np.float32)
        if qlat.size == 0:
            return out
        off, qt, v = _sparse_columns(Qc)
        _exact_lat_kernel(self.inv_off, self.inv_row, self.inv_val, self.tok_off,
                          self.n_latents, self.n_shards, self.shard_size, self.n_docs,
                          self._max_rows, qlat, off, qt, v, Qc.shape[1],
                          n_workers or get_num_threads(), out)
        return out

    def exact_all_hyb(self, pq: PreparedQuery, n_workers: int = 0, *, qlat=None, Qc=None,
                      dense_min: int = 3, pad: int = 0) -> np.ndarray:
        """Exhaustive Eq. 4 (hybrid dense/sparse query-column kernel, parallel).
        ``pad`` > 0 pads the query-token axis to a multiple of ``pad`` (zero columns add
        +0 to every sum, so results are unchanged)."""
        from numba import get_num_threads

        if self.inv_off is None:
            self.add_token_postings()
        self._ensure_rows()
        qlat = pq.qlat if qlat is None else qlat
        Qc = pq.Qc if Qc is None else Qc
        out = np.zeros(self.n_docs, np.float32)
        if qlat.size == 0:
            return out
        if pad > 1 and Qc.shape[1] % pad:
            Qp = np.zeros((Qc.shape[0], Qc.shape[1] + pad - Qc.shape[1] % pad), np.float32)
            Qp[:, : Qc.shape[1]] = Qc
            Qc = Qp
        off, qt, v = _sparse_columns(Qc)
        _exact_hyb_kernel(self.inv_off, self.inv_row, self.inv_val, self.tok_off,
                          self.n_latents, self.n_shards, self.shard_size, self.n_docs,
                          self._max_rows, qlat, np.ascontiguousarray(Qc), off, qt, v,
                          int(dense_min), n_workers or get_num_threads(), out)
        return out

    def exact_batch_hyb(self, pqs, n_workers: int = 0, dense_min: int = 0) -> np.ndarray:
        """Exhaustive Eq. 4 for a batch of prepared queries -> float32[len(pqs), n_docs]."""
        from numba import get_num_threads

        if self.inv_off is None:
            self.add_token_postings()
        self._ensure_rows()
        out = np.zeros((len(pqs), self.n_docs), np.float32)
        lats = [p.qlat for p in pqs if p.qlat.size]
        if not lats:
            return out
        union = np.unique(np.concatenate(lats)).astype(np.int32)
        ncols = int(sum(p.n_qt for p in pqs))
        Q = np.zeros((union.size, max(ncols, 1)), np.float32)
        col_off = np.zeros(len(pqs) + 1, np.int64)
        c = 0
        for b, p in enumerate(pqs):
            if p.qlat.size:
                rows = np.searchsorted(union, p.qlat)
                Q[rows, c : c + p.n_qt] = p.Qc
            c += p.n_qt
            col_off[b + 1] = c
        off, qt, v = _sparse_columns(Q)
        dm = dense_min or max(3, ncols // 6)
        _exact_hyb_batch_kernel(self.inv_off, self.inv_row, self.inv_val, self.tok_off,
                                self.n_latents, self.n_shards, self.shard_size, self.n_docs,
                                self._max_rows, union, Q, off, qt, v, int(dm), col_off,
                                n_workers or get_num_threads(), out)
        return out

    def exact_all_dense(self, pq: PreparedQuery, n_workers: int = 0) -> np.ndarray:
        """Exhaustive Eq. 4 for all docs, latent-major dense-query kernel (parallel)."""
        from numba import get_num_threads

        if self.inv_off is None:
            self.add_token_postings()
        if self._stamp32 is None:
            self._stamp32 = np.zeros(max(int(self.tok_off[-1]), 1), np.int64)
            starts = self.tok_off[: -1 : self.shard_size]
            ends = self.tok_off[np.minimum(np.arange(1, self.n_shards + 1) * self.shard_size,
                                           self.n_docs)]
            self._max_rows = int((ends - starts).max()) if starts.size else 1
        out = np.zeros(self.n_docs, np.float32)
        if pq.qlat.size == 0:
            return out
        self._gen += 1
        _exact_dense_kernel(self.inv_off, self.inv_row, self.inv_val, self.row_doc,
                            self.tok_off, self.n_latents, self.n_shards, self.shard_size,
                            self.n_docs, max(self._max_rows, 1), pq.qlat, pq.Qc,
                            self._stamp32, self._gen, n_workers or get_num_threads(), out)
        return out


# --------------------------------------------------------------------------- top-k drivers


def top_sorted(total: np.ndarray, top_docs: int, min_score: float = 0.0):
    """Same selection as the baseline ``_top``: scores > min_score, top_docs, desc."""
    from ssr.retrieval.exact_index_maxsim import _top

    return _top(total, top_docs, min_score)


@dataclass
class SearchStats:
    n_scored: int = 0
    n_ub_positive: int = 0
    rounds: int = 0
    t_prep: float = 0.0
    t_ub: float = 0.0
    t_exact: float = 0.0
    t_select: float = 0.0


def exact_topk_pruned(index: FastIndex, query, *, top_docs: int = 100,
                      first_batch: int = 0, chunk: int = 0, slack: float = 1e-6,
                      stats: SearchStats | None = None, parallel: bool = True):
    """Exact Eq. 4 top-``top_docs`` with safe UB pruning (identical top-k to exhaustive)."""
    t0 = time.perf_counter()
    pq = index.prepare(query)
    t1 = time.perf_counter()
    if pq.qlat.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    ub = index.upper_bounds(pq)
    t2 = time.perf_counter()
    nz = np.flatnonzero(ub > 0.0)
    k = int(top_docs)
    first_batch = first_batch or max(2 * k, 256)
    chunk = chunk or max(k, 256)
    ub_nz = ub[nz]
    scored_docs: list[np.ndarray] = []
    scored_vals: list[np.ndarray] = []
    t_exact = 0.0
    rounds = 0
    if nz.size <= first_batch:
        order = nz[np.argsort(-ub_nz, kind="stable")]
        te = time.perf_counter()
        scored_docs.append(order)
        scored_vals.append(index.score_docs(pq, order, parallel=parallel))
        t_exact += time.perf_counter() - te
        rounds = 1
    else:
        part = np.argpartition(-ub_nz, first_batch - 1)
        head = part[:first_batch]
        head = head[np.argsort(-ub_nz[head], kind="stable")]
        te = time.perf_counter()
        sc = index.score_docs(pq, nz[head], parallel=parallel)
        t_exact += time.perf_counter() - te
        scored_docs.append(nz[head])
        scored_vals.append(sc)
        rounds = 1
        topk = np.sort(sc)[::-1][:k]
        theta = float(topk[-1]) if topk.size >= k else 0.0
        tail = part[first_batch:]
        ub_tail = ub_nz[tail]
        # docs that can still enter the top-k: UB (with slack) >= theta
        alive = ub_tail * (1.0 + slack) + slack >= theta
        tail = tail[alive]
        if tail.size:
            tail = tail[np.argsort(-ub_nz[tail], kind="stable")]
            pos = 0
            while pos < tail.size:
                if ub_nz[tail[pos]] * (1.0 + slack) + slack < theta:
                    break
                blk = tail[pos : pos + chunk]
                # drop the part of the block already below theta
                blk = blk[ub_nz[blk] * (1.0 + slack) + slack >= theta]
                te = time.perf_counter()
                sc = index.score_docs(pq, nz[blk], parallel=parallel)
                t_exact += time.perf_counter() - te
                scored_docs.append(nz[blk])
                scored_vals.append(sc)
                rounds += 1
                topk = np.concatenate([topk, sc])
                if topk.size > k:
                    topk = -np.partition(-topk, k - 1)[:k]
                theta = float(topk.min()) if topk.size >= k else 0.0
                pos += chunk
    docs = np.concatenate(scored_docs)
    vals = np.concatenate(scored_vals)
    total = np.zeros(index.n_docs, np.float32)
    total[docs] = vals
    out = top_sorted(total, k)
    t3 = time.perf_counter()
    if stats is not None:
        stats.n_scored += int(docs.size)
        stats.n_ub_positive += int(nz.size)
        stats.rounds += rounds
        stats.t_prep += t1 - t0
        stats.t_ub += t2 - t1
        stats.t_exact += t_exact
        stats.t_select += (t3 - t2) - t_exact
    return out


def exact_topk_fwd_all(index: FastIndex, query, *, top_docs: int = 100, parallel: bool = True):
    """Exhaustive exact Eq. 4 over every doc sharing a latent with the query (forward)."""
    pq = index.prepare(query)
    if pq.qlat.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    cand = np.arange(index.n_docs, dtype=np.int64)
    sc = index.score_docs(pq, cand, parallel=parallel)
    return top_sorted(sc, top_docs)


def exact_topk_tok(index: FastIndex, query, *, top_docs: int = 100):
    """Exhaustive exact Eq. 4 via sharded token postings (parallel inverted kernel)."""
    pq = index.prepare(query, with_pairs=True)
    total = index.exact_all_tok(pq)
    return top_sorted(total, top_docs)


def ssrpp_topk(index: FastIndex, query, *, top_docs: int = 100, coarse_k: int = 4,
               n_candidates: int = 2000, min_score: float = 0.0, parallel: bool = True):
    """SSR++ (Eq. 12 coarse top-n, exact Eq. 4 on candidates) - same output as the
    baseline ``ssr_plus_plus_topk`` (bit-identical coarse and exact scores)."""
    coarse = index.coarse_scores(query, coarse_k)
    cand, _ = top_sorted(coarse, n_candidates, 0.0)
    if cand.size == 0:
        return cand, np.zeros(0, np.float32)
    pq = index.prepare(query)
    sc = index.score_docs(pq, cand, parallel=parallel)
    total = np.zeros(index.n_docs, np.float32)
    total[cand] = sc
    return top_sorted(total, top_docs, min_score)


def exact_topk_lat(index: FastIndex, query, *, top_docs: int = 100):
    """Exhaustive exact Eq. 4 (latent-major sparse-column kernel, parallel); bit-identical
    scores to the baseline ``exact_doc_scores``."""
    pq = index.prepare(query)
    return top_sorted(index.exact_all_lat(pq), top_docs)


def exact_topk_hyb(index: FastIndex, query, *, top_docs: int = 100, dense_min: int = 3):
    """Exhaustive exact Eq. 4, hybrid dense/sparse query-column kernel (parallel);
    bit-identical scores to the baseline ``exact_doc_scores``.  Default exact method."""
    pq = index.prepare(query)
    return top_sorted(index.exact_all_hyb(pq, dense_min=dense_min), top_docs)


def truncate_query(Qc: np.ndarray, qlat: np.ndarray, t_latents: int):
    """Keep each query token's ``t_latents`` largest weights (rows of Qc = latents)."""
    QT = np.zeros_like(Qc)
    for i in range(Qc.shape[1]):
        col = Qc[:, i]
        nzr = np.flatnonzero(col)
        if nzr.size > t_latents:
            nzr = nzr[np.argsort(-col[nzr], kind="stable")[:t_latents]]
        QT[nzr, i] = col[nzr]
    keep = QT.any(axis=1)
    return np.ascontiguousarray(qlat[keep]), np.ascontiguousarray(QT[keep])


def lb_rerank_topk(index: FastIndex, query, *, top_docs: int = 100, t_latents: int = 4,
                   n_candidates: int = 1000, parallel: bool = True):
    """Approximate: candidates = top-``n_candidates`` by exact MaxSim of the query truncated
    to each token's top-``t_latents`` latents (a lower bound of Eq. 4), then exact Eq. 4
    refinement of the candidates (forward index)."""
    pq = index.prepare(query)
    if pq.qlat.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    ql, QT = truncate_query(pq.Qc, pq.qlat, t_latents)
    lb = index.exact_all_hyb(pq, qlat=ql, Qc=QT)
    cand, _ = top_sorted(lb, n_candidates, 0.0)
    sc = index.score_docs(pq, cand, parallel=parallel)
    total = np.zeros(index.n_docs, np.float32)
    total[cand] = sc
    return top_sorted(total, top_docs)


def exact_topk_hyb_batch(index: FastIndex, queries, *, top_docs: int = 100, dense_min: int = 0):
    """Batched exhaustive exact Eq. 4 (throughput mode): one pass over the token postings
    of the union of the batch's latents.  Same results as ``exact_topk_hyb`` per query."""
    pqs = [index.prepare(q) for q in queries]
    S = index.exact_batch_hyb(pqs, dense_min=dense_min)
    return [top_sorted(S[b], top_docs) for b in range(len(queries))]
