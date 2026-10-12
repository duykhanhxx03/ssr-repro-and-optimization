"""Why safe UB pruning does not pay off for SSR: tightness of per-document upper bounds.

For each query: exact per-query-token MaxSim s_i(D), the 100th best exact score theta,
and several valid upper bounds UB(D) >= S(Q, D):
  mu        : sum_i min(c_i, sum_{u in A(q_i)} q_i^u mu_{D,u})             (Eq. 12 full K, capped)
  T=t       : sum_i min(c_i, P_i^t(D) + min(||q_i,R||, sum_{u in R_i} q_i^u mu_{D,u}))
              where P_i^t = exact MaxSim of token i restricted to its top-t latents
              (computing P^t already costs a large part of exhaustive scoring)
Reports the fraction of docs with UB >= theta (docs that would still need exact scoring)
and the mean slack per query token.  Output: opt/results/bound_tightness.json

Usage: python opt/analysis_bounds.py nfcorpus:test scifact:train fiqa:dev [--nq 40 --threads 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

OPT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPT))
from bench import CORPUS  # noqa: E402
from ssr_opt.fast_exact import FastIndex  # noqa: E402


@njit(parallel=True, cache=True)
def per_tok_best(tok_off, fw_lat, fw_val, qcode, Qc, out):
    n_docs = tok_off.shape[0] - 1
    nq = Qc.shape[1]
    K = fw_lat.shape[1]
    for d in prange(n_docs):
        acc = np.zeros(nq, np.float32)
        for j in range(tok_off[d], tok_off[d + 1]):
            acc[:] = 0
            for t in range(K):
                r = qcode[fw_lat[j, t]]
                if r >= 0:
                    for i in range(nq):
                        acc[i] += fw_val[j, t] * Qc[r, i]
            for i in range(nq):
                if acc[i] > out[d, i]:
                    out[d, i] = acc[i]


@njit(parallel=True, cache=True)
def per_tok_mu(mu_off, mu_doc, mu_val, L, n_shards, shard_size, n_docs, qlat, Qc, out):
    nq = Qc.shape[1]
    for s in prange(n_shards):
        for r in range(qlat.shape[0]):
            k = s * L + qlat[r]
            for p in range(mu_off[k], mu_off[k + 1]):
                d = mu_doc[p]
                for i in range(nq):
                    out[d, i] += mu_val[p] * Qc[r, i]


def analyse(fi, queries, k=100):
    res: dict = {}
    for q in queries:
        pq = fi.prepare(q)
        if pq.qlat.size == 0:
            continue
        nq_eff = int((pq.Qc > 0).any(axis=0).sum())
        S = np.zeros((fi.n_docs, pq.n_qt), np.float32)
        per_tok_best(fi.tok_off, fi.fw_lat, fi.fw_val, pq.qcode, pq.Qc, S)
        tot = S.sum(1)
        theta = np.sort(tot)[::-1][min(k, tot.size) - 1]
        top = tot >= theta
        res.setdefault("theta_per_token", []).append(theta / nq_eff)
        res.setdefault("mean_s_i_top100", []).append(S[top].sum(1).mean() / nq_eff)
        res.setdefault("mean_s_i_others", []).append(S[~top].sum(1).mean() / nq_eff)
        MU = np.zeros_like(S)
        per_tok_mu(fi.mu_off, fi.mu_doc, fi.mu_val, fi.n_latents, fi.n_shards, fi.shard_size,
                   fi.n_docs, pq.qlat, pq.Qc, MU)
        ub = np.minimum(MU, pq.cap).sum(1)
        res.setdefault("mu: frac_docs_ub>=theta", []).append(float((ub >= theta).mean()))
        res.setdefault("mu: slack_per_token", []).append(float((ub - tot).mean() / nq_eff))
        for T in (4, 8, 16):
            QT = np.zeros_like(pq.Qc)
            QR = pq.Qc.copy()
            for i in range(pq.n_qt):
                col = pq.Qc[:, i]
                nzr = np.flatnonzero(col)
                keep = nzr[np.argsort(-col[nzr])[:T]]
                QT[keep, i] = col[keep]
                QR[keep, i] = 0
            P = np.zeros_like(S)
            per_tok_best(fi.tok_off, fi.fw_lat, fi.fw_val, pq.qcode, QT, P)
            MR = np.zeros_like(S)
            per_tok_mu(fi.mu_off, fi.mu_doc, fi.mu_val, fi.n_latents, fi.n_shards, fi.shard_size,
                       fi.n_docs, pq.qlat, QR, MR)
            qr = (np.sqrt((QR.astype(np.float64) ** 2).sum(0)) * fi.max_tok_norm * 1.00001).astype(np.float32)
            ubt = np.minimum(P + np.minimum(MR, qr), pq.cap).sum(1)
            assert np.all(ubt >= tot - 1e-5)
            res.setdefault(f"T={T}: frac_docs_ub>=theta", []).append(float((ubt >= theta).mean()))
            res.setdefault(f"T={T}: slack_per_token", []).append(float((ubt - tot).mean() / nq_eff))
    return {k2: float(np.mean(v)) for k2, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--nq", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    set_num_threads(args.threads)
    from ssr.retrieval.sparse_repr import load_sparse_corpus

    out_path = OPT / "results" / "bound_tightness.json"
    out = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for spec in args.specs:
        ds, sp = spec.split(":")
        _, docs = load_sparse_corpus(CORPUS[ds])
        fi = FastIndex.build(docs)
        del docs
        _, qs = load_sparse_corpus(OPT / "qcache" / f"{ds}_{sp}.npz")
        r = analyse(fi, qs[: args.nq])
        r["n_queries"] = min(args.nq, len(qs))
        out[spec] = r
        print(spec, json.dumps(r, indent=1), flush=True)
        out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
