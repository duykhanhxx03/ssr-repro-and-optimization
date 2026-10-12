"""Independent re-scoring of the repo's cached sparse corpus.

Uses the corpus sparse codes cached by the repo pipeline (npz), re-encodes queries,
then computes exact sparse MaxSim with one GPU sparse matmul per query and nDCG@10
with a self-written implementation (linear gain, log2 discount, as in pytrec_eval /
BEIR). Agreement with the repo's number means its retriever and metrics are not the
source of any discrepancy.

Usage: python crosscheck.py --ckpt ckpt/SSR-tok --data-dir data/mteb --dataset nfcorpus \
           --corpus-npz cache/nfcorpus/corpus_32k_test.npz [--token-scope non-cls] [--extra-skip stopwords]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "SSR"))

from run_eval import load_checkpoint  # noqa: E402
from ssr.retrieval.mteb_io import load_mteb_split  # noqa: E402
from ssr.retrieval.sparse_repr import load_sparse_corpus  # noqa: E402


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus-npz", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--token-scope", default="non-cls")
    p.add_argument("--extra-skip", default=None)
    a = p.parse_args()
    dev = "cuda"
    torch.set_grad_enabled(False)

    split = load_mteb_split(a.data_dir.resolve(), a.dataset, a.split)
    doc_ids, docs = load_sparse_corpus(a.corpus_npz)
    assert doc_ids == split.corpus_ids, "cached corpus does not match split"
    n_lat = docs[0].n_latents

    # Doc-token sparse matrix (n_doc_tokens x n_lat) + owning doc index per row.
    rows, cols, vals, owner = [], [], [], []
    r = 0
    for di, e in enumerate(docs):
        for t in range(e.n_tokens):
            m = e.indices[t] >= 0
            k = int(m.sum())
            if k == 0:
                continue
            rows.append(np.full(k, r)); cols.append(e.indices[t][m]); vals.append(e.values[t][m])
            owner.append(di); r += 1
    D = torch.sparse_csr_tensor(
        torch.from_numpy(np.cumsum([0] + [len(x) for x in cols])).long(),
        torch.from_numpy(np.concatenate(cols)).long(),
        torch.from_numpy(np.concatenate(vals)).float(),
        size=(r, n_lat),
    ).to(dev)
    owner_t = torch.tensor(owner, device=dev)
    print(f"[crosscheck] {len(docs)} docs, {r} non-empty doc tokens, nnz={D.values().numel()}")

    model = load_checkpoint(a.ckpt.resolve(), dev, a.token_scope, a.extra_skip)
    topk = model.sae_module.topk
    scores_all = []
    for qi, qtext in enumerate(split.query_texts):
        q = model.encode([qtext], is_query=True, convert_to_tensor=True, show_progress_bar=False)[0]
        q = q.float().to(dev)
        # Keep each token's top-k positive activations and L2-normalise (as the repo does).
        v, idx = torch.topk(q, k=topk, dim=-1)
        v = torch.where(v > 1e-8, v, torch.zeros_like(v))
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qd = torch.zeros_like(q).scatter_(-1, idx, v)
        qd = qd[qd.abs().sum(-1) > 0]
        tok_scores = (D @ qd.T)  # (n_doc_tokens, n_q_tokens)
        per_doc = torch.full((len(docs), qd.shape[0]), 0.0, device=dev)
        per_doc.scatter_reduce_(0, owner_t[:, None].expand_as(tok_scores), tok_scores, reduce="amax")
        s = per_doc.sum(1)
        top = torch.topk(s, k=min(100, len(docs))).indices.tolist()
        scores_all.append([doc_ids[i] for i in top])

    # The repo's loader drops relevance grades; read them back for the official (graded) metric.
    graded: dict[str, dict[str, int]] = {}
    for line in (a.data_dir.resolve() / a.dataset / "qrels" / f"{a.split}.tsv").read_text(encoding="utf-8").split(chr(10)):
        if line.strip():
            qid, _, did, rel = line.split("\t")
            if int(rel) > 0:
                graded.setdefault(qid, {})[did] = int(rel)
    pairs = [(qid, ranked) for qid, ranked in zip(split.query_ids, scores_all) if graded.get(qid)]
    nd_bin = [ndcg_at_k(r, {d: 1 for d in graded[q]}) for q, r in pairs]
    nd_grd = [ndcg_at_k(r, graded[q]) for q, r in pairs]
    print(
        f"[crosscheck] {a.dataset}: nDCG@10 binary = {100 * sum(nd_bin) / len(nd_bin):.2f}, "
        f"graded (BEIR/MTEB) = {100 * sum(nd_grd) / len(nd_grd):.2f} over {len(pairs)} queries"
    )


if __name__ == "__main__":
    main()
