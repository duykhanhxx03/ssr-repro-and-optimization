"""Exact Eq. 4 scoring on the GPU for large cached sparse corpora.

Doc-token sparse codes form one CSR matrix (rows = doc tokens, cols = latents); for a
batch of queries, ``D @ Q^T`` gives every token-token dot product, ``scatter_reduce``
takes the max over each document's tokens, and a sum over each query's tokens gives
S(Q, D). Same maths as the CPU kernel, validated on NFCorpus (must give 43.97).

  python gpu_exact_score.py --dataset quora --corpus-npz cache/fast/quora/corpus_ssr_tok.npz \
      --splits test,dev --ignore-identical-ids
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSR"))

from crosscheck import ndcg_at_k  # noqa: E402
from quora_ssr import encode_sparse, read_split  # noqa: E402
from ssr.model import load_ssr_checkpoint  # noqa: E402
from ssr.retrieval.sparse_repr import load_sparse_corpus  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus-npz", type=Path, required=True)
    p.add_argument("--splits", default="test")
    p.add_argument("--ignore-identical-ids", action="store_true")
    p.add_argument("--query-batch", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=0)
    a = p.parse_args()
    dev = "cuda"
    torch.set_grad_enabled(False)

    # Encode every split's queries first and release the encoder, so that the GPU only
    # holds the doc-token matrix while scoring (otherwise VRAM spills to system memory).
    model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device=dev)
    data = ROOT / "data" / "mteb" / a.dataset
    encoded = {}
    for split in a.splits.split(","):
        q_ids, q_texts, qrels = read_split(data, split)
        if a.max_queries:
            q_ids, q_texts = q_ids[: a.max_queries], q_texts[: a.max_queries]
        encoded[split] = (q_ids, qrels, encode_sparse(model, q_texts, is_query=True, batch_size=128))
    del model
    torch.cuda.empty_cache()

    t0 = time.time()
    doc_ids, docs = load_sparse_corpus(a.corpus_npz)
    n_lat = docs[0].n_latents
    crow, cols, vals, seg_len = [0], [], [], np.zeros(len(doc_ids), dtype=np.int64)
    for di, e in enumerate(docs):
        m = e.indices >= 0
        cnt = m.sum(axis=1)
        keep = cnt > 0
        if not keep.any():
            continue
        cols.append(e.indices[m].astype(np.int32))
        vals.append(e.values[m].astype(np.float32))
        crow.extend((crow[-1] + np.cumsum(cnt[keep])).tolist())
        seg_len[di] = int(keep.sum())
    del docs
    # int64 indices: with int32, every CSR matmul re-converted the 1.9 GB index on the GPU.
    D = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=torch.int64),
        torch.from_numpy(np.concatenate(cols).astype(np.int64)),
        torch.from_numpy(np.concatenate(vals)),
        size=(len(crow) - 1, n_lat),
    ).to(dev)
    del cols, vals, crow
    # Rows are grouped by document, so the per-document max is a segment max (no
    # materialised (rows x query-tokens) index tensor as scatter_reduce would need).
    seg_len_t = torch.from_numpy(seg_len).to(dev)
    n_docs = len(doc_ids)
    pos = {d: i for i, d in enumerate(doc_ids)}
    print(
        f"[gpu] {a.dataset}: {n_docs} docs, {D.shape[0]} doc tokens, nnz={D.values().numel()} "
        f"(load+build {time.time() - t0:.0f}s)",
        flush=True,
    )

    for split, (q_ids, qrels, queries) in encoded.items():
        t0 = time.time()
        nd = []
        for s in range(0, len(queries), a.query_batch):
            batch = queries[s : s + a.query_batch]
            cols_q, owners_q = [], []
            for bi, q in enumerate(batch):
                for t in range(q.n_tokens):
                    m = q.indices[t] >= 0
                    if m.any():
                        v = np.zeros(n_lat, dtype=np.float32)
                        v[q.indices[t][m]] = q.values[t][m]
                        cols_q.append(v)
                        owners_q.append(bi)
            if not cols_q:
                nd.extend(0.0 for _ in batch)
                continue
            Q = torch.from_numpy(np.stack(cols_q, axis=1)).to(dev)  # (n_lat, n_qtok)
            S = D @ Q  # (n_doc_tokens, n_qtok)
            M = torch.segment_reduce(S, "max", lengths=seg_len_t, axis=0, unsafe=True, initial=0.0)
            tot = torch.zeros((n_docs, len(batch)), device=dev)
            tot.index_add_(1, torch.tensor(owners_q, device=dev), M)
            for bi in range(len(batch)):
                qid = q_ids[s + bi]
                col = tot[:, bi]
                if a.ignore_identical_ids and qid in pos:
                    col[pos[qid]] = -1.0
                top = torch.topk(col, k=100).indices.tolist()
                nd.append(ndcg_at_k([doc_ids[i] for i in top], qrels[qid]))
            del S, M, tot, Q
            if (s // a.query_batch) % 500 == 0:
                print(
                    f"[gpu] {split}: {s + len(batch)}/{len(queries)} queries, "
                    f"{1000 * (time.time() - t0) / (s + len(batch)):.1f} ms/query",
                    flush=True,
                )
        print(
            f"[gpu] {a.dataset}/{split}: nDCG@10 graded = {100 * sum(nd) / len(nd):.2f} "
            f"({len(nd)} queries, {1000 * (time.time() - t0) / len(nd):.1f} ms/query)",
            flush=True,
        )


if __name__ == "__main__":
    main()
