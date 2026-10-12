"""Third check: repo's query encoder + repo's metric, but brute-force exact MaxSim in scipy.

Isolates the repo's inverted-index retriever: if this matches crosscheck.py, the
retriever is what produces the low score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "SSR"))

from run_eval import load_checkpoint  # noqa: E402
from ssr.retrieval.metrics import RetrievalMetrics, compute_retrieval_metrics  # noqa: E402
from ssr.retrieval.mteb_io import load_mteb_split  # noqa: E402
from ssr.retrieval.retriever import RetrieverConfig, SparseMaxSimRetriever  # noqa: E402
from ssr.retrieval.sparse_repr import load_sparse_corpus  # noqa: E402

root = Path(__file__).resolve().parent
dataset, npz = sys.argv[1], Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
torch.set_grad_enabled(False)

split = load_mteb_split(root / "data" / "mteb", dataset, "test")
doc_ids, docs = load_sparse_corpus(npz)
assert doc_ids == split.corpus_ids

indptr, cols, vals, starts = [0], [], [], []
for e in docs:
    starts.append(len(indptr) - 1)
    for t in range(e.n_tokens):
        m = e.indices[t] >= 0
        cols.append(e.indices[t][m]); vals.append(e.values[t][m]); indptr.append(indptr[-1] + int(m.sum()))
n_lat = docs[0].n_latents
D = sp.csr_matrix((np.concatenate(vals), np.concatenate(cols), np.array(indptr)), shape=(len(indptr) - 1, n_lat))
starts = np.array(starts)

model = load_checkpoint(root / "ckpt" / "SSR-tok", device, "non-cls", None)
enc = SparseMaxSimRetriever(RetrieverConfig(encode_batch_size=16))
queries = enc.encode_queries(model, split.query_texts, n_latents=n_lat, device=device, show_progress=False)

ranked = []
for q in queries:
    Q = np.zeros((q.n_tokens, n_lat), dtype=np.float32)
    for t in range(q.n_tokens):
        m = q.indices[t] >= 0
        Q[t, q.indices[t][m]] = q.values[t][m]
    S = np.asarray((D @ sp.csr_matrix(Q).T).todense())  # (n_doc_tokens, n_q_tokens)
    per_doc = np.maximum.reduceat(S, starts, axis=0).sum(1)
    top = np.argsort(-per_doc)[:100]
    ranked.append([(doc_ids[i], float(per_doc[i])) for i in top])

m = compute_retrieval_metrics(
    query_ids=split.query_ids, ranked_results=ranked, relevant_docs=split.relevant_docs,
    metrics=RetrievalMetrics(accuracy_at_k=(10,), precision_recall_at_k=(10,), mrr_at_k=(), ndcg_at_k=(10,), map_at_k=(100,)),
)
print(f"[verify_scipy] {dataset}: repo-metric ndcg@10 = {m['ndcg@10']:.4f} (binary), queries={len(ranked)}")
