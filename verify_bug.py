"""Reproduce the repo's index score by simulating its formula, to pin down the bug.

exact   : sum_t max_s sum_u q_t^u d_s^u            (true MaxSim, Eq. 4)
repo?   : sum_t max_u q_t^u * max_s d_s^u          (what _accumulate_segment_chunk computes)
eq12    : sum_t sum_u q_t^u * max_s d_s^u          (paper's coarse upper bound, Eq. 12)
"""
import sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / "SSR"))
from run_eval import load_checkpoint
from ssr.retrieval.metrics import RetrievalMetrics, compute_retrieval_metrics
from ssr.retrieval.mteb_io import load_mteb_split
from ssr.retrieval.retriever import RetrieverConfig, SparseMaxSimRetriever
from ssr.retrieval.sparse_repr import load_sparse_corpus

root = Path(__file__).resolve().parent
dataset, npz = sys.argv[1], Path(sys.argv[2])
torch.set_grad_enabled(False)
split = load_mteb_split(root / "data" / "mteb", dataset, "test")
doc_ids, docs = load_sparse_corpus(npz)
n_lat = docs[0].n_latents
# mu[d, u] = max over doc tokens of d_s^u
r, c, v = [], [], []
for di, e in enumerate(docs):
    m = e.indices >= 0
    lat, val = e.indices[m], e.values[m]
    best = {}
    for u, x in zip(lat.tolist(), val.tolist()):
        if x > best.get(u, 0.0): best[u] = x
    r += [di] * len(best); c += list(best.keys()); v += list(best.values())
MU = sp.csc_matrix((np.array(v, np.float32), (np.array(r), np.array(c))), shape=(len(docs), n_lat))

model = load_checkpoint(root / "ckpt" / "SSR-tok", "cpu", "non-cls", None)
queries = SparseMaxSimRetriever(RetrieverConfig(encode_batch_size=16)).encode_queries(
    model, split.query_texts, n_latents=n_lat, device="cpu", show_progress=False)

rk_repo, rk_eq12 = [], []
for q in queries:
    s_repo = np.zeros(len(docs), np.float32); s_eq12 = np.zeros(len(docs), np.float32)
    for t in range(q.n_tokens):
        m = q.indices[t] >= 0
        best_t = np.zeros(len(docs), np.float32)
        for u, qu in zip(q.indices[t][m], q.values[t][m]):
            col = MU[:, int(u)].toarray().ravel() * qu
            best_t = np.maximum(best_t, col); s_eq12 += col
        s_repo += best_t
    for s, out in ((s_repo, rk_repo), (s_eq12, rk_eq12)):
        top = np.argsort(-s)[:100]; out.append([(doc_ids[i], float(s[i])) for i in top])
M = RetrievalMetrics(accuracy_at_k=(10,), precision_recall_at_k=(10,), mrr_at_k=(), ndcg_at_k=(10,), map_at_k=(100,))
for name, rk in (("max-over-latents (repo formula?)", rk_repo), ("Eq.12 sum-over-latents", rk_eq12)):
    print(f"[verify_bug] {dataset} {name}: ndcg@10 = {compute_retrieval_metrics(query_ids=split.query_ids, ranked_results=rk, relevant_docs=split.relevant_docs, metrics=M)['ndcg@10']:.4f}")
