"""Same-machine efficiency comparison on FiQA: ColBERTv2 + PLAID (fast-plaid) vs SSR-tok.

Measures, for each system: corpus encoding time, index build time, index size, per-query
search latency (batch of 1, after warm-up; p50/p95) and graded nDCG@10 on test/dev.
SSR variants: exact Eq. 4 (CPU numba kernel, one index over the corpus), SSR++ (paper
config, CPU) and exact on GPU (CSR matmul + segment max).

  python plaid_bench.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSR"))
sys.path.insert(0, str(ROOT / "opt"))

from crosscheck import ndcg_at_k  # noqa: E402
from quora_ssr import encode_sparse, read_corpus, read_split  # noqa: E402

DATASET = "fiqa"
SPLITS = ("test", "dev")
DATA = ROOT / "data" / "mteb" / DATASET
OUT = ROOT / "results" / "plaid_bench"
N_WARM = 20


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def lat_stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


def timed_search(fn, queries, q_ids, qrels, doc_ids_of) -> dict:
    for q in queries[:N_WARM]:
        fn(q)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms, nd = [], []
    for qid, q in zip(q_ids, queries):
        t0 = time.perf_counter()
        ranked_idx = fn(q)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t0))
        nd.append(ndcg_at_k(doc_ids_of(ranked_idx), qrels[qid]))
    return {"ndcg@10": 100 * float(np.mean(nd)), **lat_stats(ms), "n_queries": len(q_ids)}


def bench_plaid(doc_ids, doc_texts, splits) -> dict:
    from pylate import indexes, models

    res: dict = {"system": "ColBERTv2 + PLAID (fast-plaid, nbits=2)"}
    model = models.ColBERT(model_name_or_path="lightonai/colbertv2.0", device="cuda")
    t0 = time.time()
    D = model.encode(doc_texts, is_query=False, batch_size=64, show_progress_bar=False)
    res["encode_s"] = time.time() - t0
    folder = ROOT / "cache" / "plaid_bench"
    shutil.rmtree(folder, ignore_errors=True)
    # fast-plaid on CPU did not finish building the FiQA index in >50 min (k-means), so GPU only.
    for dev in ("cuda",):
        idx = indexes.PLAID(index_folder=str(folder), index_name=f"fiqa_{dev}", override=True,
                            nbits=2, device=dev, show_progress=False)
        t0 = time.time()
        idx.add_documents(documents_ids=doc_ids, documents_embeddings=D)
        res[f"index_build_s_{dev}"] = time.time() - t0
        res["index_bytes"] = dir_size(folder / f"fiqa_{dev}")
        print(f"[plaid] index {dev}: build {res[f'index_build_s_{dev}']:.1f}s, {res['index_bytes'] / 2**20:.0f} MiB, "
              f"encode {res['encode_s']:.0f}s", flush=True)
        for split, (q_ids, q_texts, qrels) in splits.items():
            Q = model.encode(q_texts, is_query=True, batch_size=64, show_progress_bar=False)

            def fn(q, idx=idx):
                return idx([q], k=100)[0]

            res[f"{split}_{dev}"] = timed_search(fn, Q, q_ids, qrels, lambda r: [x["id"] for x in r])
            print(f"[plaid] {split} {dev}: {res[f'{split}_{dev}']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def bench_ssr(doc_ids, doc_texts, splits) -> dict:
    """SSR-tok: encoding, index build/size, and search with the optimised exact kernel
    (fast_hyb), the LB-rerank approximation and the GPU exact scorer. The repo's own
    (single-threaded) kernels were timed by the optimisation run in repro/opt; they are
    not repeated here to keep peak RAM within the machine."""
    import gc

    from numba import get_num_threads
    from ssr.model import load_ssr_checkpoint
    from ssr.retrieval.inverted_index import build_chunk_index
    from ssr_opt.fast_exact import FastIndex, exact_topk_hyb, lb_rerank_topk

    res: dict = {"system": "SSR-tok"}
    model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device="cuda")
    n_lat = int(model.sae_module.n_latents)
    # Corpus codes: reuse the repo pipeline's cached encoding of FiQA (identical model and
    # settings); encoding throughput is timed on the first 5,000 docs and extrapolated
    # (batch 16: long FiQA docs x 16,384-d SAE outputs overflow 8 GB VRAM at batch 64).
    from ssr.retrieval.sparse_repr import load_sparse_corpus

    t0 = time.time()
    encode_sparse(model, doc_texts[:5000], is_query=False, batch_size=16)
    res["encode_s"] = (time.time() - t0) * len(doc_texts) / 5000
    res["encode_s_note"] = "extrapolated from 5,000 docs"
    cached_ids, docs = load_sparse_corpus(ROOT / "cache" / "fixed" / "fiqa" / "corpus_32k_test.npz")
    assert cached_ids == doc_ids
    enc_q = {s: encode_sparse(model, t, is_query=True, batch_size=128) for s, (_, t, _) in splits.items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    pos_of = lambda r: [doc_ids[i] for i in r]  # noqa: E731

    t0 = time.time()
    index = build_chunk_index(docs, n_latents=n_lat)
    res["repo_index_build_s_cpu"] = time.time() - t0
    c = index.compact
    res["repo_index_bytes"] = int(c.doc_idx.nbytes + c.token_idx.nbytes + c.values.nbytes + c.offsets.nbytes)
    res["n_postings"] = int(c.n_postings)
    del index, c
    gc.collect()

    t0 = time.time()
    fast = FastIndex.build(docs, shard_docs=32)  # blocks of 32 docs, as in repro/opt/bench.py
    fast.add_token_postings()
    res["opt_index_build_s_cpu"] = time.time() - t0
    res["opt_index_bytes"] = {k: int(v) for k, v in fast.nbytes().items()}
    res["opt_threads"] = get_num_threads()

    crow, cols, vals, seg = [0], [], [], np.zeros(len(docs), np.int64)
    for di, e in enumerate(docs):
        m = e.indices >= 0
        cnt = m.sum(axis=1)
        keep = cnt > 0
        if keep.any():
            cols.append(e.indices[m].astype(np.int64)); vals.append(e.values[m])
            crow.extend((crow[-1] + np.cumsum(cnt[keep])).tolist()); seg[di] = int(keep.sum())
    del docs
    Dg = torch.sparse_csr_tensor(torch.tensor(crow), torch.from_numpy(np.concatenate(cols)),
                                 torch.from_numpy(np.concatenate(vals)), size=(len(crow) - 1, n_lat)).cuda()
    seg_t = torch.from_numpy(seg).cuda()
    del cols, vals, crow
    gc.collect()

    def gpu_exact(q):
        rows = []
        for t in range(q.n_tokens):
            m = q.indices[t] >= 0
            if m.any():
                v = np.zeros(n_lat, np.float32); v[q.indices[t][m]] = q.values[t][m]; rows.append(v)
        Q = torch.from_numpy(np.stack(rows, 1)).cuda()
        M = torch.segment_reduce(Dg @ Q, "max", lengths=seg_t, axis=0, unsafe=True, initial=0.0)
        return torch.topk(M.sum(1), k=100).indices.tolist()

    for split, (q_ids, _, qrels) in splits.items():
        Q = enc_q[split]
        runs = {
            "opt_exact_cpu": lambda q: exact_topk_hyb(fast, q, top_docs=100)[0],
            "opt_lbrerank_t8_1000_cpu": lambda q: lb_rerank_topk(fast, q, top_docs=100, t_latents=8,
                                                                  n_candidates=1000)[0],
            "exact_gpu": gpu_exact,
        }
        for name, fn in runs.items():
            res[f"{split}_{name}"] = timed_search(fn, Q, q_ids, qrels, pos_of)
            print(f"[ssr] {split} {name}: {res[f'{split}_{name}']}", flush=True)
    return res


def main() -> None:
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    doc_ids, doc_texts = read_corpus(DATA / "corpus.jsonl")
    splits = {s: read_split(DATA, s) for s in SPLITS}
    which = sys.argv[1:] or ["plaid", "ssr"]
    meta = {"cpu_threads": os.cpu_count(), "gpu": torch.cuda.get_device_name(0), "dataset": DATASET,
            "n_docs": len(doc_ids)}
    for name in which:
        r = bench_plaid(doc_ids, doc_texts, splits) if name == "plaid" else bench_ssr(doc_ids, doc_texts, splits)
        r.update(meta)
        (OUT / f"{name}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
