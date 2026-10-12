"""Benchmark harness: baseline vs optimised SSR retrieval on one dataset/split.

Loads the cached sparse corpus once, builds each needed index once, loads cached query
encodings (opt/encode_queries.py), warms up (JIT), then times every query individually.
Per run it writes opt/results/runs/<dataset>_<split>__<method>__t<threads>.{json,npz}
(metrics, latency stats, build time, RAM; per-query top-k ids/scores).  Exactness vs the
reference method is computed by opt/report.py from the npz files.

Methods (``name`` or ``name:key=val,key=val``):
  base_exact        repo exact_maxsim_topk, whole corpus as one index (single-threaded numba)
  base_ssrpp        repo ssr_plus_plus_topk, whole corpus (kc=4, n=2000 unless given)
  repo_exact        repo default eval path: ExactSparseMaxSimRetriever, 5000-doc chunks
  repo_ssrpp        repo default SSR++ path: SparseMaxSimRetriever (pruned), 5000-doc chunks
  fast_hyb          exact, hybrid dense/sparse latent-major sharded kernel (parallel) [bit-identical]
  fast_hyb_batch    exact, fast_hyb over a batch of bs queries at once (throughput mode)
  fast_lat          exact, latent-major sharded kernel, sparse query columns  [bit-identical]
  fast_tok          exact, per-query-token sharded kernel (parallel)    [bit-identical]
  fast_ub           exact, mu-bound ordered scoring with early stop     [identical top-k]
  fast_ssrpp        SSR++ port (mu coarse + forward exact refine)       [bit-identical to base_ssrpp]
  fast_lb           approx: truncated-query exact LB candidates (t, n) + exact refine
  gpu_exact         exact Eq. 4 on GPU (cuSPARSE SpMM + segment max), float32 ~1e-6

Usage:
  python opt/bench.py --dataset nfcorpus --split test --methods base_exact fast_lat --threads 6
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
OPT = REPRO / "opt"
os.environ.setdefault("TMP", str(OPT / "tmp"))
os.environ.setdefault("TEMP", str(OPT / "tmp"))
os.environ.setdefault("HF_HOME", str(REPRO / "hf_cache"))
sys.path.insert(0, str(OPT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

CORPUS = {
    "nfcorpus": REPRO / "cache/A/nfcorpus/corpus_32k_test.npz",
    "scifact": REPRO / "cache/A/scifact/corpus_32k_test.npz",
    "fiqa": REPRO / "cache/fixed/fiqa/corpus_32k_test.npz",
}
DATA = REPRO / "data" / "mteb"
RUNS = OPT / "results" / "runs"


def parse_method(spec: str):
    name, _, rest = spec.partition(":")
    kw = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        kw[k] = int(v) if v.lstrip("-").isdigit() else v
    return name, kw


def method_label(spec: str) -> str:
    return spec.replace(":", "-").replace(",", "-").replace("=", "")


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def peak_gb() -> float:
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1e9


def other_eval_procs() -> list[str]:
    out = []
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if any(k in cmd for k in ("eval_mteb", "run_eval", "eval_backbone", "eval_dense",
                                      "quora_", "colbert")):
                out.append(f"{p.info['pid']}: {cmd[:160]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def load_queries(dataset: str, split: str):
    from ssr.retrieval.sparse_repr import load_sparse_corpus

    qids, qs = load_sparse_corpus(OPT / "qcache" / f"{dataset}_{split}.npz")
    return qids, qs


def load_qrels(dataset: str, split: str):
    from ssr.retrieval.mteb_io import load_qrels_tsv

    return load_qrels_tsv(DATA / dataset / "qrels" / f"{split}.tsv")


def metrics_for(query_ids, ranked, qrels):
    from ssr.retrieval.metrics import RetrievalMetrics, compute_retrieval_metrics

    m = RetrievalMetrics(accuracy_at_k=(1, 10, 100), precision_recall_at_k=(1, 10, 100),
                         mrr_at_k=(), ndcg_at_k=(10,), map_at_k=(100,))
    return compute_retrieval_metrics(query_ids=query_ids, ranked_results=ranked,
                                     relevant_docs=qrels, metrics=m)


def lat_stats(lat_s: np.ndarray) -> dict:
    ms = lat_s * 1000.0
    return {"mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max()),
            "n_timed": int(ms.size)}


class Ctx:
    """Lazily built shared state for one dataset (corpus, indexes)."""

    def __init__(self, dataset: str, shard_docs: int):
        from ssr.retrieval.sparse_repr import load_sparse_corpus

        t0 = time.perf_counter()
        self.doc_ids, self.docs = load_sparse_corpus(CORPUS[dataset])
        self.load_seconds = time.perf_counter() - t0
        self.n_latents = int(self.docs[0].n_latents)
        self.shard_docs = shard_docs
        self._base = None
        self._base_build = None
        self._base_mu = None
        self._fast = None
        self._fast_build = None
        self._gpu = None

    def base_index(self):
        if self._base is None:
            from ssr.retrieval import exact_index_maxsim as B
            from ssr.retrieval.inverted_index import build_chunk_index

            t0 = time.perf_counter()
            self._base = build_chunk_index(self.docs, n_latents=self.n_latents, block_size=64)
            B._state(self._base)  # per-index layout + scratch (lazily built by the repo)
            self._base_build = time.perf_counter() - t0
        return self._base

    def base_mu(self):
        from ssr.retrieval import exact_index_maxsim as B

        idx = self.base_index()
        if self._base_mu is None:
            t0 = time.perf_counter()
            B._state(idx).mu_postings()
            self._base_mu = time.perf_counter() - t0
        return idx

    def fast_index(self, token_postings: bool):
        from ssr_opt.fast_exact import FastIndex

        if self._fast is None:
            t0 = time.perf_counter()
            self._fast = FastIndex.build(self.docs, with_token_postings=False,
                                         shard_docs=self.shard_docs)
            self._fast_build = {"forward+mu": time.perf_counter() - t0}
        if token_postings and self._fast.inv_off is None:
            t0 = time.perf_counter()
            self._fast.add_token_postings()
            self._fast._ensure_rows()
            self._fast_build["token_postings"] = time.perf_counter() - t0
        return self._fast

    def gpu_index(self, index_dtype: str = "int64"):
        if self._gpu is None or self._gpu.index_dtype != index_dtype:
            from ssr_opt.gpu_exact import GpuExactIndex

            self._gpu = None
            gc.collect()
            self._gpu = GpuExactIndex(self.fast_index(False), index_dtype=index_dtype)
        return self._gpu


def make_runner(ctx: Ctx, name: str, kw: dict, top_k: int):
    """Returns (per_query_fn | None, batch_fn | None, build_info, exact_flag)."""
    from ssr.retrieval import exact_index_maxsim as B

    if name == "base_exact":
        idx = ctx.base_index()
        fn = lambda q: B.exact_maxsim_topk(q, idx, top_docs=top_k)  # noqa: E731
        return fn, None, {"build_s": ctx._base_build}, "yes (reference)"
    if name == "base_ssrpp":
        idx = ctx.base_mu()
        kc, n = int(kw.get("kc", 4)), int(kw.get("n", 2000))
        fn = lambda q: B.ssr_plus_plus_topk(q, idx, top_docs=top_k, coarse_k=kc,  # noqa: E731
                                            n_candidates=n)
        return fn, None, {"build_s": ctx._base_build, "mu_build_s": ctx._base_mu}, "no (SSR++)"
    if name in ("fast_hyb", "fast_hyb_batch", "fast_lat", "fast_tok", "fast_ub", "fast_ssrpp",
                "fast_lb"):
        from ssr_opt import fast_exact as F

        fi = ctx.fast_index(token_postings=name in ("fast_hyb", "fast_hyb_batch", "fast_lat",
                                                    "fast_tok", "fast_lb"))
        info = {"build_s": float(sum(ctx._fast_build.values())), "build_parts": ctx._fast_build,
                "index_mb": {k: v / 1e6 for k, v in fi.nbytes().items()},
                "n_shards": fi.n_shards, "shard_docs": fi.shard_size}
        if name == "fast_hyb":
            dm = int(kw.get("dm", 3))
            return (lambda q: F.exact_topk_hyb(fi, q, top_docs=top_k, dense_min=dm)), None, info,                 "yes (bit-identical)"
        if name == "fast_hyb_batch":
            info["batch"] = int(kw.get("bs", 8))
            return None, (lambda qs: F.exact_topk_hyb_batch(fi, qs, top_docs=top_k)), info,                 "yes (bit-identical)"
        if name == "fast_lat":
            return (lambda q: F.exact_topk_lat(fi, q, top_docs=top_k)), None, info, "yes (bit-identical)"
        if name == "fast_tok":
            return (lambda q: F.exact_topk_tok(fi, q, top_docs=top_k)), None, info, "yes (bit-identical)"
        if name == "fast_ub":
            return (lambda q: F.exact_topk_pruned(fi, q, top_docs=top_k)), None, info, "yes (safe pruning)"
        if name == "fast_ssrpp":
            kc, n = int(kw.get("kc", 4)), int(kw.get("n", 2000))
            return (lambda q: F.ssrpp_topk(fi, q, top_docs=top_k, coarse_k=kc, n_candidates=n)), \
                None, info, "no (SSR++)"
        if name == "fast_lb":
            t, n = int(kw.get("t", 4)), int(kw.get("n", 1000))
            return (lambda q: F.lb_rerank_topk(fi, q, top_docs=top_k, t_latents=t, n_candidates=n)), \
                None, info, "no (approx)"
    if name == "gpu_exact":
        g = ctx.gpu_index("int32" if int(kw.get("i32", 0)) else "int64")
        bs = int(kw.get("bs", 1))
        info = {"build_s": g.build_seconds + float(sum(ctx._fast_build.values())),
                "vram_mb": g.vram_bytes() / 1e6, "batch": bs}

        def batch_fn(qs):
            return g.topk(qs, top_docs=top_k)

        return None, batch_fn, info, "yes (float32, ~1e-6)"
    if name in ("repo_exact", "repo_ssrpp"):
        return None, None, {}, "yes (reference)" if name == "repo_exact" else "no (SSR++)"
    raise ValueError(f"unknown method {name}")


def run_repo_default(ctx: Ctx, name: str, kw: dict, queries, top_k: int):
    """Run the repo's own batch retriever (as eval_mteb does) and return per-query lists."""
    from ssr.retrieval.retriever import RetrieverConfig, SparseMaxSimRetriever
    from ssr.retrieval.exact_retriever import ExactSparseMaxSimRetriever
    from ssr.retrieval.sparse_repr import prune_sparse_rows

    id_to_idx = {d: i for i, d in enumerate(ctx.doc_ids)}
    t0 = time.perf_counter()
    if name == "repo_exact":
        r = ExactSparseMaxSimRetriever(RetrieverConfig(mode="exact", corpus_chunk_size=5000,
                                                       block_size=64))
        ranked = r.retrieve(queries=queries, corpus=ctx.docs, corpus_ids=ctx.doc_ids,
                            top_k=top_k, score_device="cpu", show_progress=False)
        timing = r.last_retrieval_timing.as_dict()
    else:
        kc, n = int(kw.get("kc", 4)), int(kw.get("n", 2000))
        r = SparseMaxSimRetriever(RetrieverConfig(mode="pruned", corpus_chunk_size=5000,
                                                  block_size=64, prune_topk=kc,
                                                  n_candidates=n))
        qp = [prune_sparse_rows(q, kc) for q in queries]
        ranked = r.retrieve(query_sparse=qp, query_sparse_fine=queries,
                            corpus_sparse=ctx.docs, corpus_ids=ctx.doc_ids, top_k=top_k,
                            show_progress=False)
        timing = {}
    wall = time.perf_counter() - t0
    out = []
    for hits in ranked:
        hits = sorted(hits, key=lambda x: x[1], reverse=True)
        out.append((np.array([id_to_idx[c] for c, _ in hits], np.int64),
                    np.array([s for _, s in hits], np.float32)))
    return out, wall, timing


def _threading_layer():
    import numba

    try:
        return numba.threading_layer()
    except ValueError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[6])
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--shard-docs", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    import numba

    RUNS.mkdir(parents=True, exist_ok=True)
    others = other_eval_procs()
    if others:
        print("WARNING: other eval processes running:\n  " + "\n  ".join(others))
    qids, queries = load_queries(args.dataset, args.split)
    if args.max_queries:
        qids, queries = qids[: args.max_queries], queries[: args.max_queries]
    qrels = load_qrels(args.dataset, args.split)
    ctx = Ctx(args.dataset, args.shard_docs)
    print(f"{args.dataset}:{args.split} docs={len(ctx.docs)} queries={len(queries)} "
          f"load={ctx.load_seconds:.1f}s rss={rss_gb():.2f}GB", flush=True)

    for spec in args.methods:
        name, kw = parse_method(spec)
        thread_list = args.threads if name.startswith("fast_") else [1 if name.startswith(("base_", "repo_")) else 0]
        for th in thread_list:
            if th > 0:
                numba.set_num_threads(th)
            gc.collect()
            t_build0 = time.perf_counter()
            per_q, batch_fn, info, exact = make_runner(ctx, name, kw, args.top_k)
            build_wall = time.perf_counter() - t_build0
            if ctx.docs is not None and ctx._fast is not None and not any(
                    parse_method(m)[0].startswith(("base_", "repo_")) for m in args.methods):
                ctx.docs = None  # the fast/GPU indexes no longer need the corpus list
                gc.collect()
            lat = None
            extra = {}
            if name.startswith("repo_"):
                results, wall, timing = run_repo_default(ctx, name, kw, queries, args.top_k)
                lat = None
                extra = {"wall_s": wall, "repo_timing": timing,
                         "mean_ms_incl_index_build": 1000 * wall / len(queries)}
                if timing:
                    extra["mean_ms_scoring_only"] = 1000 * timing["retrieval_score_seconds"] / len(queries)
                    info["build_s"] = timing["retrieval_index_seconds"]
            elif per_q is not None:
                for q in queries[: args.warmup]:
                    per_q(q)
                lat = np.zeros(len(queries))
                results = []
                for i, q in enumerate(queries):
                    t0 = time.perf_counter()
                    r = per_q(q)
                    lat[i] = time.perf_counter() - t0
                    results.append(r)
            else:
                bs = int(kw.get("bs", 8 if name == "fast_hyb_batch" else 1))
                batch_fn(queries[: max(bs, args.warmup)])
                lat = np.zeros(len(queries))
                results = []
                for s in range(0, len(queries), bs):
                    t0 = time.perf_counter()
                    rr = batch_fn(queries[s : s + bs])
                    dt = time.perf_counter() - t0
                    lat[s : s + len(rr)] = dt / len(rr)
                    results.extend(rr)
            ranked = [[(ctx.doc_ids[d], float(s)) for d, s in zip(ids, sc)] for ids, sc in results]
            met = metrics_for(qids, ranked, qrels)
            label = method_label(spec)
            rec = {
                "dataset": args.dataset, "split": args.split, "method": label, "spec": spec,
                "threads": th if th > 0 else None, "n_queries": len(queries), "top_k": args.top_k,
                "exact": exact, "ndcg@10": met["ndcg@10"], "recall@100": met["recall@100"],
                "recall@10": met["recall@10"], "metrics": met,
                "latency": lat_stats(lat) if lat is not None else None,
                "index": info, "build_wall_s": build_wall, "rss_gb": rss_gb(), "peak_gb": peak_gb(),
                "other_eval_procs": others, "cpu": platform.processor(),
                "numba_threading_layer": _threading_layer(),
                "tag": args.tag, **extra,
            }
            k = args.top_k
            ids_arr = np.full((len(results), k), -1, np.int64)
            sc_arr = np.full((len(results), k), np.nan, np.float32)
            for i, (ids, sc) in enumerate(results):
                ids_arr[i, : len(ids)] = ids[:k]
                sc_arr[i, : len(sc)] = sc[:k]
            stem = f"{args.dataset}_{args.split}__{label}__t{th}"
            np.savez(RUNS / f"{stem}.npz", ids=ids_arr, scores=sc_arr,
                     latency=lat if lat is not None else np.zeros(0))
            (RUNS / f"{stem}.json").write_text(json.dumps(rec, indent=2, default=float))
            ls = rec["latency"]
            lat_txt = (f"mean={ls['mean_ms']:.2f}ms p50={ls['p50_ms']:.2f} p95={ls['p95_ms']:.2f}"
                       if ls else f"mean(batch,incl build)={extra.get('mean_ms_incl_index_build', 0):.1f}ms")
            print(f"[{label} t={th}] nDCG@10={100 * met['ndcg@10']:.2f} R@100={100 * met['recall@100']:.2f} "
                  f"{lat_txt} build={info.get('build_s')} rss={rss_gb():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
