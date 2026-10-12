"""Independent spot-check of the optimised exact scorer (opt/ssr_opt/fast_exact.py).

For every query of a split: full score vectors of `FastIndex.exact_all_hyb` vs the repo's
fixed exact kernel `exact_doc_scores` must be identical; also times both.

  python verify_fast_hyb.py nfcorpus test cache/A/nfcorpus/corpus_32k_test.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "SSR"), str(ROOT / "opt")]

from numba import get_num_threads, set_num_threads  # noqa: E402

from ssr.model import load_ssr_checkpoint  # noqa: E402
from ssr.retrieval.exact_index_maxsim import exact_doc_scores  # noqa: E402
from ssr.retrieval.inverted_index import build_chunk_index  # noqa: E402
from ssr.retrieval.mteb_io import load_mteb_split  # noqa: E402
from ssr.retrieval.retriever import RetrieverConfig, SparseMaxSimRetriever  # noqa: E402
from ssr.retrieval.sparse_repr import load_sparse_corpus  # noqa: E402
from ssr_opt.fast_exact import FastIndex  # noqa: E402

dataset, split, npz = sys.argv[1], sys.argv[2], Path(sys.argv[3])
torch.set_grad_enabled(False)
doc_ids, docs = load_sparse_corpus(npz)
sp = load_mteb_split(ROOT / "data" / "mteb", dataset, split)
model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device="cpu")
queries = SparseMaxSimRetriever(RetrieverConfig(encode_batch_size=32)).encode_queries(
    model, sp.query_texts, n_latents=docs[0].n_latents, device="cpu", show_progress=False)

base = build_chunk_index(docs, n_latents=docs[0].n_latents)
fast = FastIndex.build(docs)
for q in queries[:5]:  # JIT warm-up for both
    exact_doc_scores(q, base)
    fast.exact_all_hyb(fast.prepare(q))

max_diff, t_base, t_fast1, t_fastn = 0.0, 0.0, 0.0, 0.0
n_threads = get_num_threads()
for q in queries:
    t0 = time.perf_counter(); a = exact_doc_scores(q, base); t_base += time.perf_counter() - t0
    set_num_threads(1)
    t0 = time.perf_counter(); b1 = fast.exact_all_hyb(fast.prepare(q)); t_fast1 += time.perf_counter() - t0
    set_num_threads(n_threads)
    t0 = time.perf_counter(); bn = fast.exact_all_hyb(fast.prepare(q)); t_fastn += time.perf_counter() - t0
    max_diff = max(max_diff, float(np.abs(a - b1).max()), float(np.abs(a - bn).max()))
n = len(queries)
print(f"[verify] {dataset}/{split}: {n} queries, max |score diff| over all docs = {max_diff:.3g}")
print(f"[verify] repo exact kernel {1000 * t_base / n:.1f} ms/q | fast_hyb 1 thread {1000 * t_fast1 / n:.1f} ms/q"
      f" | fast_hyb {n_threads} threads {1000 * t_fastn / n:.1f} ms/q")
