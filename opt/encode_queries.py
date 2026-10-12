"""Encode MTEB queries once per (dataset, split) and cache the sparse token embeddings.

Mirrors ``ssr.retrieval.eval_mteb.evaluate_split`` exactly: ``load_ssr_checkpoint``
(token_scope from checkpoint / non-cls), ``SparseMaxSimRetriever.encode_queries`` with
encode_batch_size=8, ``final_topk = sae.topk`` and ``n_latents = sae.n_latents``.

Output: opt/qcache/{slug}_{split}.npz (same layout as ``save_sparse_corpus``;
``doc_ids`` holds the query ids).

Usage: python opt/encode_queries.py nfcorpus:test nfcorpus:dev scifact:train ...
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPRO / "hf_cache"))
os.environ.setdefault("TMP", str(REPRO / "opt" / "tmp"))
os.environ.setdefault("TEMP", str(REPRO / "opt" / "tmp"))

import torch  # noqa: E402

from ssr.model import load_ssr_checkpoint  # noqa: E402
from ssr.retrieval.mteb_io import load_mteb_split  # noqa: E402
from ssr.retrieval.retriever import RetrieverConfig, SparseMaxSimRetriever  # noqa: E402
from ssr.retrieval.sparse_repr import save_sparse_corpus  # noqa: E402

CKPT = REPRO / "ckpt" / "SSR-tok"
DATA = REPRO / "data" / "mteb"
OUT = REPRO / "opt" / "qcache"


def main(specs: list[str]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ssr_checkpoint(str(CKPT), device=device)
    retriever = SparseMaxSimRetriever(RetrieverConfig(mode="exact", encode_batch_size=8))
    retriever.config.final_topk = int(model.sae_module.topk)
    n_latents = int(model.sae_module.n_latents)
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        slug, split = spec.split(":")
        out = OUT / f"{slug}_{split}.npz"
        if out.is_file():
            print(f"skip {out} (exists)")
            continue
        sp = load_mteb_split(DATA, slug, split)
        t0 = time.perf_counter()
        with torch.inference_mode():
            q = retriever.encode_queries(
                model, sp.query_texts, n_latents=n_latents, device=device, show_progress=False
            )
        save_sparse_corpus(out, sp.query_ids, q)
        print(f"{slug}:{split} {len(q)} queries in {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
