"""SSR-tok exact retrieval for large corpora: batched GPU encoding + repo exact kernel.

Encoding mirrors pylate ColBERT.encode as used by the repo (tokenize with [Q]/[D]
prefix, forward through BERT + SAE, attention mask, punctuation skiplist for documents,
L2 normalisation), but runs in batches and keeps only the top-32 latents per token on
the GPU instead of returning 16384-d dense rows per token. Scoring is the exact Eq. 4
kernel from ssr/retrieval/exact_index_maxsim.py over one index of the whole corpus.

  python quora_ssr.py --dataset quora --splits test,dev --ignore-identical-ids
  python quora_ssr.py --validate nfcorpus   # compare encodings with the repo's cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSR"))

from crosscheck import ndcg_at_k  # noqa: E402
from ssr.model import load_ssr_checkpoint  # noqa: E402
from ssr.retrieval.exact_index_maxsim import exact_doc_scores  # noqa: E402
from ssr.retrieval.inverted_index import build_chunk_index  # noqa: E402
from ssr.retrieval.sparse_repr import (  # noqa: E402
    SparseTokenEmbeddings,
    load_sparse_corpus,
    save_sparse_corpus,
)

NL = chr(10)


def encode_sparse(model, texts, *, is_query: bool, batch_size: int) -> list[SparseTokenEmbeddings]:
    k = int(model.sae_module.topk)
    n_lat = int(model.sae_module.n_latents)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[SparseTokenEmbeddings | None] = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        feats = model.tokenize([texts[i] for i in idx], is_query=is_query)
        feats = {kk: v.to(model.device) if torch.is_tensor(v) else v for kk, v in feats.items()}
        with torch.inference_mode():
            emb = model(feats)["token_embeddings"]  # (B, L, n_lat), <= k nonzeros per row
        mask = feats["attention_mask"].bool()
        if not is_query:
            mask &= model.skiplist_mask(feats["input_ids"], model.skiplist).bool()
        vals, lats = torch.topk(emb.float(), k=k, dim=-1)
        vals = torch.where(vals > 1e-8, vals, torch.zeros_like(vals))
        vals = vals / vals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lats = torch.where(vals > 0, lats, torch.full_like(lats, -1))
        vals, lats, mask = vals.cpu().numpy(), lats.cpu().numpy().astype(np.int32), mask.cpu().numpy()
        for j, i in enumerate(idx):
            m = mask[j]
            out[i] = SparseTokenEmbeddings(indices=lats[j][m], values=vals[j][m], n_latents=n_lat)
    return out  # type: ignore[return-value]


def encode_sparse_fast(model, texts, *, is_query: bool, batch_size: int) -> list[SparseTokenEmbeddings]:
    """Same codes as ``encode_sparse`` without materialising 16,384-d rows per token.

    Runs the transformer, keeps only the tokens pylate would keep (attention mask, and
    the punctuation skiplist for documents), applies the SAE encoder to those rows
    (pre-activation, top-k, ReLU as in SparseAutoencoderCore) and L2-normalises the k
    values. [CLS] rows (token_scope non-cls) stay as empty rows, like the original path.
    """
    sae = model.sae_module
    core = sae.sae
    k = int(sae.topk)
    n_lat = int(sae.n_latents)
    cls_id = model.tokenizer.cls_token_id
    W = core.encoder.weight  # (n_lat, d)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[SparseTokenEmbeddings | None] = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        feats = model.tokenize([texts[i] for i in idx], is_query=is_query)
        feats = {kk: v.to(model.device) if torch.is_tensor(v) else v for kk, v in feats.items()}
        with torch.inference_mode():
            te = model[0](feats)["token_embeddings"]  # (B, L, d)
            mask = feats["attention_mask"].bool()
            if not is_query:
                mask &= model.skiplist_mask(feats["input_ids"], model.skiplist).bool()
            enc = mask.clone()
            if sae.token_scope == "non-cls":
                enc &= ~feats["input_ids"].eq(int(cls_id))
            x = te[enc]
            if core.normalize:
                x, _info = core.preprocess(x)
            pre = torch.nn.functional.linear(x - core.pre_bias, W, core.latent_bias)
            vals, lats = torch.topk(pre, k=k, dim=-1)
            vals = torch.relu(vals)
            vals = torch.where(vals > 1e-8, vals, torch.zeros_like(vals))
            vals = vals / vals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            lats = torch.where(vals > 0, lats, torch.full_like(lats, -1))
            # scatter encoded rows back into each document's kept-token sequence
            B, L = mask.shape
            full_v = torch.zeros((B, L, k), device=te.device)
            full_l = torch.full((B, L, k), -1, dtype=lats.dtype, device=te.device)
            full_v[enc] = vals
            full_l[enc] = lats
        full_v, full_l, mask = full_v.cpu().numpy(), full_l.cpu().numpy().astype(np.int32), mask.cpu().numpy()
        for j, i in enumerate(idx):
            m = mask[j]
            out[i] = SparseTokenEmbeddings(indices=full_l[j][m], values=full_v[j][m], n_latents=n_lat)
    return out  # type: ignore[return-value]


def read_corpus(path: Path):
    ids, texts = [], []
    for line in path.read_text(encoding="utf-8").split(NL):
        if line.strip():
            row = json.loads(line)
            ids.append(str(row["_id"]))
            texts.append(row["text"])
    return ids, texts


def read_split(data: Path, split: str):
    q_ids, q_texts, qrels = [], [], {}
    for line in (data / "queries" / f"{split}.tsv").read_text(encoding="utf-8").split(NL):
        if line.strip():
            a, b = line.split("\t", 1)
            q_ids.append(a)
            q_texts.append(b)
    for line in (data / "qrels" / f"{split}.tsv").read_text(encoding="utf-8").split(NL):
        if line.strip():
            qid, _, did, rel = line.split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, {})[did] = int(rel)
    keep = [i for i, q in enumerate(q_ids) if qrels.get(q)]
    return [q_ids[i] for i in keep], [q_texts[i] for i in keep], qrels


def validate(model, dataset: str) -> None:
    """Our batched encodings must match the repo's per-document encodings (cache/A)."""
    ids, docs = load_sparse_corpus(ROOT / "cache" / "A" / dataset / "corpus_32k_test.npz")
    _, texts = read_corpus(ROOT / "data" / "mteb" / dataset / "corpus.jsonl")
    n = 400
    mine = encode_sparse(model, texts[:n], is_query=False, batch_size=64)
    same_rows = tot_rows = 0
    max_err = 0.0
    for a, b in zip(mine, docs[:n]):
        assert a.n_tokens == b.n_tokens, (a.n_tokens, b.n_tokens)
        for t in range(a.n_tokens):
            sa = {int(u): float(v) for u, v in zip(a.indices[t], a.values[t]) if u >= 0}
            sb = {int(u): float(v) for u, v in zip(b.indices[t], b.values[t]) if u >= 0}
            tot_rows += 1
            if sa.keys() == sb.keys():
                same_rows += 1
                if sa:
                    max_err = max(max_err, max(abs(sa[u] - sb[u]) for u in sa))
    print(
        f"[validate] {dataset}: {n} docs, identical latent sets in {same_rows}/{tot_rows} token rows, "
        f"max |value diff| on those rows = {max_err:.2e}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--splits", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ignore-identical-ids", action="store_true")
    p.add_argument("--validate")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device="cuda")

    if a.validate:
        validate(model, a.validate)
        return

    data = ROOT / "data" / "mteb" / a.dataset
    cache = ROOT / "cache" / "fast" / a.dataset / "corpus_ssr_tok.npz"
    doc_ids, doc_texts = read_corpus(data / "corpus.jsonl")
    if cache.is_file():
        cached_ids, docs = load_sparse_corpus(cache)
        assert cached_ids == doc_ids
        print(f"[ssr] loaded cached corpus encodings: {cache}", flush=True)
    else:
        t0 = time.time()
        docs = encode_sparse(model, doc_texts, is_query=False, batch_size=a.batch_size)
        print(f"[ssr] encoded {len(docs)} docs in {time.time() - t0:.0f}s", flush=True)
        save_sparse_corpus(cache, doc_ids, docs)

    t0 = time.time()
    index = build_chunk_index(docs, n_latents=int(model.sae_module.n_latents))
    print(f"[ssr] index: {index.compact.n_postings} postings, built in {time.time() - t0:.0f}s", flush=True)
    del docs
    pos = {d: i for i, d in enumerate(doc_ids)}

    for split in a.splits.split(","):
        q_ids, q_texts, qrels = read_split(data, split)
        queries = encode_sparse(model, q_texts, is_query=True, batch_size=a.batch_size)
        t0 = time.time()
        nd = []
        for qid, q in zip(q_ids, queries):
            s = exact_doc_scores(q, index)
            if a.ignore_identical_ids and qid in pos:
                s[pos[qid]] = 0.0
            top = np.argpartition(-s, 100)[:100]
            top = top[np.argsort(-s[top], kind="stable")]
            nd.append(ndcg_at_k([doc_ids[i] for i in top], qrels[qid]))
        print(
            f"[ssr] {a.dataset}/{split}: nDCG@10 graded = {100 * sum(nd) / len(nd):.2f} "
            f"({len(nd)} queries, {1000 * (time.time() - t0) / len(nd):.0f} ms/query)",
            flush=True,
        )


if __name__ == "__main__":
    main()
