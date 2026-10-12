"""MS MARCO passage re-ranking: SSR-tok vs ColBERTv2 (vs the SSR backbone, dense).

Stage `bm25`: Lucene-style BM25 (k1=0.9, b=0.4, Anserini's MS MARCO setting) over the
full 8.8M-passage collection, built with numpy in chunks; writes the top-100 candidates
for dev-small (6,980 queries, MRR@10) and TREC DL 2019 (43 queries, nDCG@10), plus the
texts of all candidate passages.
Stage `rerank`: exact late-interaction re-scoring of the same candidates with each model
(SSR: Eq. 4 on the sparse codes; dense models: MaxSim on normalised token embeddings).

  python msmarco_rerank.py bm25
  python msmarco_rerank.py rerank --models ssr colbertv2 backbone
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "msmarco_raw"
OUT = ROOT / "data" / "msmarco_rerank"
NL = chr(10)
TOPK = 100
K1, B = 0.9, 0.4

STOPWORDS = set(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)  # Lucene's English stopword set
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").split(NL)):
        parts = line.strip().split("\t")
        if len(parts) != 3 or not parts[2].lstrip("-").isdigit():
            continue  # header / blank
        qid, did, rel = parts
        qrels.setdefault(qid, {})[did] = int(rel)
    return qrels


class Analyzer:
    def __init__(self) -> None:
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        self.cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
        out = []
        for t in toks:
            s = self.cache.get(t)
            if s is None:
                s = self.stemmer.stemWord(t)
                if len(self.cache) < 2_000_000:
                    self.cache[t] = s
            out.append(s)
        return out


def stage_bm25() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    an = Analyzer()
    vocab: dict[str, int] = {}
    doc_ids: list[str] = []
    dl_list: list[int] = []
    parts_doc, parts_term, parts_tf = [], [], []
    buf_d, buf_t, buf_f = [], [], []
    t0 = time.time()
    for di, row in enumerate(iter_jsonl(RAW / "corpus.jsonl")):
        doc_ids.append(str(row["_id"]))
        toks = an(((row.get("title") or "") + " " + row["text"]).strip())
        dl_list.append(len(toks))
        for term, tf in Counter(toks).items():
            tid = vocab.get(term)
            if tid is None:
                tid = vocab[term] = len(vocab)
            buf_d.append(di)
            buf_t.append(tid)
            buf_f.append(tf)
        if len(buf_d) > 20_000_000:
            parts_doc.append(np.asarray(buf_d, np.int32))
            parts_term.append(np.asarray(buf_t, np.int32))
            parts_tf.append(np.asarray(buf_f, np.int16))
            buf_d, buf_t, buf_f = [], [], []
        if di % 1_000_000 == 0:
            print(f"[bm25] tokenised {di:,} docs, vocab {len(vocab):,} ({time.time() - t0:.0f}s)", flush=True)
    parts_doc.append(np.asarray(buf_d, np.int32))
    parts_term.append(np.asarray(buf_t, np.int32))
    parts_tf.append(np.asarray(buf_f, np.int16))
    del buf_d, buf_t, buf_f
    doc = np.concatenate(parts_doc)
    term = np.concatenate(parts_term)
    tf = np.concatenate(parts_tf).astype(np.float32)
    del parts_doc, parts_term, parts_tf
    n_docs = len(doc_ids)
    dl = np.asarray(dl_list, np.float32)
    avgdl = float(dl.mean())
    print(f"[bm25] {n_docs:,} docs, {doc.size:,} postings, avgdl {avgdl:.1f} ({time.time() - t0:.0f}s)", flush=True)

    order = np.argsort(term, kind="stable")
    doc, term, tf = doc[order], term[order], tf[order]
    del order
    df = np.bincount(term, minlength=len(vocab)).astype(np.float64)
    offsets = np.zeros(len(vocab) + 1, np.int64)
    np.cumsum(df.astype(np.int64), out=offsets[1:])
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = (K1 * (1.0 - B + B * dl / avgdl)).astype(np.float32)
    weight = idf[term] * tf * (K1 + 1.0) / (tf + norm[doc])
    del term, tf
    print(f"[bm25] index ready ({time.time() - t0:.0f}s)", flush=True)

    queries = {str(r["_id"]): r["text"] for r in iter_jsonl(RAW / "queries.jsonl")}
    runs: dict[str, dict[str, list[str]]] = {}
    needed: set[str] = set()
    for split in ("dev", "test"):
        qrels = read_qrels(RAW / "qrels" / f"{split}.tsv")
        run: dict[str, list[str]] = {}
        for qid in qrels:
            scores = np.zeros(n_docs, np.float32)
            for t in set(an(queries[qid])):
                tid = vocab.get(t)
                if tid is not None:
                    s, e = offsets[tid], offsets[tid + 1]
                    scores[doc[s:e]] += weight[s:e]  # a doc occurs once per term's postings
            top = np.argpartition(-scores, TOPK)[:TOPK]
            top = top[np.argsort(-scores[top], kind="stable")]
            run[qid] = [doc_ids[i] for i in top]
            needed.update(run[qid])
        runs[split] = run
        print(f"[bm25] {split}: {len(run)} queries retrieved ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "bm25_top100.json").write_text(json.dumps(runs), encoding="utf-8")

    with open(OUT / "cand_corpus.jsonl", "w", encoding="utf-8") as f:
        for row in iter_jsonl(RAW / "corpus.jsonl"):
            if str(row["_id"]) in needed:
                text = ((row.get("title") or "") + " " + row["text"]).strip()
                f.write(json.dumps({"_id": str(row["_id"]), "text": text}, ensure_ascii=False) + NL)
    (OUT / "queries.json").write_text(
        json.dumps({q: queries[q] for s in runs.values() for q in s}), encoding="utf-8"
    )
    print(f"[bm25] wrote {len(needed):,} candidate passages ({time.time() - t0:.0f}s)", flush=True)
    report(runs, "BM25", {s: {q: list(range(TOPK, 0, -1)) for q in r} for s, r in runs.items()})


def mrr10(ranked: list[str], rel: dict[str, int], threshold: int = 1) -> float:
    for i, d in enumerate(ranked[:10]):
        if rel.get(d, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg10(ranked: list[str], rel: dict[str, int]) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def report(runs, name: str, scores) -> dict:
    out = {}
    for split, run in runs.items():
        qrels = read_qrels(RAW / "qrels" / f"{split}.tsv")
        vals_mrr, vals_ndcg = [], []
        for qid, cands in run.items():
            sc = scores[split][qid]
            ranked = [c for _, c in sorted(zip(sc, cands), key=lambda x: -x[0])]
            thr = 2 if split == "test" else 1  # TREC DL: relevance >= 2 counts as relevant for MRR
            vals_mrr.append(mrr10(ranked, qrels[qid], thr))
            vals_ndcg.append(ndcg10(ranked, qrels[qid]))
        out[split] = {"mrr@10": 100 * np.mean(vals_mrr), "ndcg@10": 100 * np.mean(vals_ndcg), "n": len(run)}
        label = "dev-small" if split == "dev" else "TREC-DL19"
        print(
            f"[{name}] {label}: MRR@10 = {out[split]['mrr@10']:.2f}, nDCG@10 = {out[split]['ndcg@10']:.2f} "
            f"({len(run)} queries)",
            flush=True,
        )
    return out


def stage_rerank(models: list[str], block: int) -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "SSR"))
    runs = json.loads((OUT / "bm25_top100.json").read_text(encoding="utf-8"))
    queries = json.loads((OUT / "queries.json").read_text(encoding="utf-8"))
    texts = {r["_id"]: r["text"] for r in iter_jsonl(OUT / "cand_corpus.jsonl")}
    jobs = [(s, q) for s, r in runs.items() for q in r]
    torch.set_grad_enabled(False)
    results = {}
    for name in models:
        t0 = time.time()
        if name == "ssr":
            from quora_ssr import encode_sparse_fast as encode_sparse  # same codes, no 16k-d rows
            from ssr.model import load_ssr_checkpoint

            model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device="cuda")
            n_lat = int(model.sae_module.n_latents)

            def enc_docs(ts):
                return encode_sparse(model, ts, is_query=False, batch_size=32)

            def enc_q(ts):
                return encode_sparse(model, ts, is_query=True, batch_size=128)

            def score(q, docs):
                qd = np.zeros((q.n_tokens, n_lat), np.float32)
                for t in range(q.n_tokens):
                    m = q.indices[t] >= 0
                    qd[t, q.indices[t][m]] = q.values[t][m]
                out = []
                for d in docs:
                    idx = np.where(d.indices >= 0, d.indices, 0)
                    val = np.where(d.indices >= 0, d.values, 0.0)
                    s = (qd[:, idx] * val[None]).sum(-1)  # (nq, nd)
                    out.append(float(s.max(axis=1).clip(min=0).sum()) if s.size else 0.0)
                return out
        else:
            from pylate import models as plm
            from ssr.model import load_ssr_checkpoint

            if name == "colbertv2":
                model = plm.ColBERT(model_name_or_path="lightonai/colbertv2.0", device="cuda")
                model.half()
            else:  # SSR backbone without the SAE, as in eval_backbone_dense.py
                model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device="cuda")
                cls_id = int(model.tokenizer.cls_token_id)

                def dense_forward(features):
                    te = features["token_embeddings"]
                    keep = ~features["input_ids"].eq(cls_id)
                    features["token_embeddings"] = te * keep.unsqueeze(-1).to(te.dtype)
                    return features

                model.sae_module.forward = dense_forward

            def _enc(ts, is_query):
                embs = model.encode(ts, is_query=is_query, batch_size=64, show_progress_bar=False,
                                    convert_to_tensor=False, normalize_embeddings=True)
                rows = []
                for e in embs:
                    t = torch.as_tensor(e).float()
                    rows.append(t[t.abs().sum(-1) > 0].half())
                return rows

            def enc_docs(ts):
                return _enc(ts, False)

            def enc_q(ts):
                return _enc(ts, True)

            def score(q, docs):
                qg = q.cuda().float()
                return [float((d.cuda().float() @ qg.T).max(dim=0).values.sum()) for d in docs]

        sc: dict[str, dict[str, list[float]]] = {s: {} for s in runs}
        for b0 in range(0, len(jobs), block):
            bj = jobs[b0 : b0 + block]
            pids = sorted({p for s, q in bj for p in runs[s][q]})
            denc = dict(zip(pids, enc_docs([texts[p] for p in pids])))
            qenc = enc_q([queries[q] for _, q in bj])
            for (s, q), qe in zip(bj, qenc):
                sc[s][q] = score(qe, [denc[p] for p in runs[s][q]])
            if (b0 // block) % 5 == 0:
                print(f"[{name}] {b0 + len(bj)}/{len(jobs)} queries ({time.time() - t0:.0f}s)", flush=True)
        results[name] = report(runs, name, sc)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / "rerank_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("bm25", "rerank"))
    p.add_argument("--models", nargs="+", default=["ssr", "colbertv2", "backbone"])
    p.add_argument("--block", type=int, default=256)
    a = p.parse_args()
    if a.stage == "bm25":
        stage_bm25()
    else:
        stage_rerank(a.models, a.block)


if __name__ == "__main__":
    main()
