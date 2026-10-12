"""Dense MaxSim with the SSR-tok checkpoint's own BERT backbone (SAE removed).

Same tokenizer, [Q]/[D] prefixes, query/doc lengths (192/512), no query expansion,
punctuation skiplist for documents and [CLS] excluded as in SSR-tok (token_scope
non-cls); token vectors are the 768-d last-layer embeddings, L2-normalised. This is
the dense late-interaction score that the paper's Theorem A says SSR approximates.

Documents are encoded and scored chunk by chunk on the GPU (nothing is stored), and
the running top-100 per query is kept.

Usage: python eval_backbone_dense.py <dataset> <split[,split...]>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSR"))

from crosscheck import ndcg_at_k  # noqa: E402
from ssr.model import load_ssr_checkpoint  # noqa: E402

NL = chr(10)
DOC_CHUNK = 2048  # docs encoded + scored per step
Q_TOKENS_PER_MM = 2048  # query tokens per matmul


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
    return q_ids, q_texts, qrels


def main() -> None:
    dataset, splits = sys.argv[1], sys.argv[2].split(",")
    data = ROOT / "data" / "mteb" / dataset
    dev = "cuda"
    torch.set_grad_enabled(False)

    model = load_ssr_checkpoint(str(ROOT / "ckpt" / "SSR-tok"), device=dev)
    sae = model.sae_module
    cls_id = int(model.tokenizer.cls_token_id)

    def dense_forward(features):
        # Bypass the SAE: keep backbone token embeddings, drop [CLS] like token_scope=non-cls.
        te = features["token_embeddings"]
        keep = ~features["input_ids"].eq(cls_id)
        features["token_embeddings"] = te * keep.unsqueeze(-1).to(te.dtype)
        return features

    sae.forward = dense_forward  # type: ignore[method-assign]

    def encode(texts, is_query):
        out = model.encode(
            texts,
            is_query=is_query,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        rows = []
        for e in out:
            t = torch.as_tensor(e, dtype=torch.float32)
            rows.append(t[t.abs().sum(-1) > 0])  # drop zeroed [CLS] / padding rows
        return rows

    # Queries of all splits, concatenated into one token matrix.
    split_info = []
    q_rows, q_owner = [], []
    for split in splits:
        q_ids, q_texts, qrels = read_split(data, split)
        keep = [i for i, q in enumerate(q_ids) if qrels.get(q)]
        q_ids = [q_ids[i] for i in keep]
        q_texts = [q_texts[i] for i in keep]
        base = sum(len(s[1]) for s in split_info)
        for qi, e in enumerate(encode(q_texts, True)):
            q_rows.append(e)
            q_owner.append(torch.full((e.shape[0],), base + qi, dtype=torch.long))
        split_info.append((split, q_ids, qrels))
    n_q = sum(len(s[1]) for s in split_info)
    Q = torch.cat(q_rows).to(dev, torch.float16)  # (n_qtok, 768)
    q_owner = torch.cat(q_owner).to(dev)

    doc_ids, doc_texts = [], []
    for line in (data / "corpus.jsonl").read_text(encoding="utf-8").split(NL):
        if line.strip():
            row = json.loads(line)
            doc_ids.append(row["_id"])
            doc_texts.append(row["text"])

    best_s = torch.full((n_q, 100), -1e9, device=dev)
    best_i = torch.full((n_q, 100), -1, dtype=torch.long, device=dev)
    t0 = time.time()
    for start in range(0, len(doc_texts), DOC_CHUNK):
        rows = encode(doc_texts[start : start + DOC_CHUNK], False)
        lens = torch.tensor([r.shape[0] for r in rows], device=dev)
        D = torch.cat(rows).to(dev, torch.float16)
        owner = torch.repeat_interleave(torch.arange(len(rows), device=dev), lens)
        doc_q = torch.zeros((len(rows), n_q), device=dev)  # summed MaxSim per (doc, query)
        for qs in range(0, Q.shape[0], Q_TOKENS_PER_MM):
            qe = min(qs + Q_TOKENS_PER_MM, Q.shape[0])
            s = D @ Q[qs:qe].T  # (n_doc_tokens, n_qtok_batch)
            m = torch.full((len(rows), qe - qs), -1e4, device=dev)
            m.scatter_reduce_(0, owner[:, None].expand_as(s), s.float(), reduce="amax")
            doc_q.index_add_(1, q_owner[qs:qe], m)
            del s, m
        cand_s = torch.cat([best_s, doc_q.T], dim=1)
        cand_i = torch.cat(
            [best_i, (torch.arange(len(rows), device=dev) + start).expand(n_q, -1)], dim=1
        )
        best_s, pos = torch.topk(cand_s, k=100, dim=1)
        best_i = torch.gather(cand_i, 1, pos)
        del D, doc_q
    elapsed = time.time() - t0

    base = 0
    for split, q_ids, qrels in split_info:
        nd = []
        for qi, qid in enumerate(q_ids):
            ranked = [doc_ids[i] for i in best_i[base + qi].tolist() if i >= 0]
            nd.append(ndcg_at_k(ranked, qrels[qid]))
        base += len(q_ids)
        print(
            f"[backbone-dense] {dataset}/{split}: nDCG@10 graded = {100 * sum(nd) / len(nd):.2f} "
            f"({len(nd)} queries)",
            flush=True,
        )
    print(f"[backbone-dense] {dataset}: corpus encode+score {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
