"""Sanity check of the evaluation pipeline with a reference model (dense ColBERT, exact MaxSim).

Same data files and same nDCG code as crosscheck.py. If ColBERTv2 lands on its published
BEIR numbers, the data/metric side of the pipeline is trustworthy.

Usage: python eval_dense_colbert.py <hf_model_id> <dataset> [split[,split...]] [--ignore-identical-ids]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from pylate import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscheck import ndcg_at_k  # noqa: E402

root = Path(__file__).resolve().parent
model_id, dataset = sys.argv[1], sys.argv[2]
splits = (sys.argv[3] if len(sys.argv) > 3 else "test").split(",")
ignore_identical = "--ignore-identical-ids" in sys.argv  # BEIR/MTEB rule for Quora, ArguAna
data = root / "data" / "mteb" / dataset
torch.set_grad_enabled(False)

import json  # noqa: E402

NL = chr(10)  # split on "\n" only: str.splitlines() also breaks on U+2028 etc. inside texts
doc_ids, doc_texts = [], []
for line in (data / "corpus.jsonl").read_text(encoding="utf-8").split(NL):
    if line.strip():
        row = json.loads(line); doc_ids.append(row["_id"]); doc_texts.append(row["text"])
doc_pos = {d: i for i, d in enumerate(doc_ids)}
model = models.ColBERT(model_name_or_path=model_id, device="cuda")
model.half()
D = model.encode(doc_texts, is_query=False, batch_size=32, show_progress_bar=True, convert_to_tensor=False)
lens = torch.tensor([d.shape[0] for d in D], device="cuda")
flat = torch.cat([torch.as_tensor(d) for d in D]).to("cuda", torch.float16)  # (n_doc_tokens, dim)
owner = torch.repeat_interleave(torch.arange(len(D), device="cuda"), lens)
for split in splits:
    q_ids, q_texts = [], []
    for line in (data / "queries" / f"{split}.tsv").read_text(encoding="utf-8").split(NL):
        if line.strip():
            a, b = line.split("\t", 1); q_ids.append(a); q_texts.append(b)
    qrels: dict[str, dict[str, int]] = {}
    for line in (data / "qrels" / f"{split}.tsv").read_text(encoding="utf-8").split(NL):
        if line.strip():
            qid, _, did, rel = line.split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, {})[did] = int(rel)

    Q = model.encode(q_texts, is_query=True, batch_size=32, show_progress_bar=False, convert_to_tensor=False)

    nb, ng = [], []
    for qid, q in zip(q_ids, Q):
        if not qrels.get(qid):
            continue
        s_tok = flat @ torch.as_tensor(q).to("cuda", torch.float16).T  # (n_doc_tokens, n_q)
        per_doc = torch.full((len(D), s_tok.shape[1]), -1e4, device="cuda", dtype=torch.float32)
        per_doc.scatter_reduce_(0, owner[:, None].expand_as(s_tok), s_tok.float(), reduce="amax")
        total = per_doc.sum(1)
        if ignore_identical and qid in doc_pos:
            total[doc_pos[qid]] = -1e9
        top = torch.topk(total, k=100).indices.tolist()
        ranked = [doc_ids[i] for i in top]
        nb.append(ndcg_at_k(ranked, {d: 1 for d in qrels[qid]})); ng.append(ndcg_at_k(ranked, qrels[qid]))
    print(f"[dense] {model_id} {dataset}/{split}: nDCG@10 binary = {100*sum(nb)/len(nb):.2f}, graded = {100*sum(ng)/len(ng):.2f} ({len(nb)} queries)")
