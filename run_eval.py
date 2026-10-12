"""Evaluate the released SSR-tok checkpoint with the repo's own eval pipeline.

Only change vs. the repo: replace eval_mteb.load_ssr, which (per GitHub issue #1)
builds a randomly initialised SAE instead of loading the checkpoint. We load the
checkpoint's Transformer + SparseAutoencoder and pass the ColBERT settings stored in
config_sentence_transformers.json (query/doc length, no query expansion, prefixes),
which SSR(modules=...) would otherwise ignore.

Usage:
  python run_eval.py --ckpt ckpt/SSR-tok --token-scope non-cls [--extra-skip stopwords] \
      -- <eval_mteb args without --model-path>
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "SSR"
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sentence_transformers.models import Transformer  # noqa: E402

from ssr.model import SSR  # noqa: E402
from ssr.retrieval import eval_mteb  # noqa: E402
from ssr.sparse_autoencoder import SparseAutoencoder  # noqa: E402

# NLTK English stopword list (inlined to avoid a download).
EN_STOPWORDS = (
    "i me my myself we our ours ourselves you your yours yourself yourselves he him his "
    "himself she her hers herself it its itself they them their theirs themselves what which "
    "who whom this that these those am is are was were be been being have has had having do "
    "does did doing a an the and but if or because as until while of at by for with about "
    "against between into through during before after above below to from up down in out on "
    "off over under again further then once here there when where why how all any both each "
    "few more most other some such no nor not only own same so than too very s t can will "
    "just don should now"
).split()


def load_checkpoint(ckpt: Path, device: str, token_scope: str, extra_skip: str | None) -> SSR:
    transformer = Transformer(str(ckpt))

    sae_dir = ckpt / "1_SparseAutoencoder"
    cfg = json.loads((sae_dir / "config.json").read_text())
    unknown = {k: cfg.pop(k) for k in list(cfg) if k not in SparseAutoencoder.config_keys}
    if unknown:
        print(f"[run_eval] ignoring SAE config keys not supported by repo: {unknown}")
    cfg.setdefault("token_scope", token_scope)
    sae = SparseAutoencoder(**cfg)
    missing, unexpected = sae.load_state_dict(load_file(str(sae_dir / "model.safetensors")), strict=False)
    # Tied decoder: encoder.weight is stored under decoder.linear.weight.
    missing = [k for k in missing if k != "sae.encoder.weight"]
    if missing or unexpected:
        raise RuntimeError(f"SAE state mismatch: missing={missing} unexpected={unexpected}")

    st_cfg = json.loads((ckpt / "config_sentence_transformers.json").read_text())
    skiplist = list(st_cfg.get("skiplist_words") or string.punctuation)
    if extra_skip == "stopwords":
        skiplist += EN_STOPWORDS

    model = SSR(
        modules=[transformer, sae],
        device=device,
        query_prefix=st_cfg["query_prefix"],
        document_prefix=st_cfg["document_prefix"],
        query_length=st_cfg["query_length"],
        document_length=st_cfg["document_length"],
        do_query_expansion=st_cfg["do_query_expansion"],
        attend_to_expansion_tokens=st_cfg["attend_to_expansion_tokens"],
        skiplist_words=skiplist,
    )
    model.eval()
    print(
        f"[run_eval] loaded {ckpt}: n_latents={sae.n_latents} topk={sae.topk} "
        f"token_scope={sae.token_scope} q_len={model.query_length} d_len={model.document_length} "
        f"expansion={model.do_query_expansion} vocab={len(model.tokenizer)} skiplist={len(skiplist)}"
    )
    return model


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("pass eval_mteb args after --")
    split_at = sys.argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--token-scope", choices=("all", "non-cls"), default="non-cls")
    p.add_argument("--extra-skip", choices=("stopwords",), default=None)
    ours = p.parse_args(sys.argv[1:split_at])
    rest = sys.argv[split_at + 1 :]

    eval_mteb.load_ssr = lambda _path, device, *_a, **_k: load_checkpoint(
        ours.ckpt.resolve(), device, ours.token_scope, ours.extra_skip
    )
    torch.set_grad_enabled(False)
    eval_mteb.main(["--model-path", str(ours.ckpt.resolve()), *rest])


if __name__ == "__main__":
    main()
