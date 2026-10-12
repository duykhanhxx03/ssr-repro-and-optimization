#!/usr/bin/env bash
# Token SAE (SSR) with the training setup the authors describe in GitHub issue #1
# (2026-07-25 / 2026-07-31), which differs from paper Table 6 in the learning rate:
#   - BERT is fine-tuned (not frozen); lr ~1e-6 ("lr=0.001 is not the most stable setting");
#   - 32 BM25 hard negatives per query -> 32 triplets, one query's triplets per batch
#     (2 queries x 32 triplets at batch size 64); warmup 20000 steps (~4% of 2 epochs);
#   - L_CE temperature 0.3; [MASK] query-expansion tokens are not used in MaxSim.
# Batch 64 needs a large GPU (an 8 GB card runs out of memory); extra args are forwarded.
set -euo pipefail

python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir data/processed/msmarco/passage \
  --sample-format triplet \
  --negatives-per-query 32 \
  --model-name bert-base-uncased \
  --sae-token-scope non-cls \
  --n-latents 16384 \
  --topk 32 \
  --auxk 2048 \
  --recon-coef 1.0 \
  --recon-4k-coef 0.125 \
  --auxk-coef 0.03125 \
  --ucl-coef 0.1 \
  --maxsim-coef 0.05 \
  --maxsim-temperature 0.3 \
  --learning-rate 1e-6 \
  --batch-size 64 \
  --epochs 2 \
  --warmup-steps 20000 \
  --output-dir output/ssr-token \
  "$@"
