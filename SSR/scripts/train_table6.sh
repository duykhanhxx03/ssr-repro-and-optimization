#!/usr/bin/env bash
# Token SAE (SSR) with the hyper-parameters of paper Table 6 / App. D.1 (BERT-base).
#
# Not specified by the paper, taken from the authors' replies in GitHub issue #1:
#   - L_CE temperature 0.3;
#   - BERT is fine-tuned (not frozen);
#   - 32 BM25 hard negatives per query, triplets of one query grouped in one batch
#     (--negatives-per-query 32, ssr/samplers.py);
#   - lr 1e-3 (Table 6) was "not the most stable setting"; ~1e-6 worked better for
#     token-level SAE training -> see scripts/train_issue1_setup.sh.
# Extra arguments are forwarded, e.g. --gpu 0 --report-to none.
set -euo pipefail

python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir data/processed/msmarco/passage \
  --sample-format triplet \
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
  --learning-rate 1e-3 \
  --batch-size 64 \
  --epochs 2 \
  --warmup-steps 20000 \
  --output-dir output/ssr-token \
  "$@"
