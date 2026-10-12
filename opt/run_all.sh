#!/usr/bin/env bash
# Benchmark sequence (run only when no other eval_mteb / run_eval / quora / colbert jobs run).
# Usage: bash opt/run_all.sh <target> ...   targets: small_base small_repo small_fast
#        fiqa_fast fiqa_sweep fiqa_test_variants fiqa_sub fiqa_base
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TMP="$PWD/opt/tmp" TEMP="$PWD/opt/tmp" HF_HOME="$PWD/hf_cache"
PY=venv/Scripts/python.exe
B="$PY opt/bench.py"
LOG=opt/logs
mkdir -p $LOG
SMALL=${SMALL_SPLITS:-nfcorpus:test nfcorpus:dev scifact:test scifact:train}

small_base() {  # repo kernels on the whole corpus as one index (single-threaded numba)
  for s in $SMALL; do ds=${s%%:*}; sp=${s##*:}
    $B --dataset $ds --split $sp --methods base_exact base_ssrpp --tag small 2>&1 | tee $LOG/${ds}_${sp}_base.log
  done
}
small_repo() {  # repo default batch pipelines (5000-doc chunks), test splits only
  for s in $SMALL; do ds=${s%%:*}; sp=${s##*:}
    [ "$sp" = test ] || continue
    $B --dataset $ds --split $sp --methods repo_exact repo_ssrpp --tag small 2>&1 | tee $LOG/${ds}_${sp}_repo.log
  done
}
small_fast() {
  for s in $SMALL; do ds=${s%%:*}; sp=${s##*:}
    $B --dataset $ds --split $sp --threads 1 6 12 --methods fast_hyb fast_ssrpp --tag small \
      2>&1 | tee $LOG/${ds}_${sp}_fast.log
    $B --dataset $ds --split $sp --threads 6 --methods fast_lat fast_tok fast_ub gpu_exact \
      "fast_lb:t=4,n=1000" --tag small 2>&1 | tee $LOG/${ds}_${sp}_fast2.log
  done
}
fiqa_fast() {   # optimised exact + SSR++ port, full FiQA test
  $B --dataset fiqa --split test --threads 1 6 12 --methods fast_hyb fast_ssrpp --tag fiqa \
    2>&1 | tee $LOG/fiqa_test_fast.log
  $B --dataset fiqa --split test --threads 6 --methods fast_lat fast_tok gpu_exact "gpu_exact:i32=1" --tag fiqa \
    2>&1 | tee $LOG/fiqa_test_fast2.log
}
SWEEP_METHODS='fast_hyb
fast_ssrpp:kc=4,n=2000
fast_ssrpp:kc=4,n=5000
fast_ssrpp:kc=4,n=10000
fast_ssrpp:kc=8,n=2000
fast_ssrpp:kc=8,n=5000
fast_ssrpp:kc=32,n=2000
fast_ssrpp:kc=32,n=5000
fast_lb:t=2,n=1000
fast_lb:t=4,n=500
fast_lb:t=4,n=1000
fast_lb:t=4,n=2000
fast_lb:t=8,n=500
fast_lb:t=8,n=1000
fast_lb:t=8,n=2000
fast_lb:t=16,n=1000'
fiqa_sweep() {  # quality/latency trade-off of approximate variants (decide on dev/train1k)
  for sp in dev train1k; do
    $B --dataset fiqa --split $sp --threads ${SWEEP_THREADS:-6} --tag sweep --methods $SWEEP_METHODS \
      2>&1 | tee $LOG/fiqa_${sp}_sweep.log
  done
}
fiqa_test_variants() {  # same variants on test (secondary)
  $B --dataset fiqa --split test --threads 6 --tag sweep \
    --methods $(echo "$SWEEP_METHODS" | grep -v '^fast_hyb$') 2>&1 | tee $LOG/fiqa_test_sweep.log
}
fiqa_sub() {    # slow baselines on the first 100 test queries (+ optimised refs on that subset)
  $B --dataset fiqa --split test --max-queries 100 --methods base_exact base_ssrpp --tag fiqa100 \
    2>&1 | tee $LOG/fiqa_test100_base.log
  $B --dataset fiqa --split test --max-queries 100 --methods repo_exact repo_ssrpp --tag fiqa100 \
    2>&1 | tee $LOG/fiqa_test100_repo.log
  $B --dataset fiqa --split test --max-queries 100 --threads 6 --methods fast_hyb fast_ub \
    fast_ssrpp --tag fiqa100 2>&1 | tee $LOG/fiqa_test100_fast.log
}
fiqa_base() {   # repo exact on the whole corpus as one index, full test (reference lists)
  $B --dataset fiqa --split test --methods base_exact --tag fiqa 2>&1 | tee $LOG/fiqa_test_base.log
}

for t in "$@"; do $t; done
