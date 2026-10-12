## Main comparison

| dataset/split (nq) | method | threads | exact? | nDCG@10 | R@100 | mean ms | p50 | p95 | speedup vs base_exact | identical top-100 (bitwise / up to ties) |
|---|---|---|---|---|---|---|---|---|---|---|
| fiqa/test (100) | base_exact | 1 | yes (reference) | 45.36 | 73.95 | 2331.63 | 2103.16 | 3942.54 | 1.0 | 1.000 / 1.000 |
| fiqa/test (100) | repo_exact | 1 | yes (reference) | 45.36 | 73.95 | 1634.69 |  |  | 1.4 | 0.990 / 1.000 |
| fiqa/test (100) | base_ssrpp | 1 | no (SSR++) | 37.58 | 52.02 | 478.60 | 426.71 | 854.69 | 4.9 | 0.010 / 0.010 |
| fiqa/test (100) | repo_ssrpp | 1 | no (SSR++) | 37.58 | 52.02 | 1618.53 |  |  | 1.4 | 0.010 / 0.010 (vs base_ssrpp: 1.000) |
| fiqa/test (100) | fast_ub | 6 | yes (safe pruning) | 45.36 | 73.95 | 206.71 | 210.73 | 278.13 | 11.3 | 0.990 / 1.000 |
| fiqa/test (648) | fast_hyb | 1 | yes (bit-identical) | 41.41 | 70.85 | 320.36 | 318.49 | 496.63 |  | 1.000 / 1.000 |
| fiqa/test (648) | fast_hyb | 6 | yes (bit-identical) | 41.41 | 70.85 | 82.33 | 81.58 | 127.22 |  | 1.000 / 1.000 |
| fiqa/test (648) | fast_hyb | 12 | yes (bit-identical) | 41.41 | 70.85 | 54.40 | 53.01 | 82.08 |  | 1.000 / 1.000 |
| fiqa/test (648) | fast_lat | 6 | yes (bit-identical) | 41.41 | 70.85 | 119.80 | 115.34 | 205.28 |  | 1.000 / 1.000 |
| fiqa/test (648) | fast_tok | 6 | yes (bit-identical) | 41.41 | 70.85 | 268.86 | 257.33 | 499.30 |  | 1.000 / 1.000 |
| fiqa/test (648) | gpu_exact |  | yes (float32, ~1e-6) | 41.41 | 70.85 | 55.88 | 55.46 | 59.55 |  | 0.000 / 1.000 |
| fiqa/test (648) | gpu_exact-i321 |  | yes (float32, ~1e-6) | 41.41 | 70.85 | 52.10 | 50.42 | 54.57 |  | 0.000 / 1.000 |
| fiqa/test (648) | fast_ssrpp | 1 | no (SSR++) | 36.86 | 50.90 | 59.53 | 59.26 | 85.89 |  | 0.008 / 0.008 |
| fiqa/test (648) | fast_ssrpp | 6 | no (SSR++) | 36.86 | 50.90 | 18.56 | 18.24 | 27.46 |  | 0.008 / 0.008 |
| fiqa/test (648) | fast_ssrpp | 12 | no (SSR++) | 36.86 | 50.90 | 11.74 | 11.40 | 16.87 |  | 0.008 / 0.008 |
| nfcorpus/dev (324) | base_exact | 1 | yes (reference) | 27.37 | 26.18 | 41.70 | 29.53 | 105.21 | 1.0 | 1.000 / 1.000 |
| nfcorpus/dev (324) | repo_exact | 1 | yes (reference) | 27.41 | 26.18 | 42.59 |  |  | 1.0 | 0.522 / 1.000 |
| nfcorpus/dev (324) | base_ssrpp | 1 | no (SSR++) | 27.39 | 25.95 | 28.55 | 20.38 | 71.00 | 1.5 | 0.250 / 0.515 |
| nfcorpus/dev (324) | repo_ssrpp | 1 | no (SSR++) | 27.39 | 25.95 | 86.10 |  |  | 0.5 | 0.244 / 0.515 (vs base_ssrpp: 0.543) |
| nfcorpus/dev (324) | fast_hyb | 1 | yes (bit-identical) | 27.37 | 26.18 | 12.26 | 10.66 | 21.88 | 3.4 | 1.000 / 1.000 |
| nfcorpus/dev (324) | fast_hyb | 6 | yes (bit-identical) | 27.37 | 26.18 | 3.54 | 3.08 | 6.82 | 11.8 | 1.000 / 1.000 |
| nfcorpus/dev (324) | fast_hyb | 12 | yes (bit-identical) | 27.37 | 26.18 | 2.43 | 2.02 | 4.77 | 17.2 | 1.000 / 1.000 |
| nfcorpus/dev (324) | fast_lat | 6 | yes (bit-identical) | 27.37 | 26.18 | 3.80 | 2.97 | 7.89 | 11.0 | 1.000 / 1.000 |
| nfcorpus/dev (324) | fast_tok | 6 | yes (bit-identical) | 27.37 | 26.18 | 7.03 | 5.10 | 16.33 | 5.9 | 1.000 / 1.000 |
| nfcorpus/dev (324) | fast_ub | 6 | yes (safe pruning) | 27.39 | 26.18 | 14.71 | 14.28 | 20.89 | 2.8 | 0.552 / 1.000 |
| nfcorpus/dev (324) | gpu_exact |  | yes (float32, ~1e-6) | 27.37 | 26.18 | 7.52 | 7.47 | 7.90 | 5.5 | 0.000 / 1.000 |
| nfcorpus/dev (324) | fast_ssrpp | 1 | no (SSR++) | 27.39 | 25.95 | 26.10 | 25.13 | 34.83 | 1.6 | 0.250 / 0.515 (vs base_ssrpp: 1.000) |
| nfcorpus/dev (324) | fast_ssrpp | 6 | no (SSR++) | 27.39 | 25.95 | 8.23 | 7.91 | 11.15 | 5.1 | 0.250 / 0.515 (vs base_ssrpp: 1.000) |
| nfcorpus/dev (324) | fast_ssrpp | 12 | no (SSR++) | 27.39 | 25.95 | 5.21 | 4.73 | 7.90 | 8.0 | 0.250 / 0.515 (vs base_ssrpp: 1.000) |
| nfcorpus/dev (324) | fast_lb-t4-n1000 | 6 | no (approx) | 27.38 | 25.87 | 5.70 | 5.48 | 8.24 | 7.3 | 0.207 / 0.417 |
| nfcorpus/test (323) | base_exact | 1 | yes (reference) | 43.99 | 36.96 | 44.09 | 30.41 | 110.78 | 1.0 | 1.000 / 1.000 |
| nfcorpus/test (323) | repo_exact | 1 | yes (reference) | 43.97 | 36.96 | 43.64 |  |  | 1.0 | 0.517 / 1.000 |
| nfcorpus/test (323) | base_ssrpp | 1 | no (SSR++) | 43.94 | 37.02 | 29.42 | 21.22 | 73.53 | 1.5 | 0.294 / 0.604 |
| nfcorpus/test (323) | repo_ssrpp | 1 | no (SSR++) | 43.98 | 37.02 | 84.19 |  |  | 0.5 | 0.303 / 0.604 (vs base_ssrpp: 0.486) |
| nfcorpus/test (323) | fast_hyb | 1 | yes (bit-identical) | 43.99 | 36.96 | 11.95 | 10.47 | 22.51 | 3.7 | 1.000 / 1.000 |
| nfcorpus/test (323) | fast_hyb | 6 | yes (bit-identical) | 43.99 | 36.96 | 3.77 | 3.27 | 7.79 | 11.7 | 1.000 / 1.000 |
| nfcorpus/test (323) | fast_hyb | 12 | yes (bit-identical) | 43.99 | 36.96 | 2.53 | 2.12 | 6.02 | 17.4 | 1.000 / 1.000 |
| nfcorpus/test (323) | fast_lat | 6 | yes (bit-identical) | 43.99 | 36.96 | 3.87 | 3.15 | 7.91 | 11.4 | 1.000 / 1.000 |
| nfcorpus/test (323) | fast_tok | 6 | yes (bit-identical) | 43.99 | 36.96 | 6.97 | 5.21 | 17.22 | 6.3 | 1.000 / 1.000 |
| nfcorpus/test (323) | fast_ub | 6 | yes (safe pruning) | 44.04 | 36.96 | 14.48 | 14.10 | 20.41 | 3.0 | 0.514 / 1.000 |
| nfcorpus/test (323) | gpu_exact |  | yes (float32, ~1e-6) | 43.96 | 36.96 | 7.53 | 7.44 | 7.89 | 5.9 | 0.000 / 1.000 |
| nfcorpus/test (323) | fast_ssrpp | 1 | no (SSR++) | 43.94 | 37.02 | 26.12 | 25.09 | 34.72 | 1.7 | 0.294 / 0.604 (vs base_ssrpp: 1.000) |
| nfcorpus/test (323) | fast_ssrpp | 6 | no (SSR++) | 43.94 | 37.02 | 8.05 | 7.81 | 11.51 | 5.5 | 0.294 / 0.604 (vs base_ssrpp: 1.000) |
| nfcorpus/test (323) | fast_ssrpp | 12 | no (SSR++) | 43.94 | 37.02 | 5.03 | 4.69 | 7.16 | 8.8 | 0.294 / 0.604 (vs base_ssrpp: 1.000) |
| nfcorpus/test (323) | fast_lb-t4-n1000 | 6 | no (approx) | 43.89 | 37.03 | 5.65 | 5.42 | 8.21 | 7.8 | 0.263 / 0.498 |
| scifact/test (300) | base_exact | 1 | yes (reference) | 87.83 | 100.00 | 365.86 | 333.81 | 677.85 | 1.0 | 1.000 / 1.000 |
| scifact/test (300) | repo_exact | 1 | yes (reference) | 87.83 | 100.00 | 353.62 |  |  | 1.0 | 0.997 / 1.000 |
| scifact/test (300) | base_ssrpp | 1 | no (SSR++) | 87.83 | 99.67 | 189.25 | 174.29 | 357.08 | 1.9 | 0.477 / 0.480 |
| scifact/test (300) | repo_ssrpp | 1 | no (SSR++) | 87.83 | 99.67 | 281.97 |  |  | 1.3 | 0.480 / 0.480 (vs base_ssrpp: 0.997) |
| scifact/test (300) | fast_hyb | 1 | yes (bit-identical) | 87.83 | 100.00 | 52.93 | 46.65 | 102.22 | 6.9 | 1.000 / 1.000 |
| scifact/test (300) | fast_hyb | 6 | yes (bit-identical) | 87.83 | 100.00 | 17.44 | 15.08 | 34.69 | 21.0 | 1.000 / 1.000 |
| scifact/test (300) | fast_hyb | 12 | yes (bit-identical) | 87.83 | 100.00 | 12.79 | 10.38 | 29.27 | 28.6 | 1.000 / 1.000 |
| scifact/test (300) | fast_lat | 6 | yes (bit-identical) | 87.83 | 100.00 | 19.98 | 18.11 | 37.10 | 18.3 | 1.000 / 1.000 |
| scifact/test (300) | fast_tok | 6 | yes (bit-identical) | 87.83 | 100.00 | 38.11 | 35.50 | 69.52 | 9.6 | 1.000 / 1.000 |
| scifact/test (300) | fast_ub | 6 | yes (safe pruning) | 87.83 | 100.00 | 33.56 | 32.61 | 47.54 | 10.9 | 1.000 / 1.000 |
| scifact/test (300) | gpu_exact |  | yes (float32, ~1e-6) | 87.83 | 100.00 | 11.59 | 11.36 | 13.20 | 31.6 | 0.000 / 1.000 |
| scifact/test (300) | fast_ssrpp | 1 | no (SSR++) | 87.83 | 99.67 | 44.86 | 43.05 | 62.47 | 8.2 | 0.477 / 0.480 (vs base_ssrpp: 1.000) |
| scifact/test (300) | fast_ssrpp | 6 | no (SSR++) | 87.83 | 99.67 | 14.50 | 14.01 | 21.29 | 25.2 | 0.477 / 0.480 (vs base_ssrpp: 1.000) |
| scifact/test (300) | fast_ssrpp | 12 | no (SSR++) | 87.83 | 99.67 | 10.04 | 9.32 | 16.50 | 36.4 | 0.477 / 0.480 (vs base_ssrpp: 1.000) |
| scifact/test (300) | fast_lb-t4-n1000 | 6 | no (approx) | 87.83 | 99.67 | 13.34 | 12.40 | 22.84 | 27.4 | 0.690 / 0.690 |
| scifact/train (809) | base_exact | 1 | yes (reference) | 69.39 | 92.84 | 364.22 | 337.07 | 664.34 | 1.0 | 1.000 / 1.000 |
| scifact/train (809) | repo_exact | 1 | yes (reference) | 69.39 | 92.84 | 336.67 |  |  | 1.1 | 1.000 / 1.000 |
| scifact/train (809) | base_ssrpp | 1 | no (SSR++) | 69.35 | 91.98 | 181.96 | 165.85 | 336.55 | 2.0 | 0.471 / 0.472 |
| scifact/train (809) | repo_ssrpp | 1 | no (SSR++) | 69.35 | 91.98 | 212.06 |  |  | 1.7 | 0.471 / 0.472 (vs base_ssrpp: 1.000) |
| scifact/train (809) | fast_hyb | 1 | yes (bit-identical) | 69.39 | 92.84 | 50.76 | 46.22 | 88.17 | 7.2 | 1.000 / 1.000 |
| scifact/train (809) | fast_hyb | 6 | yes (bit-identical) | 69.39 | 92.84 | 17.71 | 15.54 | 33.22 | 20.6 | 1.000 / 1.000 |
| scifact/train (809) | fast_hyb | 12 | yes (bit-identical) | 69.39 | 92.84 | 12.80 | 10.36 | 25.73 | 28.5 | 1.000 / 1.000 |
| scifact/train (809) | fast_lat | 6 | yes (bit-identical) | 69.39 | 92.84 | 19.43 | 17.80 | 34.80 | 18.7 | 1.000 / 1.000 |
| scifact/train (809) | fast_tok | 6 | yes (bit-identical) | 69.39 | 92.84 | 36.87 | 34.26 | 65.37 | 9.9 | 1.000 / 1.000 |
| scifact/train (809) | fast_ub | 6 | yes (safe pruning) | 69.39 | 92.84 | 33.95 | 33.18 | 47.89 | 10.7 | 0.999 / 1.000 |
| scifact/train (809) | gpu_exact |  | yes (float32, ~1e-6) | 69.39 | 92.84 | 11.48 | 11.33 | 12.91 | 31.7 | 0.000 / 1.000 |
| scifact/train (809) | fast_ssrpp | 1 | no (SSR++) | 69.35 | 91.98 | 45.09 | 43.28 | 62.84 | 8.1 | 0.471 / 0.472 (vs base_ssrpp: 1.000) |
| scifact/train (809) | fast_ssrpp | 6 | no (SSR++) | 69.35 | 91.98 | 14.18 | 13.90 | 20.49 | 25.7 | 0.471 / 0.472 (vs base_ssrpp: 1.000) |
| scifact/train (809) | fast_ssrpp | 12 | no (SSR++) | 69.35 | 91.98 | 9.52 | 9.02 | 13.94 | 38.3 | 0.471 / 0.472 (vs base_ssrpp: 1.000) |
| scifact/train (809) | fast_lb-t4-n1000 | 6 | no (approx) | 69.39 | 92.59 | 13.44 | 12.83 | 21.37 | 27.1 | 0.665 / 0.665 |

## FiQA test: same first 100 queries

whole-index baseline base_exact: 2332 ms/query (single-threaded numba); repo default chunked exact: 1635 ms/query scoring-only (2019 incl. per-run chunk index builds)

| method | threads | mean ms (first 100) | speedup vs base_exact | speedup vs repo default | top-100 identical to base_exact (bitwise / up to ties) |
|---|---|---|---|---|---|
| base_ssrpp | 1 | 478.6 | 4.9 | 3.4 | 0.01 / 0.01 |
| repo_ssrpp | 1 | 1618.5 | 1.4 | 1.0 | 0.01 / 0.01 |
| fast_hyb | 1 | 325.7 | 7.2 | 5.0 | 1.00 / 1.00 |
| fast_hyb | 6 | 84.1 | 27.7 | 19.4 | 1.00 / 1.00 |
| fast_hyb | 12 | 55.2 | 42.2 | 29.6 | 1.00 / 1.00 |
| fast_lat | 6 | 121.8 | 19.1 | 13.4 | 1.00 / 1.00 |
| fast_tok | 6 | 266.6 | 8.7 | 6.1 | 1.00 / 1.00 |
| fast_ub | 6 | 206.7 | 11.3 | 7.9 | 0.99 / 1.00 |
| gpu_exact |  | 55.9 | 41.7 | 29.3 | 0.00 / 1.00 |
| gpu_exact-i321 |  | 52.6 | 44.4 | 31.1 | 0.00 / 1.00 |
| fast_ssrpp | 1 | 60.8 | 38.4 | 26.9 | 0.01 / 0.01 (vs base_ssrpp: bitwise 1.00) |
| fast_ssrpp | 6 | 19.0 | 122.9 | 86.2 | 0.01 / 0.01 (vs base_ssrpp: bitwise 1.00) |
| fast_ssrpp | 12 | 12.3 | 189.9 | 133.1 | 0.01 / 0.01 (vs base_ssrpp: bitwise 1.00) |

## FiQA approximate variants (quality vs latency, 6 threads)

| split (nq) | method | nDCG@10 | R@100 | overlap@10 vs exact | overlap@100 vs exact | mean ms | p95 ms | latency / fast_hyb |
|---|---|---|---|---|---|---|---|---|
| dev (500) | fast_hyb | 22.80 | 51.38 | 1.000 | 1.000 | 82.22 | 130.93 | 1.00 |
| dev (500) | fast_ssrpp-kc32-n2000 | 16.30 | 29.83 | 0.513 | 0.487 | 30.37 | 43.38 | 0.37 |
| dev (500) | fast_ssrpp-kc32-n5000 | 20.32 | 39.90 | 0.664 | 0.655 | 48.57 | 67.56 | 0.59 |
| dev (500) | fast_ssrpp-kc4-n10000 | 22.25 | 48.74 | 0.884 | 0.868 | 60.12 | 88.95 | 0.73 |
| dev (500) | fast_ssrpp-kc4-n2000 | 18.11 | 33.10 | 0.590 | 0.545 | 18.13 | 26.35 | 0.22 |
| dev (500) | fast_ssrpp-kc4-n5000 | 20.60 | 42.20 | 0.755 | 0.727 | 35.12 | 49.75 | 0.43 |
| dev (500) | fast_ssrpp-kc8-n2000 | 18.01 | 33.53 | 0.581 | 0.540 | 21.51 | 31.29 | 0.26 |
| dev (500) | fast_ssrpp-kc8-n5000 | 20.53 | 41.60 | 0.739 | 0.716 | 39.30 | 56.55 | 0.48 |
| dev (500) | fast_lb-t16-n1000 | 22.80 | 51.38 | 1.000 | 1.000 | 70.86 | 110.00 | 0.86 |
| dev (500) | fast_lb-t2-n1000 | 18.82 | 34.92 | 0.646 | 0.531 | 33.41 | 53.70 | 0.41 |
| dev (500) | fast_lb-t4-n1000 | 22.40 | 46.23 | 0.911 | 0.796 | 42.76 | 69.03 | 0.52 |
| dev (500) | fast_lb-t4-n2000 | 22.58 | 48.52 | 0.957 | 0.900 | 48.67 | 76.48 | 0.59 |
| dev (500) | fast_lb-t4-n500 | 21.03 | 41.49 | 0.833 | 0.662 | 39.40 | 63.08 | 0.48 |
| dev (500) | fast_lb-t8-n1000 | 22.81 | 50.72 | 0.997 | 0.972 | 55.21 | 87.05 | 0.67 |
| dev (500) | fast_lb-t8-n2000 | 22.80 | 51.32 | 1.000 | 0.993 | 61.40 | 95.31 | 0.75 |
| dev (500) | fast_lb-t8-n500 | 22.83 | 50.06 | 0.987 | 0.914 | 52.11 | 79.90 | 0.63 |
| train1k (1000) | fast_hyb | 20.91 | 50.96 | 1.000 | 1.000 | 83.53 | 129.46 | 1.00 |
| train1k (1000) | fast_ssrpp-kc32-n2000 | 17.11 | 31.97 | 0.531 | 0.490 | 31.15 | 43.12 | 0.37 |
| train1k (1000) | fast_ssrpp-kc32-n5000 | 19.64 | 40.35 | 0.680 | 0.657 | 49.45 | 68.12 | 0.59 |
| train1k (1000) | fast_ssrpp-kc4-n10000 | 20.43 | 47.28 | 0.895 | 0.871 | 60.32 | 86.76 | 0.72 |
| train1k (1000) | fast_ssrpp-kc4-n2000 | 17.90 | 34.78 | 0.617 | 0.547 | 18.37 | 27.18 | 0.22 |
| train1k (1000) | fast_ssrpp-kc4-n5000 | 20.07 | 42.93 | 0.771 | 0.729 | 36.23 | 51.52 | 0.43 |
| train1k (1000) | fast_ssrpp-kc8-n2000 | 18.31 | 35.06 | 0.604 | 0.545 | 21.68 | 30.91 | 0.26 |
| train1k (1000) | fast_ssrpp-kc8-n5000 | 19.95 | 43.21 | 0.760 | 0.722 | 39.62 | 56.23 | 0.47 |
| train1k (1000) | fast_lb-t16-n1000 | 20.91 | 50.96 | 1.000 | 1.000 | 71.46 | 107.35 | 0.86 |
| train1k (1000) | fast_lb-t2-n1000 | 17.70 | 34.05 | 0.662 | 0.528 | 34.06 | 52.57 | 0.41 |
| train1k (1000) | fast_lb-t4-n1000 | 20.00 | 44.55 | 0.914 | 0.797 | 43.62 | 69.09 | 0.52 |
| train1k (1000) | fast_lb-t4-n2000 | 20.77 | 47.96 | 0.962 | 0.897 | 49.44 | 75.88 | 0.59 |
| train1k (1000) | fast_lb-t4-n500 | 19.63 | 41.24 | 0.844 | 0.666 | 40.17 | 62.64 | 0.48 |
| train1k (1000) | fast_lb-t8-n1000 | 20.91 | 50.27 | 0.997 | 0.972 | 55.11 | 86.14 | 0.66 |
| train1k (1000) | fast_lb-t8-n2000 | 20.91 | 50.75 | 1.000 | 0.994 | 61.40 | 93.67 | 0.74 |
| train1k (1000) | fast_lb-t8-n500 | 20.89 | 49.07 | 0.991 | 0.916 | 52.57 | 81.63 | 0.63 |
