# Tái lập và tối ưu SSR (No More K-means, ICML 2026)

Repo này chứa toàn bộ phần thực nghiệm của bài tập lớn môn **Tìm kiếm và truy xuất
thông tin**, thực hiện trên bài báo:

> L. Guo, Y. Wang, T. Wen, A. Feng, S. Jegelka, C. You.
> *No More K-means: Single-Stage Sparse Coding for Efficient Multi-Vector Retrieval.*
> ICML 2026, PMLR 306. [arXiv:2605.30120](https://arxiv.org/abs/2605.30120) ·
> [PMLR](https://proceedings.mlr.press/v306/guo26x.html)

Nội dung gồm ba phần: (1) chạy lại mã nguồn và checkpoint do tác giả công bố, đối chiếu
mã nguồn với bài báo và sửa các điểm sai lệch; (2) kiểm chứng kết quả trên BEIR và
MS MARCO với ColBERTv2 làm đối chứng; (3) tối ưu thuật toán truy hồi và so sánh với
ColBERTv2 + PLAID trên cùng một máy.

## Tài liệu và liên kết liên quan

| Nội dung | Liên kết |
|---|---|
| Mã nguồn chính thức của tác giả | https://github.com/Y-Research-SBU/SSR (commit `c64bc05`, 26/06/2026, giấy phép MIT) |
| Checkpoint SSR-tok | https://huggingface.co/Y-Research-Group/SSR (đưa lên ngày 05/10/2026) |
| Issue #1: *Difficulty reproducing BEIR numbers from Table 1* | https://github.com/Y-Research-SBU/SSR/issues/1 |
| Issue #2: *Is the 4k reconstruction loss missing from the training objective?* | https://github.com/Y-Research-SBU/SSR/issues/2 |
| Trang dự án | https://y-research-sbu.github.io/SSR/ |

Trong hai issue trên, tác giả đã xác nhận (31/07/2026 và 04/08/2026) rằng mã nguồn công bố
nạp một SAE khởi tạo ngẫu nhiên thay vì checkpoint, chỉ dùng một hard negative cho mỗi
query thay vì 32, và thiếu số hạng ⅛·L_recon(4k) trong hàm mất mát; tác giả cũng cho biết
learning rate thực dùng (khoảng 1e-6) và nhiệt độ của L_CE (0,3) khác với Bảng 6. Tính đến
ngày 12/10/2026, repo gốc chưa có commit sửa lỗi. Các sai lệch khác được phát hiện trong
quá trình thực hiện (công thức chấm điểm trên chỉ mục ngược, ánh xạ chỉ số neuron, bước lọc
thô của SSR++, cách tính nDCG, dựng chỉ mục với corpus lớn) được mô tả trong báo cáo.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `SSR/` | Mã nguồn gốc của tác giả tại commit `c64bc05`, **đã áp dụng bản sửa** của nhóm. File `LICENSE` (MIT) được giữ nguyên. |
| `ssr_paper_fixes.patch` | Bản sửa dưới dạng patch (16 file); áp dụng được trực tiếp lên commit `c64bc05` của repo gốc bằng `git apply`. |
| `SSR/ssr/retrieval/exact_index_maxsim.py` | Kernel chấm điểm MaxSim thưa theo công thức (4) và bước lọc thô của SSR++ theo công thức (11)–(12). |
| `SSR/ssr/samplers.py`, `SSR/scripts/train_table6.sh`, `SSR/scripts/train_issue1_setup.sh` | Batch sampler gom 32 hard negative của cùng một query vào một batch, và hai script huấn luyện (theo Bảng 6 và theo trả lời của tác giả trong issue #1). |
| `SSR/tests/` | Unit test cho kernel exact, SSR++ và batch sampler. |
| `run_eval.py` | Đánh giá checkpoint công bố trên BEIR thông qua `ssr.retrieval.eval_mteb`. |
| `crosscheck.py`, `verify_scipy.py`, `verify_bug.py` | Ba cách tính độc lập dùng để phát hiện và xác nhận lỗi chấm điểm. |
| `eval_dense_colbert.py`, `eval_backbone_dense.py` | Đối chứng ColBERTv2 và backbone BERT của chính checkpoint (bỏ tầng SAE) trên từng split. |
| `quora_ssr.py`, `gpu_exact_score.py` | Encode theo batch và chấm điểm exact trên GPU cho corpus lớn (Quora). |
| `msmarco_rerank.py` | BM25 (Lucene, k1 = 0,9, b = 0,4) lấy top-100 và re-rank bằng SSR, ColBERTv2 và backbone trên MS MARCO dev-small và TREC DL 2019. |
| `plaid_bench.py`, `verify_fast_hyb.py` | So sánh với ColBERTv2 + PLAID (fast-plaid) trên FiQA; kiểm tra kernel tối ưu. |
| `opt/` | Phần tối ưu: `ssr_opt/fast_exact.py` (kernel `fast_hyb`, LB-rerank, SSR++ tăng tốc), `ssr_opt/gpu_exact.py`, script benchmark `bench.py`, `tests/`, kết quả trong `results/`. |
| `results/`, `logs/` | Kết quả (JSON) và log của các lần chạy được trích dẫn trong báo cáo. |

Các thư mục sau không có trong repo, cần tạo lại theo hướng dẫn bên dưới:
`data/` (BEIR, MS MARCO), `ckpt/` (checkpoint), `cache/` (embedding đã encode), `venv/`.

## Môi trường

Thực nghiệm được thực hiện trên Intel i5-12400F (6 nhân / 12 luồng), 32 GB RAM,
NVIDIA RTX 5060 8 GB, Windows 11, với Python 3.13, PyTorch 2.11 (CUDA 12.8),
transformers 5.3, sentence-transformers 5.3, PyLate 1.6, numba 0.61.

```bash
python -m venv venv
venv/Scripts/activate                      # Linux/macOS: source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e "SSR[eval,data]" pylate sentence-transformers transformers numba scipy pytest
```

## Dữ liệu và checkpoint

```bash
# BEIR theo định dạng MTEB (corpus.jsonl, queries/<split>.tsv, qrels/<split>.tsv) vào data/mteb/<dataset>/
python SSR/prepare_mteb_eval.py --help

# Checkpoint SSR-tok
huggingface-cli download Y-Research-Group/SSR --include "SSR-tok/*" --local-dir ckpt

# MS MARCO passage: collection.tsv, queries.dev.small.tsv, qrels dev và TREC DL 2019 vào data/msmarco_raw/
```

Script chuẩn bị dữ liệu của tác giả chỉ tải split test. Các split train/dev của NFCorpus,
SciFact, FiQA và Quora được lấy từ bộ BEIR gốc và đặt cùng thư mục với split test.

## Chạy lại các thí nghiệm

```bash
# 1. Đánh giá checkpoint công bố bằng mã đã sửa (MaxSim theo công thức (4))
python -m ssr.retrieval.eval_mteb --task retrieval --variant ssr \
    --model-path ckpt/SSR-tok --dataset nfcorpus --data-dir data/mteb \
    --mode exact --score-device index --split test        # nDCG@10 = 0.4397

# 2. Kiểm tra theo split và các mô hình đối chứng
python run_eval.py --ckpt ckpt/SSR-tok --token-scope non-cls -- \
    --task retrieval --variant ssr --dataset scifact --data-dir data/mteb --split train
python eval_dense_colbert.py lightonai/colbertv2.0 scifact train,test
python eval_backbone_dense.py scifact train,test
python quora_ssr.py --dataset quora --splits test,dev --ignore-identical-ids

# 3. Ba cách tính độc lập xác nhận lỗi chấm điểm
python crosscheck.py --ckpt ckpt/SSR-tok --data-dir data/mteb --dataset nfcorpus \
    --corpus-npz cache/A/nfcorpus/corpus_32k_test.npz
python verify_scipy.py nfcorpus cache/A/nfcorpus/corpus_32k_test.npz
python verify_bug.py nfcorpus cache/A/nfcorpus/corpus_32k_test.npz

# 4. MS MARCO: BM25 top-100 rồi re-rank
python msmarco_rerank.py bm25
python msmarco_rerank.py rerank --models ssr colbertv2 backbone

# 5. Hiệu năng: PLAID và kernel tối ưu trên FiQA
python plaid_bench.py
python verify_fast_hyb.py nfcorpus test cache/A/nfcorpus/corpus_32k_test.npz
bash opt/run_all.sh fiqa_fast fiqa_sweep
python opt/bench.py --dataset fiqa --split dev --methods fast_hyb fast_ssrpp --threads 6

# 6. Chạy test
python -m pytest SSR/tests -q
python opt/tests/test_fast_exact.py
```

## Kết quả chính

nDCG@10 trên split test (graded relevance theo chuẩn BEIR):

| Tập dữ liệu | Bài báo | Mã gốc | Mã đã sửa | ColBERTv2 (đo lại) |
|---|---:|---:|---:|---:|
| NFCorpus | 38,8 | 8,3 | 43,97 | 34,40 |
| SciFact | 72,9 | 7,5 | 87,83 | 69,16 |
| FiQA-2018 | 39,4 | – | 41,41 | 35,80 |
| Quora | 88,3 | – | 85,42 | 85,67 |

Trên split train/dev của cùng tập dữ liệu, checkpoint công bố thấp hơn ColBERTv2 từ 0,7
đến 12,6 điểm; mức tăng chỉ xuất hiện ở split test. Trên MS MARCO dev-small (re-rank
BM25 top-100), SSR-tok đạt MRR@10 23,18 so với 35,86 của ColBERTv2.

Phần tối ưu: kernel `fast_hyb` cho kết quả xếp hạng trùng khớp hoàn toàn với bản gốc,
đạt 54 ms/query trên FiQA (12 luồng) so với 1.635 ms của mã đã sửa; thời gian dựng chỉ
mục giảm từ khoảng 280 s xuống 4,8 s. Phương án xấp xỉ LB-rerank (t = 8, 1.000 ứng viên)
giữ 99,7 % top-10 của phép tính exact với 0,67 lần thời gian.

Các bảng số liệu đầy đủ, phân tích và đoạn mã trước/sau khi sửa được trình bày trong
báo cáo của nhóm.

## Giấy phép

Thư mục `SSR/` giữ nguyên giấy phép MIT của tác giả gốc. Phần code còn lại trong repo này
cũng dùng giấy phép MIT.
