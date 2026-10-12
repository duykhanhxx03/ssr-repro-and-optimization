"""GPU exhaustive exact SSR (paper Eq. 4) with torch / cuSPARSE.

The corpus is one CSR matrix A (n_doc_tokens x n_latents) on the GPU (doc tokens are
contiguous per document).  Per query (or batch of queries) the query tokens form a dense
matrix Q (n_latents x n_qt); Y = A @ Q gives every (doc token, query token) dot product,
``segment_reduce(max)`` over each document's rows gives max_j <q_i, d_j>, and the sum over
query tokens gives S(Q, D) for all documents.  float32 throughout (TF32 off); the
summation order inside cuSPARSE differs from the CPU kernels, so scores agree with the
baseline to ~1e-6 rather than bit-for-bit.
"""

from __future__ import annotations

import time

import numpy as np
import torch

torch.sparse.check_sparse_tensor_invariants.disable()  # silence the per-tensor warning


class GpuExactIndex:
    def __init__(self, fast_index, device: str = "cuda", index_dtype: str = "int64") -> None:
        """Build from a ``FastIndex`` forward index (no extra host copy of the corpus).

        ``index_dtype``: int64 CSR indices (default).  With int32 indices torch converts
        them to int64 inside every sparse matmul, which made SpMM several times slower."""
        torch.backends.cuda.matmul.allow_tf32 = False
        t0 = time.perf_counter()
        fi = fast_index
        self.device = torch.device(device)
        self.n_docs = fi.n_docs
        self.n_latents = fi.n_latents
        lat = fi.fw_lat
        val = fi.fw_val
        valid_counts = (lat < fi.n_latents).sum(axis=1).astype(np.int64)
        crow = np.zeros(lat.shape[0] + 1, np.int64)
        np.cumsum(valid_counts, out=crow[1:])
        # rows have the padding at the end (or nowhere): keep only valid entries
        mask = lat < fi.n_latents
        cols = lat[mask]  # int16 on the host; widened on the GPU
        vals = val[mask]
        del mask
        tdt = torch.int64 if index_dtype == "int64" else torch.int32
        if tdt == torch.int32 and crow[-1] >= 2**31:
            raise ValueError("nnz >= 2^31: int32 CSR indices overflow")
        self.index_dtype = index_dtype
        self.crow = torch.from_numpy(crow).to(self.device).to(tdt)
        self.col = torch.from_numpy(cols).to(self.device).to(tdt)
        self.val = torch.from_numpy(vals).to(self.device)
        del cols, vals
        self.A = torch.sparse_csr_tensor(  # crow and col share one index dtype
            self.crow, self.col, self.val, size=(lat.shape[0], fi.n_latents),
            device=self.device,
        )
        self.lengths = torch.from_numpy(np.diff(fi.tok_off)).to(self.device)
        self.lat_present = fi.lat_present
        torch.cuda.synchronize()
        self.build_seconds = time.perf_counter() - t0

    def vram_bytes(self) -> int:
        isz = 8 if self.index_dtype == "int64" else 4
        return int(self.crow.numel() * isz + self.col.numel() * isz + self.val.numel() * 4)

    def _dense_queries(self, queries):
        nqs = [q.n_tokens for q in queries]
        tot = int(sum(nqs))
        Q = np.zeros((self.n_latents, max(tot, 1)), np.float32)
        c = 0
        for q in queries:
            ind = np.asarray(q.indices)
            v = np.asarray(q.values, dtype=np.float32)
            ok = (ind >= 0) & (ind < self.n_latents) & (v > 0)
            rows, ks = np.nonzero(ok)
            Q[ind[rows, ks], c + rows] = v[rows, ks]
            c += q.n_tokens
        return Q, nqs

    @torch.inference_mode()
    def scores(self, queries) -> torch.Tensor:
        """Exact Eq. 4 scores, shape (len(queries), n_docs), on the GPU."""
        Q, nqs = self._dense_queries(queries)
        Qg = torch.from_numpy(Q).to(self.device, non_blocking=True)
        Y = torch.sparse.mm(self.A, Qg)  # (n_tok, sum nq)
        M = torch.segment_reduce(Y, "max", lengths=self.lengths, axis=0, unsafe=True)
        del Y
        out = torch.empty((len(queries), self.n_docs), dtype=torch.float32, device=self.device)
        c = 0
        for qi, nq in enumerate(nqs):
            if nq == 0:
                out[qi].zero_()
            else:
                # sequential float32 sum over query tokens (same order as the CPU kernels)
                acc = M[:, c].clone()
                for i in range(1, nq):
                    acc += M[:, c + i]
                out[qi] = acc
            c += nq
        return out

    @torch.inference_mode()
    def topk(self, queries, top_docs: int = 100):
        S = self.scores(queries)
        k = min(int(top_docs), S.shape[1])
        vals, idx = torch.topk(S, k, dim=1, largest=True, sorted=True)
        vals = vals.cpu().numpy()
        idx = idx.cpu().numpy().astype(np.int64)
        out = []
        for i in range(len(queries)):
            keep = vals[i] > 0
            out.append((idx[i][keep], vals[i][keep]))
        return out
