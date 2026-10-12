"""Compact final tables from opt/results/comparison.json + per-query latency arrays.

Writes opt/results/summary.md and opt/results/summary.json:
  * main table: baseline vs optimised methods per dataset/split
  * FiQA speedups computed on the *same* first-100 test queries (whole-index baseline was
    only run on those 100 queries; per-query latencies of the full-test runs are sliced)
  * FiQA quality/latency trade-off of approximate variants (dev / train1k, test secondary)
Run report.py first.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np

OPT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPT))
RES = OPT / "results"
RUNS = RES / "runs"

MAIN_METHODS = [
    ("base_exact", 1), ("repo_exact", 1), ("base_ssrpp", 1), ("repo_ssrpp", 1),
    ("fast_hyb", 1), ("fast_hyb", 6), ("fast_hyb", 12), ("fast_lat", 6),
    ("fast_tok", 6), ("fast_ub", 6), ("gpu_exact", None),
    ("gpu_exact-i321", None), ("fast_ssrpp", 1), ("fast_ssrpp", 6), ("fast_ssrpp", 12),
    ("fast_lb-t4-n1000", 6),
]


def load_rows():
    return json.loads((RES / "comparison.json").read_text())


def lat_array(ds, sp, method, th):
    f = RUNS / f"{ds}_{sp}__{method}__t{th if th is not None else 0}.npz"
    if not f.is_file():
        return None
    z = np.load(f)
    return z["latency"] if z["latency"].size else None


def fmt(v, nd=2):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    rows = load_rows()
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["split"], r["n_queries"]), []).append(r)
    out_lines = []
    summary = {"main": [], "fiqa_same100": [], "fiqa_tradeoff": []}

    # ---------------------------------------------------------------- main table
    hdr = ("| dataset/split (nq) | method | threads | exact? | nDCG@10 | R@100 | mean ms | p50 | p95 "
           "| speedup vs base_exact | identical top-100 (bitwise / up to ties) |")
    out_lines += ["## Main comparison", "", hdr, "|" + "---|" * 11]
    for key in sorted(by):
        ds, sp, nq = key
        if ds == "fiqa" and sp != "test":
            continue
        rs = by[key]
        for m, th in MAIN_METHODS:
            r = next((x for x in rs if x["method"] == m and (th is None or x["threads"] == th)), None)
            if r is None:
                continue
            ident = (f"{r.get('bitwise_frac', ''):.3f} / {r.get('tie_equal_frac', ''):.3f}"
                     if isinstance(r.get("bitwise_frac"), float) else "")
            if r.get("twin"):
                ident += f" (vs {r['twin']}: {r['twin_bitwise_frac']:.3f})"
            line = (f"| {ds}/{sp} ({nq}) | {m} | {fmt(r['threads'])} | {r['exact']} | "
                    f"{r['ndcg@10']:.2f} | {r['recall@100']:.2f} | {fmt(r['lat_mean_ms'])} | "
                    f"{fmt(r['lat_p50_ms'])} | {fmt(r['lat_p95_ms'])} | "
                    f"{fmt(r['speedup_vs_base_exact'], 1)} | {ident} |")
            out_lines.append(line)
            summary["main"].append(r)

    # ---------------------------------------------------------------- FiQA same-100 speedups
    sub = {(r["method"], r["threads"]): r for r in by.get(("fiqa", "test", 100), [])}
    full = {(r["method"], r["threads"]): r for r in by.get(("fiqa", "test", 648), [])}
    base_rec = sub.get(("base_exact", 1))
    repo_rec = sub.get(("repo_exact", 1))
    if base_rec is not None:
        bz = np.load(RUNS / "fiqa_test__base_exact__t1.npz")
        b100 = float(bz["latency"].mean() * 1000)
        out_lines += ["", "## FiQA test: same first 100 queries", "",
                      f"whole-index baseline base_exact: {b100:.0f} ms/query (single-threaded numba); "
                      + (f"repo default chunked exact: {repo_rec['lat_mean_ms']:.0f} ms/query scoring-only "
                         f"({json.loads((RUNS / 'fiqa_test__repo_exact__t1.json').read_text())['mean_ms_incl_index_build']:.0f}"
                         " incl. per-run chunk index builds)" if repo_rec else ""), "",
                      "| method | threads | mean ms (first 100) | speedup vs base_exact | speedup vs repo default "
                      "| top-100 identical to base_exact (bitwise / up to ties) |", "|---|---|---|---|---|---|"]
        from report import compare
        for m, th in MAIN_METHODS:
            if m in ("base_exact", "repo_exact"):
                continue
            key = (m, th)
            if th is None:
                key = next((k for k in full if k[0] == m), None) or next((k for k in sub if k[0] == m), None)
            if key in full:
                f = RUNS / f"fiqa_test__{m}__t{key[1] if key[1] is not None else 0}.npz"
                z = np.load(f)
                if z["ids"].shape[0] != 648:
                    continue
                mean100 = float(z["latency"][:100].mean() * 1000)
                c = compare(bz["ids"], bz["scores"], z["ids"][:100], z["scores"][:100])
            elif key in sub:
                r = sub[key]
                mean100 = r["lat_mean_ms"]
                c = {"bitwise_frac": r.get("bitwise_frac"), "tie_equal_frac": r.get("tie_equal_frac")}
            else:
                continue
            sp_b = b100 / mean100
            sp_r = (repo_rec["lat_mean_ms"] / mean100) if repo_rec else None
            extra = ""
            if m == "fast_ssrpp" and key in full and ("base_ssrpp", 1) in sub:
                tz = np.load(RUNS / "fiqa_test__base_ssrpp__t1.npz")
                tc = compare(tz["ids"], tz["scores"], z["ids"][:100], z["scores"][:100])
                extra = f" (vs base_ssrpp: bitwise {tc['bitwise_frac']:.2f})"
                c["twin_bitwise_frac"] = tc["bitwise_frac"]
            out_lines.append(f"| {m} | {fmt(key[1])} | {mean100:.1f} | {sp_b:.1f} | {fmt(sp_r, 1)} | "
                             f"{c['bitwise_frac']:.2f} / {c['tie_equal_frac']:.2f}{extra} |")
            summary["fiqa_same100"].append({"method": m, "threads": key[1], "mean_ms": mean100,
                                            "speedup_vs_base_exact": sp_b,
                                            "speedup_vs_repo_exact": sp_r, **c})

    # ---------------------------------------------------------------- FiQA trade-off
    out_lines += ["", "## FiQA approximate variants (quality vs latency, 6 threads)", "",
                  "| split (nq) | method | nDCG@10 | R@100 | overlap@10 vs exact | overlap@100 vs exact "
                  "| mean ms | p95 ms | latency / fast_hyb |", "|" + "---|" * 9]
    for key in sorted(by):
        ds, sp, nq = key
        if ds != "fiqa":
            continue
        rs = by[key]
        if not any(r["method"].startswith(("fast_lb", "fast_ssrpp-")) for r in rs):
            continue
        order = sorted(rs, key=lambda r: (not r["method"].startswith("fast_hyb"),
                                          r["method"].startswith("fast_lb"), r["method"]))
        for r in order:
            if r["threads"] not in (6, None) or r["method"].startswith(("gpu", "fast_lat", "fast_tok", "fast_ub")):
                continue
            out_lines.append(
                f"| {sp} ({nq}) | {r['method']} | {r['ndcg@10']:.2f} | {r['recall@100']:.2f} | "
                f"{fmt(r.get('overlap@10'), 3)} | {fmt(r.get('overlap@100'), 3)} | {fmt(r['lat_mean_ms'])} | "
                f"{fmt(r['lat_p95_ms'])} | {fmt(r.get('lat_rel_fast_hyb_t6'))} |")
            summary["fiqa_tradeoff"].append(r)
    (RES / "summary.md").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
