"""Build the comparison table from opt/results/runs/*.json|npz.

Exactness of every run is checked against the reference run of the same dataset/split
(``base_exact`` if present with the same queries, else ``fast_hyb`` / ``fast_lat`` which are
verified bit-identical to it):
  * bitwise      : fraction of queries whose top-k ids AND scores are identical
  * tie_equal    : rank-aligned scores within 1e-4 and the id sets differ only by docs whose
                   score is within 1e-4 of the k-th score (permutation among exact ties)
  * max_dscore   : max |score diff| over rank-aligned positions
  * overlap@10/100 (for approximate methods)

Writes opt/results/comparison.{csv,json,md}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OPT = Path(__file__).resolve().parent
RUNS = OPT / "results" / "runs"
OUT = OPT / "results"


def compare(ref_ids, ref_sc, ids, sc, tol=1e-4):
    nq = ref_ids.shape[0]
    bit = tie = 0
    maxd = 0.0
    ov10 = []
    ov100 = []
    for i in range(nq):
        a_i, a_s = ref_ids[i], ref_sc[i]
        b_i, b_s = ids[i], sc[i]
        ka = int((a_i >= 0).sum())
        kb = int((b_i >= 0).sum())
        if ka == kb and np.array_equal(a_i, b_i) and np.array_equal(a_s[:ka], b_s[:kb]):
            bit += 1
        if ka == kb and ka > 0:
            d = float(np.abs(a_s[:ka] - b_s[:kb]).max())
            maxd = max(maxd, d)
            sa, sb = set(a_i[:ka].tolist()), set(b_i[:kb].tolist())
            boundary = float(a_s[ka - 1])
            diff_ok = True
            sym = sa ^ sb
            if sym:
                score_of = {int(x): float(s) for x, s in zip(a_i[:ka], a_s[:ka])}
                score_of.update({int(x): float(s) for x, s in zip(b_i[:kb], b_s[:kb])})
                diff_ok = all(abs(score_of[x] - boundary) <= tol for x in sym)
            if d <= tol and diff_ok:
                tie += 1
        elif ka == kb == 0:
            tie += 1
        k10 = min(10, ka)
        ov10.append(len(set(a_i[:k10].tolist()) & set(b_i[:min(10, kb)].tolist())) / max(k10, 1))
        ov100.append(len(set(a_i[:ka].tolist()) & set(b_i[:kb].tolist())) / max(ka, 1))
    return {"bitwise_frac": bit / nq, "tie_equal_frac": tie / nq, "max_dscore": maxd,
            "overlap@10": float(np.mean(ov10)), "overlap@100": float(np.mean(ov100))}


def main():
    recs = []
    for jf in sorted(RUNS.glob("*.json")):
        r = json.loads(jf.read_text())
        r["_npz"] = str(jf.with_suffix(".npz"))
        recs.append(r)
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["dataset"], r["split"], r["n_queries"]), []).append(r)
    rows = []
    for (ds, sp, nq), rs in sorted(groups.items()):
        ref = next((r for r in rs if r["method"] == "base_exact"), None)
        for alt in ("fast_hyb", "fast_lat"):  # verified bit-identical to base_exact
            if ref is None:
                ref = next((r for r in rs if r["method"] == alt), None)
        ref_npz = np.load(ref["_npz"]) if ref else None
        base_lat = None
        hyb6 = None
        for r in rs:
            if r["method"] == "base_exact" and r.get("latency"):
                base_lat = r["latency"]["mean_ms"]
            if r["method"] == "fast_hyb" and r.get("threads") == 6 and r.get("latency"):
                hyb6 = r["latency"]["mean_ms"]
        for r in rs:
            z = np.load(r["_npz"])
            cmp = compare(ref_npz["ids"], ref_npz["scores"], z["ids"], z["scores"]) if ref else {}
            # bitwise check against the baseline twin of an approximate method (same params)
            twin_name = None
            if r["method"].startswith("fast_ssrpp"):
                twin_name = r["method"].replace("fast_ssrpp", "base_ssrpp")
            elif r["method"].startswith("repo_ssrpp"):
                twin_name = r["method"].replace("repo_ssrpp", "base_ssrpp")
            twin = next((t for t in rs if t["method"] == twin_name), None)
            if twin is not None:
                tz = np.load(twin["_npz"])
                tc = compare(tz["ids"], tz["scores"], z["ids"], z["scores"])
                cmp["twin"] = twin_name
                cmp["twin_bitwise_frac"] = tc["bitwise_frac"]
                cmp["twin_tie_equal_frac"] = tc["tie_equal_frac"]
            lat = r.get("latency") or {}
            mean = lat.get("mean_ms", r.get("mean_ms_scoring_only", r.get("mean_ms_incl_index_build")))
            rows.append({
                "dataset": ds, "split": sp, "n_queries": nq, "method": r["method"],
                "threads": r["threads"], "exact": r["exact"],
                "ndcg@10": round(100 * r["ndcg@10"], 2), "recall@100": round(100 * r["recall@100"], 2),
                "lat_mean_ms": round(mean, 2) if mean is not None else None,
                "lat_p50_ms": round(lat["p50_ms"], 2) if lat else None,
                "lat_p95_ms": round(lat["p95_ms"], 2) if lat else None,
                "speedup_vs_base_exact": round(base_lat / mean, 1) if (base_lat and mean) else None,
                "lat_rel_fast_hyb_t6": round(mean / hyb6, 2) if (hyb6 and mean) else None,
                "build_s": round(r["index"].get("build_s") or 0.0, 1),
                "peak_gb": round(r.get("peak_gb", 0.0), 2),
                "ref": ref["method"] if ref else None,
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cmp.items()},
                "other_eval_procs": len(r.get("other_eval_procs") or []),
                "tag": r.get("tag", ""),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    if rows:
        keys = []
        for row in rows:
            keys += [k for k in row if k not in keys]
        with open(OUT / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    cols = ["dataset", "split", "n_queries", "method", "threads", "exact", "ndcg@10", "recall@100",
            "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "speedup_vs_base_exact",
            "lat_rel_fast_hyb_t6", "build_s", "peak_gb",
            "bitwise_frac", "tie_equal_frac", "max_dscore", "overlap@10", "overlap@100",
            "twin", "twin_bitwise_frac", "twin_tie_equal_frac", "other_eval_procs"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "-q" not in sys.argv:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
