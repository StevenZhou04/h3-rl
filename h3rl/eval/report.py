"""Merge manifest + VideoAlign + local scores for one or more eval runs; print per-run means and, with
--ref, PAIRED deltas vs the reference run over identical (prompt, seed) keys with a bootstrap CI."""
import argparse, json, os, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--ref", default=None); a = ap.parse_args()
METRICS = ["VQ", "MQ", "TA", "dino_adjacent", "dino_first_last", "ff_dino", "ff_lpips", "black_frame_free", "flicker_free", "motion_mean"]
def load(run):
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(f"{run}/manifest.jsonl")}
    for f in ("scores_videoalign.jsonl", "scores_local.jsonl"):
        if os.path.exists(f"{run}/{f}"):
            for l in open(f"{run}/{f}"):
                s = json.loads(l); rows.get(s["key"], {}).update(s)
    return rows
runs = {r: load(r) for r in a.runs}; ref = load(a.ref) if a.ref else None
def group_means(rows):
    out = {}
    for m in METRICS:
        v = [r[m] for r in rows.values() if m in r and r[m] is not None]
        if v: out[m] = (float(np.mean(v)), len(v))
    return out
for run, rows in runs.items():
    gm = group_means(rows); print(f"\n== {os.path.basename(run.rstrip('/'))}  n={len(rows)}")
    print("  " + "  ".join(f"{m}={v:.4f}(n={n})" for m, (v, n) in gm.items()))
    by = collections.defaultdict(dict)
    for k, r in rows.items(): by[r.get("source") or r.get("variant") or r.get("task")][k] = r
    for g, sub in sorted(by.items()): print(f"  [{g}] n={len(sub)} " + " ".join(f"{m}={v:.3f}" for m, (v, n) in group_means(sub).items() if m in ("VQ", "MQ", "TA", "dino_adjacent")))
    if ref:
        common = [k for k in rows if k in ref]; print(f"  paired vs ref over {len(common)} keys:")
        rng = np.random.default_rng(0)
        for m in METRICS:
            d = np.array([rows[k][m] - ref[k][m] for k in common if m in rows[k] and m in ref[k]])
            if len(d) < 2: continue
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]; lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"    {m:16s} delta={d.mean():+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  win={np.mean(d > 0):.2f}  n={len(d)}")
