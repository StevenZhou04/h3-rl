"""Reward sanity check: every reward term should prefer real rollouts over deliberately degraded copies.

  python scripts/reward_sanity.py <rollouts dir> --gpus 0 1 2 3 [--n 16] [--workers videoalign hpspp ...]

Takes up to N rollouts (<key>.mp4 + <key>.json with the prompt, as h3rl.train writes them), makes degraded copies
with ffmpeg, scores everything with the reward workers, and prints for each term and degradation how often the original
scores higher (win rate) and the mean score drop. Expected drops are marked: a term that does not prefer the original
where it should, or prefers it where it should not care, points at a broken worker (preprocessing, prompt, sign).
"""
from __future__ import annotations
import argparse, glob, json, os, random, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3rl.rewards import queue as Q
from h3rl.rewards.procs import Workers
from h3rl.rewards.registry import WORKERS

DEGRADE = {   # name -> ffmpeg video filter (None: same video), terms expected to drop
    "shuffled": ("random=frames=64:seed=0", {"flow_coherence", "va_mq", "soli_phys", "cut_free", "ur_physics"}),
    "frozen": ("select='eq(n\\,0)',loop=loop=-1:size=1:start=0,setpts=N/FRAME_RATE/TB", {"flow_motion", "flow_raw", "va_mq"}),
    "blurred": ("gblur=sigma=10", {"hps", "hps_min", "va_vq", "ur_style"}),
    "noisy": ("noise=alls=60:allf=t", {"hps", "hps_min", "va_vq"}),
    "wrong_prompt": (None, {"va_ta", "soli_ta", "ur_align"}),
}


def degrade(src: str, dst: str, vf: str | None, dur: float) -> bool:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src]
    cmd += (["-vf", vf, "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy"] if vf else ["-c", "copy"])
    return subprocess.run(cmd + [dst]).returncode == 0 and os.path.getsize(dst) > 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rollouts"); ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--workers", nargs="*", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    vids = sorted(p for p in glob.glob(f"{a.rollouts}/**/*.mp4", recursive=True) if os.path.exists(p[:-4] + ".json"))
    random.Random(0).shuffle(vids); vids = vids[:a.n]
    if len(vids) < 2: sys.exit(f"need at least 2 rollouts with .json next to them under {a.rollouts}")
    out = Path(a.out or tempfile.mkdtemp(prefix="reward_sanity_")); (out / "videos").mkdir(parents=True, exist_ok=True)
    workers = a.workers or [w for w in WORKERS if WORKERS[w].get("kind", "video") == "video"]
    group_workers = [w for w in workers if WORKERS[w].get("kind") == "group"]; video_workers = [w for w in workers if w not in group_workers]
    prompts = [json.load(open(v[:-4] + ".json"))["prompt"] for v in vids]
    items = []   # (key, mp4, prompt, video index, variant)
    for i, (v, pr) in enumerate(zip(vids, prompts)):
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v], capture_output=True, text=True).stdout or 5)
        items.append((f"v{i:03d}_orig", v, pr, i, "orig"))
        for name, (vf, _) in DEGRADE.items():
            if vf is None:
                items.append((f"v{i:03d}_{name}", v, prompts[(i + 1) % len(prompts)], i, name)); continue
            dst = str(out / "videos" / f"v{i:03d}_{name}.mp4")
            if degrade(v, dst, vf, dur): items.append((f"v{i:03d}_{name}", dst, pr, i, name))
    print(f"{len(vids)} rollouts, {len(items)} videos to score with {workers} on GPUs {a.gpus} | {out}", flush=True)
    qd = out / "queue"; ws = Workers(workers, qd, a.gpus)
    try:
        ws.wait_loaded()
        for key, mp4, pr, _, _ in items: Q.submit(str(qd), key, mp4, pr, video_workers)
        gkeys = []   # pairwise judges: one group (original, degraded) per pair
        for key, mp4, pr, i, var in items:
            if var == "orig" or not group_workers: continue
            gk = f"g_{key}"; orig = next(x for x in items if x[3] == i and x[4] == "orig"); gkeys.append((gk, orig[0], key, i, var))
            for w in group_workers: Q.submit_group(str(qd), gk, [{"key": orig[0], "mp4": orig[1]}, {"key": key, "mp4": mp4}], orig[2], w)
        got = Q.collect(str(qd), [x[0] for x in items], video_workers, a.timeout) if video_workers else {}
        ggot = Q.collect(str(qd), [g[0] for g in gkeys], group_workers, a.timeout) if gkeys else {}
    finally:
        ws.stop()
    scores = {}   # (i, variant) -> {term: value}
    errors = {}
    for key, _, _, i, var in items:
        d = {}
        for w, res in got.get(key, {}).items():
            d.update(res.get("scores") or {})
            if res.get("error"): errors[w] = errors.get(w, 0) + 1
        scores[(i, var)] = d
    for gk, ok, dk, i, var in gkeys:
        for w, res in ggot.get(gk, {}).items():
            m = res.get("members") or {}
            for t, v in (m.get(ok) or {}).items(): scores[(i, "orig")].setdefault(f"{t}@{var}", v)
            for t, v in (m.get(dk) or {}).items(): scores[(i, var)][f"{t}@{var}"] = v
            if res.get("error"): errors[w] = errors.get(w, 0) + 1
    terms = sorted({t for d in scores.values() for t in d})
    print(f"\nworker errors: {errors or 'none'}")
    print(f"\n{'term':<22}" + "".join(f"{v:>16}" for v in DEGRADE) + "     (win rate of the original | mean drop; * = expected to drop)")
    bad = []
    for t in terms:
        base = t.split("@")[0]; row = f"{t:<22}"
        for var, (_, expect) in DEGRADE.items():
            pairs = [(scores[(i, 'orig')][t], scores[(i, var)][t]) for i in range(len(vids))
                     if (i, var) in scores and t in scores[(i, 'orig')] and t in scores[(i, var)]]
            if not pairs or ("@" in t and not t.endswith("@" + var)): row += f"{'-':>16}"; continue
            win = float(np.mean([o > d for o, d in pairs])); drop = float(np.mean([o - d for o, d in pairs]))
            mark = "*" if base in expect or "@" in t else " "
            row += f"{win:>7.2f} {drop:>+7.3f}{mark}"
            if mark == "*" and win < 0.6: bad.append(f"{t} vs {var}: win {win:.2f}")
        print(row)
    print("\nsuspicious (expected drop, original wins < 60%):", bad or "none")
    json.dump({f"{i}|{v}": d for (i, v), d in scores.items()}, open(out / "scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
