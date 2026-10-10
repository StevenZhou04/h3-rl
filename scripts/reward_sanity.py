"""Reward sanity check: does every reward term react to what it should, and only to that?

  python scripts/reward_sanity.py "<glob of rollout mp4s>" [...] --gpus 4 5 6 7 [--n 36] [--workers videoalign hpspp ...]

Takes up to N rollouts (<key>.mp4 + <key>.json with the prompt, as h3rl.train writes them; pass a glob of the _k0
rollouts for one per prompt group) and makes variants with ffmpeg. Every variant, including a "control", is re-encoded with the same x264 settings,
and terms are compared against the control (not the original file), so a judge cannot tell variants apart by encoding.

  control                   re-encode only (the baseline)
  retest                    the control file again under another key: the term must give the same score (determinism)
  blur2/5/10, noise15/30/60, crf38/48, lowres4/8     graded visual damage: quality terms should drop, monotonically
  frozen_half, frozen       play the first half then hold / hold the middle frame throughout: motion terms should drop
  shuf8, shuf64             local frame shuffles (window 8 / 64): temporal-coherence terms should drop
  reversed                  played backwards (informative: physics judges may or may not notice)
  wrong_prompt              control video scored against another rollout's prompt: alignment terms should drop
  trimmed                   (10 s clips) the first 124 frames, as long as a 5 s rollout: a length bias shows up here
  resized                   rescaled to the other training canvas (544x960 <-> 480x832): a resolution bias shows up here

For each term and variant it prints the win rate of the control (ties 1/2), the mean drop in units of the term's spread
across videos (sd of the control scores), and a paired sign-flip permutation p-value, then lists:
  MISS       expected to drop, but does not (win < 0.6 or not significant)
  WRONG WAY  expected to drop, rises significantly (the judge prefers the damaged video: exploitable)
  SENSITIVE  not expected to move, moves significantly by > 0.3 sd (bias or a judge that sees something unexpected)
  NOT MONOTONE  a stronger degradation of the same kind drops less (by more than 2 standard errors)
  NONDETERMINISTIC  retest differs from control by more than 2% of the term's spread
"""
from __future__ import annotations
import argparse, glob, json, os, random, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3rl.rewards import queue as Q
from h3rl.rewards.procs import Workers
from h3rl.rewards.registry import WORKERS

QUALITY = {"hps", "hps_min", "va_vq", "ur_style"}
MOTION = {"flow_motion", "flow_raw", "va_mq", "ur_physics", "soli_phys"}
TEMPORAL = {"flow_coherence", "va_mq", "ur_physics", "soli_phys"}
ALIGN = {"va_ta", "soli_ta", "ur_align"}
# name -> (family, strength, video filter template or None (same file as control), terms expected to drop)
VARIANTS = {
    "retest": ("retest", 0, None, set()),
    "blur2": ("blur", 1, "gblur=sigma=2", QUALITY), "blur5": ("blur", 2, "gblur=sigma=5", QUALITY), "blur10": ("blur", 3, "gblur=sigma=10", QUALITY),
    "noise15": ("noise", 1, "noise=alls=15:allf=t", QUALITY), "noise30": ("noise", 2, "noise=alls=30:allf=t", QUALITY),
    "noise60": ("noise", 3, "noise=alls=60:allf=t", QUALITY),
    "crf38": ("crf", 1, "null", QUALITY), "crf48": ("crf", 2, "null", QUALITY),
    "lowres4": ("lowres", 1, "scale={w4}:{h4}:flags=area,scale={W}:{H}:flags=bilinear", QUALITY),
    "lowres8": ("lowres", 2, "scale={w8}:{h8}:flags=area,scale={W}:{H}:flags=bilinear", QUALITY),
    "frozen_half": ("frozen", 1, "trim=end_frame={half},tpad=stop_mode=clone:stop={rest_half}", MOTION),
    "frozen": ("frozen", 2, "trim=start_frame={half}:end_frame={half1},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={rest1}", MOTION),
    "shuf8": ("shuf", 1, "random=frames=8:seed=1", TEMPORAL), "shuf64": ("shuf", 2, "random=frames=64:seed=1", TEMPORAL),
    "reversed": ("reversed", 0, "reverse", set()),
    "wrong_prompt": ("wrong_prompt", 0, None, ALIGN),
    "trimmed": ("trimmed", 0, "trim=end_frame=124", set()),
    "resized": ("resized", 0, "scale={RW}:{RH}:flags=bicubic", set()),
}
THINK_VARIANTS = ["blur10", "noise60", "frozen", "shuf64", "reversed"]
CANVAS = {(960, 544): (832, 480), (832, 480): (960, 544)}


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", path], capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]; n, d = s["r_frame_rate"].split("/")
    return {"W": int(s["width"]), "H": int(s["height"]), "N": int(s["nb_read_frames"]), "fps": float(n) / float(d)}


def encode(src: str, dst: str, vf: str, info: dict, crf: int = 14, frames: int | None = None) -> str | None:
    n = frames or info["N"]
    cmd = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vf", vf, "-frames:v", str(n), "-t", f"{n / info['fps']:.4f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", f"{info['fps']:g}", "-threads", "4", "-c:a", "copy", dst]
    if subprocess.run(cmd).returncode != 0 or not os.path.exists(dst): return f"ffmpeg failed: {' '.join(cmd)}"
    got = probe(dst)
    if got["N"] != n: return f"{dst}: {got['N']} frames, expected {n}"
    return None


def perm_p(d: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test of mean(d) == 0."""
    if len(d) < 2 or np.all(d == 0): return 1.0
    rng = np.random.default_rng(seed); s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rollouts", nargs="+", help="glob(s) of rollout mp4s")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0]); ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--n", type=int, default=36); ap.add_argument("--think_n", type=int, default=12, help="originals judged pairwise")
    ap.add_argument("--workers", nargs="*", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=4 * 3600); ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--rl_progress", type=float, default=0.5, help="training progress passed to the workers (HPS iter_step ramp)")
    a = ap.parse_args()
    vids = sorted({os.path.abspath(p) for g in a.rollouts for p in glob.glob(g, recursive=True) if os.path.exists(p[:-4] + ".json")})
    random.Random(0).shuffle(vids); vids = sorted(vids[:a.n])
    if len(vids) < 4: sys.exit(f"need at least 4 rollouts with .json next to them, got {len(vids)}")
    out = Path(a.out or tempfile.mkdtemp(prefix="reward_sanity_")).resolve()
    if (out / "queue").exists(): sys.exit(f"{out} holds an earlier run (its results would be collected again); use a new --out")
    vd = out / "videos"; vd.mkdir(parents=True, exist_ok=True); meta = {"rl_progress": a.rl_progress}
    workers = a.workers or list(WORKERS)
    group_workers = [w for w in workers if WORKERS[w].get("kind") == "group"]; video_workers = [w for w in workers if w not in group_workers]
    infos = [probe(v) for v in vids]; prompts = [json.load(open(v[:-4] + ".json"))["prompt"] for v in vids]
    by_len = {}
    for i, inf in enumerate(infos): by_len.setdefault(inf["N"], []).append(i)

    jobs, items = [], []   # items: (key, mp4, prompt, video index, variant)
    for i, (v, inf) in enumerate(zip(vids, infos)):
        W, H, N = inf["W"], inf["H"], inf["N"]; RW, RH = CANVAS.get((W, H), (W, H))
        fmt = dict(W=W, H=H, w4=W // 4 // 2 * 2, h4=H // 4 // 2 * 2, w8=W // 8 // 2 * 2, h8=H // 8 // 2 * 2, half=N // 2, half1=N // 2 + 1, rest_half=N - N // 2,
                   rest1=N - 1, RW=RW, RH=RH)
        ctrl = str(vd / f"v{i:03d}_control.mp4"); jobs.append((v, ctrl, "null", inf, 14, None)); items.append((f"v{i:03d}_control", ctrl, prompts[i], i, "control"))
        for name, (fam, _, vf, _) in VARIANTS.items():
            key = f"v{i:03d}_{name}"
            if name == "retest": items.append((key, ctrl, prompts[i], i, name)); continue
            if name == "wrong_prompt":
                same = by_len[N]; j = same[(same.index(i) + 1) % len(same)]
                if j != i: items.append((key, ctrl, prompts[j], i, name))
                continue
            if name == "trimmed" and N <= 124: continue
            if name == "resized" and (RW, RH) == (W, H): continue
            dst = str(vd / f"{key}.mp4"); crf = {"crf38": 38, "crf48": 48}.get(name, 14)
            jobs.append((v, dst, vf.format(**fmt), inf, crf, 124 if name == "trimmed" else None)); items.append((key, dst, prompts[i], i, name))
    print(f"{len(vids)} rollouts ({ {n: len(ix) for n, ix in by_len.items()} } by frame count), {len(jobs)} encodes, {len(items)} videos to score "
          f"with {workers} on GPUs {a.gpus} | {out}", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex: res = list(ex.map(lambda j: encode(*j), jobs))
    failed = {j[1] for j, e in zip(jobs, res) if e}                  # wrong frame count or no file: never scored
    if failed: print(f"{len(failed)} encodes failed:\n  " + "\n  ".join([e for e in res if e][:20]), flush=True)
    items = [x for x in items if x[1] not in failed and os.path.exists(x[1])]
    have_ctrl = {x[3] for x in items if x[4] == "control"}; items = [x for x in items if x[3] in have_ctrl]   # no control, no comparison
    print(f"encoded; scoring {len(items)} videos", flush=True)

    qd = out / "queue"; ws = Workers(workers, qd, a.gpus, a.replicas)
    try:
        ws.wait_loaded()
        for key, mp4, pr, _, _ in items: Q.submit(str(qd), key, mp4, pr, video_workers, meta)
        gkeys = []   # pairwise judges: one group (control, variant) per pair
        for key, mp4, pr, i, var in items:
            if var not in THINK_VARIANTS or i >= a.think_n or not group_workers: continue
            ctrl = next(x for x in items if x[3] == i and x[4] == "control"); gk = f"g_{key}"; gkeys.append((gk, ctrl[0], key, i, var))
            for w in group_workers: Q.submit_group(str(qd), gk, [{"key": ctrl[0], "mp4": ctrl[1]}, {"key": key, "mp4": mp4}], ctrl[2], w, meta)
        got = Q.collect(str(qd), [x[0] for x in items], video_workers, a.timeout) if video_workers else {}
        ggot = Q.collect(str(qd), [g[0] for g in gkeys], group_workers, a.timeout) if gkeys else {}
    finally:
        ws.stop()

    scores, errors = {}, {}   # (i, variant) -> {term: value}
    for key, _, _, i, var in items:
        d = {}
        for w, res in got.get(key, {}).items():
            d.update(res.get("scores") or {})
            if res.get("error"): errors.setdefault(w, []).append(f"{key}: {res['error']}")
        scores[(i, var)] = d
    for gk, ck, dk, i, var in gkeys:
        for w, res in ggot.get(gk, {}).items():
            m = res.get("members") or {}
            for t, v in (m.get(ck) or {}).items(): scores[(i, "control")][f"{t}@{var}"] = v
            for t, v in (m.get(dk) or {}).items(): scores[(i, var)][f"{t}@{var}"] = v
            if res.get("error"): errors.setdefault(w, []).append(f"{gk}: {res['error']}")
    json.dump({f"{i}|{v}": d for (i, v), d in scores.items()}, open(out / "scores.json", "w"), indent=1)
    json.dump({"videos": vids, "infos": infos}, open(out / "videos.json", "w"), indent=1)
    print(f"\nworker errors: {({w: len(e) for w, e in errors.items()}) or 'none'}")
    for w, e in errors.items(): print(f"  {w}: {e[:3]}")

    terms = sorted({t for d in scores.values() for t in d})
    stats = {}   # (term, variant) -> (n, win, drop_sd, p, drop_se_sd)
    for t in terms:
        ctrl_vals = np.array([scores[(i, "control")][t] for i in range(len(vids)) if t in scores.get((i, "control"), {})], float)
        sd = float(ctrl_vals.std(ddof=1)) if len(ctrl_vals) > 2 and ctrl_vals.std() > 0 else 1.0
        for var in VARIANTS:
            if "@" in t and not t.endswith("@" + var): continue
            pairs = [(scores[(i, "control")][t], scores[(i, var)][t]) for i in range(len(vids))
                     if t in scores.get((i, "control"), {}) and t in scores.get((i, var), {})]
            if len(pairs) < 3: continue
            c, v = np.array(pairs, float).T; d = c - v
            stats[(t, var)] = (len(d), float(np.mean(np.sign(d) * 0.5 + 0.5)), float(d.mean() / sd), perm_p(d), float(d.std(ddof=1) / np.sqrt(len(d)) / sd))
    cols = [v for v in VARIANTS if any((t, v) in stats for t in terms)]
    print(f"\n{'term':<22}" + "".join(f"{v:>13}" for v in cols))
    print(f"{'':<22}" + "".join(f"{'win  drop/sd':>13}" for _ in cols) + "    (* expected to drop; ! p < 0.01)")
    flags = []
    for t in terms:
        base = t.split("@")[0]; row = f"{t:<22}"
        for var in cols:
            if (t, var) not in stats: row += f"{'-':>13}"; continue
            n, win, drop, p, se = stats[(t, var)]; exp = base in VARIANTS[var][3] or ("@" in t and var != "reversed")
            row += f"{win:5.2f}{drop:+6.2f}{'*' if exp else ' '}{'!' if p < 0.01 else ' '}"
            if var == "retest":
                if abs(drop) > 0.02 or se > 0.02: flags.append(f"NONDETERMINISTIC {t}: retest drop {drop:+.3f} sd (se {se:.3f})")
            elif exp and drop < 0 and p < 0.05: flags.append(f"WRONG WAY  {t} vs {var}: prefers the degraded video (win {win:.2f}, {drop:+.2f} sd, p={p:.3g})")
            elif exp and (win < 0.6 or p >= 0.05): flags.append(f"MISS       {t} vs {var}: win {win:.2f}, {drop:+.2f} sd, p={p:.3g}")
            elif not exp and p < 0.01 and abs(drop) > 0.3: flags.append(f"SENSITIVE  {t} vs {var}: {drop:+.2f} sd (win {win:.2f}, p={p:.3g})")
        print(row)
    for t in terms:   # monotonicity within each graded family
        fams = {}
        for var, (fam, k, _, exp) in VARIANTS.items():
            if k and (t, var) in stats and t.split("@")[0] in exp: fams.setdefault(fam, []).append((k, var))
        for fam, lv in fams.items():
            lv.sort()
            for (k1, v1), (k2, v2) in zip(lv, lv[1:]):
                d1, d2 = stats[(t, v1)][2], stats[(t, v2)][2]; se = np.hypot(stats[(t, v1)][4], stats[(t, v2)][4])
                if d2 < d1 - 2 * se: flags.append(f"NOT MONOTONE {t}: {v2} drops {d2:+.2f} sd < {v1} {d1:+.2f} sd")
    print("\nlength bias (5 s vs 10 s control means, sd units) and trimmed / resized effects:")
    for t in terms:
        if "@" in t: continue
        by = {n: [scores[(i, 'control')][t] for i in ix if t in scores.get((i, 'control'), {})] for n, ix in by_len.items()}
        allv = [x for v in by.values() for x in v]; sd = float(np.std(allv, ddof=1)) if len(allv) > 2 and np.std(allv) > 0 else 1.0
        means = "  ".join(f"{n}f {np.mean(v):+.3f}" for n, v in sorted(by.items()) if v)
        tr = stats.get((t, "trimmed")); rs = stats.get((t, "resized"))
        print(f"  {t:<20} {means}   gap {(np.mean(by.get(243, [np.nan])) - np.mean(by.get(124, [np.nan]))) / sd:+.2f} sd"
              + (f" | trimmed drop {tr[2]:+.2f} sd p={tr[3]:.3g}" if tr else "") + (f" | resized drop {rs[2]:+.2f} sd p={rs[3]:.3g}" if rs else ""))
    print("\nflags:" + ("\n  " + "\n  ".join(flags) if flags else " none"))
    json.dump({f"{t}|{v}": dict(zip(["n", "win", "drop_sd", "p", "se_sd"], s)) for (t, v), s in stats.items()} | {"flags": flags},
              open(out / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
