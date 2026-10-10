"""Held-out reward check: score eval videos of two arms (e.g. the base model and an RL checkpoint) with the reward
workers and compare them PAIRED by prompt (same prompt, same noise). Prints the mean of every term per arm, the mean
difference, how often arm B wins, and a two-sided sign-flip permutation p-value. HPSv3++ is scored at iter_step 0
(rl_progress None), so its RL-progress ramp does not inflate checkpoint scores.

  python -m h3rl.eval.reward_check --dir runs/heldout --arms base nft-00060 --gpus 0 1 2 3

Videos are read from <dir>/<arm>_*/*.mp4 (scripts/heldout_eval.sh writes them there); prompts from --eval_set.
Writes <dir>/scores_<A>_<B>.json (every term of every video)."""
import argparse, glob, json, os, sys
import numpy as np
from h3rl.rewards import queue as Q
from h3rl.rewards.procs import Workers

WORKERS = ["videoalign", "hpspp", "solireward", "flowmotion", "cutcheck"]
TRAINED = ("va_ta", "soli_ta", "hps", "flow_motion")             # the mix_v1 reward terms; the rest are quality checks
TERMS = ["va_ta", "soli_ta", "hps", "flow_motion", "va_vq", "va_mq", "hps_min", "soli_phys", "flow_raw", "flow_coherence", "n_cuts"]


def paired(d: np.ndarray, rng, n: int = 20000) -> float:
    """Two-sided sign-flip permutation p-value of mean(d) = 0."""
    if not d.any(): return 1.0
    return float((np.abs((rng.choice([-1, 1], (n, len(d))) * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval directory holding <arm>_*/ video folders")
    ap.add_argument("--arms", nargs=2, required=True, metavar=("A", "B"), help="reference arm, compared arm")
    ap.add_argument("--gpus", type=int, nargs="+", required=True, help="GPUs for the reward workers")
    ap.add_argument("--eval_set", default=os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "heldout_rl.jsonl"))
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(); A, B = a.arms
    prompts = {(r := json.loads(l))["eid"]: r["prompt"] for l in open(a.eval_set)}
    items = [(f"{arm}__{eid}", p, arm, eid) for arm in (A, B) for p in sorted(glob.glob(f"{a.dir}/{arm}_*/*.mp4"))
             for eid in [os.path.basename(p).rsplit("_s", 1)[0]]]
    unknown = sorted({eid for *_, eid in items if eid not in prompts})
    if unknown: sys.exit(f"videos without a prompt in {a.eval_set}: {unknown[:5]}")
    print(len(items), "videos", flush=True)
    if not items: sys.exit(f"no videos under {a.dir}/{A}_*/ or {a.dir}/{B}_*/")
    qd = f"{a.dir}/score_queue_{A}_{B}"; ws = Workers(WORKERS, qd, a.gpus)
    try:
        ws.wait_loaded()
        for k, p, arm, eid in items: Q.submit(qd, k, p, prompts[eid], WORKERS, {"rl_progress": None})
        got = Q.collect(qd, [x[0] for x in items], WORKERS, a.timeout)
    finally: ws.stop()
    S, errs = {}, {}
    for k, _, arm, eid in items:
        d = {}
        for w, r in got[k].items():
            d.update(r.get("scores") or {})
            if r.get("error"): errs[w] = errs.get(w, 0) + 1
        S.setdefault(eid, {})[arm] = d
    json.dump(S, open(f"{a.dir}/scores_{A}_{B}.json", "w"), indent=1); print("errors:", errs or "none")
    rng = np.random.default_rng(0)
    print(f"{'term':<16}{A:>12}{B:>12}{'diff':>10}{'B wins':>8}{'p':>8}   (paired over prompts; * = trained reward term)")
    for t in TERMS:
        pr = [(v[A][t], v[B][t]) for v in S.values() if A in v and B in v and t in v[A] and t in v[B]]
        if len(pr) < 3: continue
        x, y = np.array(pr).T; d = y - x
        star = "*" if t in TRAINED else " "
        print(f"{t:<15}{star}{x.mean():12.3f}{y.mean():12.3f}{d.mean():+10.3f}{np.mean(np.sign(d) * .5 + .5):8.2f}{paired(d, rng):8.3f}   ({len(d)})")


if __name__ == "__main__":
    main()
