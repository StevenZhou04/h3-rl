"""One table for a reward ablation: for every arm (one RL run each), the held-out change vs base of every judge term,
from the <dir>/scores_base_<arm>.json files that h3rl.eval.reward_check writes (scripts/ablation_eval.sh runs it).

  python -m h3rl.eval.ablation_table --dir runs/heldout_abl [--arms va_ta va_vq ...]

Each cell is the mean paired difference (arm - base) over the held-out prompts, in units of the term's spread across
base videos, so terms are comparable; '*' marks p < 0.05 (sign-flip permutation), '[ ]' the terms the arm was trained
on (from configs/experiments/ablation/<arm>.yaml when present). Read across a row: did training on X move only X, or
also the other judges, and which way? Written to <dir>/ablation_table.txt too."""
import argparse, glob, json, os
import numpy as np
from h3rl.eval.reward_check import TERMS, paired

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def trained_terms(arm):
    p = os.path.join(REPO, "configs", "experiments", "ablation", f"{arm}.yaml")
    if not os.path.exists(p): return set()
    from omegaconf import OmegaConf
    r = OmegaConf.load(p).get("reward")
    if isinstance(r, str): r = OmegaConf.load(os.path.join(REPO, r))
    return set((r.get("video") or {}).keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", nargs="*", help="default: every scores_base_<arm>.json in --dir")
    a = ap.parse_args()
    arms = a.arms or sorted(os.path.basename(p)[len("scores_base_"):-5] for p in glob.glob(f"{a.dir}/scores_base_*.json"))
    rng = np.random.default_rng(0); rows, lines = {}, []
    for arm in arms:
        S = json.load(open(f"{a.dir}/scores_base_{arm}.json")); row = {}
        for t in TERMS:
            pr = [(v["base"][t], v[arm][t]) for v in S.values() if "base" in v and arm in v and t in v["base"] and t in v[arm]]
            if len(pr) < 3: continue
            x, y = np.array(pr).T; sd = x.std() or 1.0; d = y - x
            row[t] = (d.mean() / sd, paired(d, rng), len(d))
        rows[arm] = row
    terms = [t for t in TERMS if any(t in r for r in rows.values())]
    lines.append(f"{'arm (trained on)':<16}" + "".join(f"{t:>15}" for t in terms) + "     n")
    for arm, row in rows.items():
        tr = trained_terms(arm); cells = []
        for t in terms:
            if t not in row: cells.append(f"{'-':>15}"); continue
            e, p, _ = row[t]; c = f"{e:+.2f}{'*' if p < 0.05 else ' '}"
            cells.append(f"{('[' + c + ']') if t in tr else c:>15}")
        lines.append(f"{arm:<16}" + "".join(cells) + f"  {max((v[2] for v in row.values()), default=0):4d}")
    lines.append("cells: mean change vs base in base-SD units; * p<0.05; [ ] = trained term. n_cuts: lower is better.")
    print("\n".join(lines)); open(f"{a.dir}/ablation_table.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
