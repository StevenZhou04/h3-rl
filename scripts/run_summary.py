"""Training trend of a run from its metrics.jsonl: the mean of the learning-signal metrics and of every reward axis over
the first, middle and last ~10% of the iterations.

  python scripts/run_summary.py runs/nft_hf_mix/metrics.jsonl

axis/hps rises during training partly by design (HPSv3++ ramps with RL progress); judge it on the held-out check."""
import json, sys

m = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not m: sys.exit("empty metrics file")
n = len(m); w = max(1, n // 10); mid = n // 2 - w // 2
avg = lambda s, k: sum(x.get(k, 0) for x in s) / len(s)
keys = ["grad_norm", "loss_kl", "R_video_std", "gated", "worst_frac"] + sorted(k for k in m[-1] if k.startswith("axis/"))
print(f"iterations={n}  window={w}")
print(f"{'metric':<24}{'first':>10}{'mid':>10}{'last':>10}")
for k in keys: print(f"{k:<24}" + "".join(f"{avg(s, k):10.3g}" for s in (m[:w], m[mid:mid + w], m[-w:])))
