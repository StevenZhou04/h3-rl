"""Reward combiner, registry and algorithm plumbing (CPU, no checkpoints)."""
import sys
from h3rl.rewards.combine import make_combiner, WORST
from h3rl.rewards.registry import workers_for, WORKERS
from h3rl.algos import ALGORITHMS
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

g = {"black_frame_free": 1.0, "flicker_free": 1.0}
c = make_combiner({"video": {"a": 1.0, "b": 0.5}})
outs = [c({"a": x, "b": -x}, g, True)["video"] for x in (0.0, 1.0, 2.0, 3.0, 4.0)]
check("weighted_z: higher a (weight 1) beats higher b (weight .5)", outs[-1] > outs[1], f"{outs}")
check("broken frame guard -> WORST", c({"a": 9.0, "b": 0}, {"black_frame_free": 0.0}, True)["video"] == WORST)
cg = make_combiner({"video": {"a": 1.0}, "gates": {"cut": {"min": 0}}})
for x in range(5): cg({"a": float(x), "cut": 0.0}, g, True)
r = cg({"a": 99.0, "cut": -1.0}, g, True)
check("gate below min -> WORST even with a huge score", r["video"] == WORST and r["gated"] == 1.0)
cf = make_combiner({"video": {"a": 1.0}, "floors": {"q": 1.0}, "floor_warmup": 4})
for x in range(8): cf({"a": 0.0, "q": 1.0 + 0.1 * (x % 2)}, g, True)   # reference q ~ 1.05
hi, lo = cf({"a": 0.0, "q": 1.5}, g, True)["video"], cf({"a": 0.0, "q": 0.5}, g, True)["video"]
check("floor: no reward for rising above the reference, penalty for dropping below", abs(hi) < 1e-9 and lo < 0, f"above {hi:+.3f} below {lo:+.3f}")
st = cf.state(); cf2 = make_combiner({"video": {"a": 1.0}, "floors": {"q": 1.0}, "floor_warmup": 4}); cf2.load(st)
check("combiner state round-trips (resume)", cf2.ref == cf.ref and cf2.seen == cf.seen)
check("registry: terms -> workers", workers_for({"va_ta", "hps", "think_winrate"}) == ["hpspp", "pairwise_think", "videoalign"])
check("registry: pairwise judge is a group worker", WORKERS["pairwise_think"].get("kind") == "group")
check("algorithms registered", {"nft", "grpo"} <= set(ALGORITHMS), f"{sorted(ALGORITHMS)}")
if FAILS: print(f"FAILED: {FAILS}"); sys.exit(1)
print("ALL PASS")
