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

# score_batch: one iteration is scored against the same statistics, so equal raw scores get equal rewards
cb = make_combiner({"video": {"a": 1.0}})
out = cb.score_batch([({"a": x}, {}, False) for x in [1.0, 2.0, 1.0, 1.0]])
assert out[0]["video"] == out[2]["video"] == out[3]["video"] < out[1]["video"], [o["video"] for o in out]
print("score_batch ok", [round(o["video"], 3) for o in out])

# total: one scalar for GRPO with the sync terms counted once; a gate without a verdict gives no reward
cb = make_combiner({"video": {"a": 1.0}, "audio": {"b": 1.0}, "sync": {"s": 1.0}})
R = cb.score_batch([({"a": x, "b": y, "s": z}, {}, True) for x, y, z in [(0, 0, 0), (1, 2, 3), (2, 1, 0)]])[1]
cb2 = make_combiner({"video": {"a": 1.0}, "audio": {"b": 1.0}})
R2 = cb2.score_batch([({"a": x, "b": y}, {}, True) for x, y in [(0, 0), (1, 2), (2, 1)]])[1]
assert abs(R2["total"] - (R2["video"] + R2["audio"])) < 1e-12                       # no sync terms: total = video + audio
Rs = R["video"] - R2["video"]; assert abs(R["total"] - (R2["video"] + R2["audio"] + Rs)) < 1e-9, (R, R2)   # sync counted once
g = make_combiner({"video": {"a": 1.0}, "gates": {"cut_free": {"min": 0}}})
out = g.score_batch([({"a": 1.0}, {}, False), ({"a": 2.0, "cut_free": -1}, {}, False), ({"a": 3.0, "cut_free": 1}, {}, False)])
assert out[0]["video"] is None and out[0]["total"] is None and out[0]["gate_unknown"] == 1.0
assert out[1]["video"] == WORST and out[1]["total"] == WORST and out[2]["video"] is not None
ga = make_combiner({"video": {"a": 1.0}, "audio": {"b": 1.0}, "gates": {"cut_free": {"min": 0}}})
out = ga.score_batch([({"a": 1.0, "b": 1.0, "cut_free": 1}, {}, True), ({"a": 2.0, "b": 9.0, "cut_free": -1}, {}, True),
                      ({"a": 3.0, "b": 9.0, "cut_free": 1}, {"black_frame_free": 0.0}, True)])
assert out[1]["audio"] == WORST and out[2]["audio"] == WORST and out[0]["audio"] != WORST, out   # gated / broken: worst on the audio branch too
print("total / gate_unknown ok")
