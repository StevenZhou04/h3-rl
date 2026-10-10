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
from h3rl.core.video_metrics import is_broken
check_ok = (not is_broken({"black_frame_free": 1.0, "flicker_free": 1.0, "frame_count": 124})
            and is_broken({"black_frame_free": 1 - 1 / 124, "flicker_free": 1.0, "frame_count": 124})          # one black frame
            and not is_broken({"black_frame_free": 1.0, "flicker_free": 1 - 3 / 123, "frame_count": 124})      # a few luma jumps (cuts)
            and is_broken({"black_frame_free": 1.0, "flicker_free": 1 - 10 / 123, "frame_count": 124}))        # strobing
assert check_ok, "is_broken thresholds"
# statistics per clip length: a term on a different scale for 10 s clips does not lose weight to the length gap
cl = make_combiner({"video": {"a": 1.0, "b": 1.0}})
for k in range(40):
    cl.score_batch([({"a": float(k % 4), "b": float(k % 3)}, {"frame_count": 124}, False),
                    ({"a": float(k % 4) - 20.0, "b": float(k % 3)}, {"frame_count": 243}, False)])
assert abs(cl.rz.sd("a@124") - cl.rz.sd("a@243")) < 1e-9 and cl.rz.sd("a@243") < 2.0, (cl.rz.sd("a@124"), cl.rz.sd("a@243"))
# running window: after a level shift the mean follows (cumulative stats would stay near the old level)
from h3rl.rewards.combine import RunningZ
rz = RunningZ(64)
for k in range(1000): rz.update("x", float(k % 2))
for k in range(400): rz.update("x", 10.0 + k % 2)
assert abs(rz.mean["x"] - 10.5) < 0.1, rz.mean["x"]
rw = RunningZ(None); xs = [0.3, 1.7, 2.2, 5.0, 4.1]
for x in xs: rw.update("y", x)
import statistics; assert abs(rw.sd("y") - statistics.stdev(xs)) < 1e-12 and abs(rw.mean["y"] - statistics.mean(xs)) < 1e-12   # Welford before the window
old = {"n": {"y": 5}, "mean": {"y": statistics.mean(xs)}, "m2": {"y": statistics.variance(xs) * 4}}; r2 = RunningZ(None); r2.load(old)
assert abs(r2.sd("y") - statistics.stdev(xs)) < 1e-12   # old checkpoints (m2) still load
assert is_broken({"black_frame_free": 0.9}) and not is_broken({"black_frame_free": 1.0})   # no frame count: any black fraction
# ranks observe the whole iteration: two ranks with different local batches end with identical statistics
def rk(): return make_combiner({"video": {"a": 1.0}})
A = [({"a": float(x)}, {"frame_count": 124}, False) for x in (0, 1, 2, 3)]; B = [({"a": float(x)}, {"frame_count": 124}, False) for x in (10, 11, 12, 13)]
r0, r1 = rk(), rk(); o0 = r0.score_batch(A, observe=A + B); o1 = r1.score_batch(B, observe=A + B)
assert r0.state() == r1.state() and o0[0]["video"] < o1[0]["video"], (o0, o1)
# legacy floor state (unsuffixed keys) keeps its reference after resume instead of re-warming on the trained policy
lf = make_combiner({"video": {"a": 1.0}, "floors": {"q": 1.0}, "floor_warmup": 4}); lf.load({"rz": {"n": {}, "mean": {}, "m2": {}}, "ref": {"q": 2.0}, "seen": {"q": 4}})
lf.score_batch([({"a": 0.0, "q": 5.0}, {"frame_count": 124}, False)]); assert lf.ref["q@124"] == 2.0 and lf.seen["q@124"] == 4, (lf.ref, lf.seen)
# text-cache keys change with anything the encoding depends on
import os, tempfile
from h3rl.data.pool import text_key
r = {"pid": "p1", "task": "t2va", "prompt": "a cat"}
assert text_key(r, 544, 960) == text_key(r, 480, 832) != text_key(dict(r, prompt="a dog"), 544, 960)   # t2va: canvas-free, text-sensitive
img = os.path.join(tempfile.mkdtemp(), "x.png"); open(img, "wb").write(b"img-bytes-1")
f = {"pid": "p2", "task": "fl2va", "prompt": "a cat", "image": img}
k1, k2 = text_key(f, 544, 960), text_key(f, 480, 832); assert k1 != k2 != text_key(dict(f, task="t2va"), 544, 960)
print("guardrails / per-length / window ok")
print("total / gate_unknown ok")
