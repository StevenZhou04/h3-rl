"""Shared prompt walk (CPU, instant): h3rl.data.pool.draw as train.py uses it, (it * n_teams + team) * P.
 (a) one iteration's groups never repeat a prompt while the pool has enough (96 teams, 200 prompts);
 (b) every prompt is used once per pass through the pool, in a new order each pass;
 (c) the same slots give the same prompts (every rank of a team, a resumed run) and depend on seed and clip length."""
import sys, collections
from h3rl.data.pool import draw
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

cand = [{"pid": f"p{i:03d}"} for i in range(200)]
def iteration(it, n_teams, P=1, seed=0, frames=124):
    return [[r["pid"] for r in draw(cand, seed, frames, (it * n_teams + t) * P, P)] for t in range(n_teams)]

rep = []
for it in range(60):
    ids = [p for t in iteration(it, 96) for p in t]
    straddles = (it * 96) // 200 != (it * 96 + 95) // 200
    if not straddles: rep.append(len(ids) - len(set(ids)))
check("(a) 96 teams x 1 prompt: no repeats within an iteration (iterations inside one pass)", rep and max(rep) == 0, f"{len(rep)} iterations checked")
ids = [p for t in iteration(3, 48, P=2) for p in t]
check("(a) 48 teams x 2 prompts", len(set(ids)) == len(ids) or (3 * 96) // 200 != (3 * 96 + 95) // 200)

walk = [r["pid"] for r in draw(cand, 0, 124, 0, 600)]
passes = [walk[i:i + 200] for i in range(0, 600, 200)]
check("(b) each pass uses every prompt exactly once", all(sorted(p) == sorted(x["pid"] for x in cand) for p in passes))
check("(b) a new order each pass", passes[0] != passes[1] != passes[2])
c = collections.Counter(walk); check("(b) usage is even after 3 passes", set(c.values()) == {3})

check("(c) deterministic: same slots, same prompts (team ranks, resume)", iteration(17, 96) == iteration(17, 96))
check("(c) a different seed or clip length gives a different walk",
      iteration(0, 96, seed=1) != iteration(0, 96) and iteration(0, 96, frames=243) != iteration(0, 96))
small = cand[:5]
check("(c) a pool smaller than an iteration wraps around evenly",
      collections.Counter(r["pid"] for r in draw(small, 0, 124, 0, 10)) == {f"p{i:03d}": 2 for i in range(5)})
if FAILS: sys.exit(1)
print("ALL PASS")
