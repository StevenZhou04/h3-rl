"""Gates for the Stage-2 backbones (CPU, seconds).
 NFT:  (a) beta=1, r=1 -> the loss is exactly the plain flow-matching loss;  (b) beta=0 -> zero gradient;
       (c) r in (0,1) interpolates: d loss / d r = ||v+ - v||^2 - ||v- - v||^2;  (d) group_r maps the group's
       best to 1.0, worst to 0.0 and a constant group to 0.5;  (e) on-grid sigmas stay inside the grid gaps.
 SAGE: (f) precise step std -> first-order std as dt -> 0 and is strictly smaller for finite dt;
       (g) the equalizer scales a timestep with 2x the median grad norm by 0.5 and returns 1.0 before it has data."""
import math, sys, os, random, torch, numpy as np
import h3rl.core.nft as T
from h3rl.core.sde import std_dev_t_for_sigma, precise_step_std
from h3rl.core.grpo import GradNormEqualizer
ok_all = True
def check(name, ok, detail=""):
    global ok_all; ok_all &= bool(ok); print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

torch.manual_seed(0)
vo, vt, v = torch.randn(3, 8), torch.randn(3, 8, requires_grad=True), torch.randn(3, 8)
def branch(b, r):
    vp = (1 - b) * vo + b * vt; vn = (1 + b) * vo - b * vt
    return r * ((vp - v) ** 2).mean() + (1 - r) * ((vn - v) ** 2).mean()
fm = ((vt - v) ** 2).mean()
check("(a) beta=1,r=1 == plain FM", torch.allclose(branch(1.0, 1.0), fm), f"{float(branch(1.0,1.0)):.5f} vs {float(fm):.5f}")
g = torch.autograd.grad(branch(0.0, 0.3), vt)[0]; check("(b) beta=0 -> zero grad", float(g.abs().max()) == 0.0)
b = 0.1; r = torch.tensor(0.4, requires_grad=True); L = branch(b, r); dLdr = torch.autograd.grad(L, r)[0]
vp = (1 - b) * vo + b * vt; vn = (1 + b) * vo - b * vt; expect = ((vp - v) ** 2).mean() - ((vn - v) ** 2).mean()
check("(c) dL/dr = ||v+-v||^2 - ||v--v||^2", torch.allclose(dLdr, expect.detach()), f"{float(dLdr):.4f} vs {float(expect):.4f}")
rr = T.group_r([0.1, 0.9, 0.5, 0.3], 0.05); check("(d) group_r best->1, worst->0", rr[1] == 1.0 and rr[0] == 0.0, str([round(x, 3) for x in rr]))
check("(d') constant group -> 0.5", all(x == 0.5 for x in T.group_r([1.0, 1.0, 1.0], 0.05)))
grid = [1.0, .931506, .839236, .703462, .5, .296538, .160764, .068494, 0.0]; rng = random.Random(1)
draws = [T.draw_sigma(grid, 0.5, rng) for _ in range(500)]
check("(e) on-grid sigmas within (0.02, 0.995) and near a grid point", all(0.02 <= s <= 0.995 for s in draws) and all(min(abs(s - g) for g in grid[:-1]) <= 0.12 for s in draws), f"min {min(draws):.3f} max {max(draws):.3f}")
eta = 0.7; near = torch.tensor(0.93)
for sig, dt in ((0.5, 1e-4), (0.8, 1e-4)):
    s1 = std_dev_t_for_sigma(torch.tensor(sig), near, eta) * math.sqrt(dt); s2 = precise_step_std(torch.tensor(sig), torch.tensor(sig - dt), near, eta)
    check(f"(f) precise -> first order at sigma={sig}, dt=1e-4", abs(float(s1) - float(s2)) / float(s1) < 1e-3, f"{float(s1):.6f} vs {float(s2):.6f}")
sig, nxt = torch.tensor(0.9), torch.tensor(0.7)
s1 = std_dev_t_for_sigma(sig, near, eta) * math.sqrt(0.2); s2 = precise_step_std(sig, nxt, near, eta)
check("(f') precise < first-order for a finite step (0.9->0.7)", float(s2) < float(s1), f"{float(s2):.4f} < {float(s1):.4f}")
# first step of the H3 grid: sigma == 1 and sigma_next == the second-highest sigma (the singularity patch value)
s_first = precise_step_std(torch.tensor(1.0), torch.tensor(0.93151), torch.tensor(0.93151), eta)
s_fo = std_dev_t_for_sigma(torch.tensor(1.0), torch.tensor(0.93151), eta) * math.sqrt(1.0 - 0.93151)
check("(f'') sigma=1 first step: precise std finite, > 0, == first-order fallback", bool(torch.isfinite(s_first)) and float(s_first) > 0 and abs(float(s_first) - float(s_fo)) < 1e-6, f"{float(s_first):.4f} vs {float(s_fo):.4f}")
eq = GradNormEqualizer(4); check("(g) equalizer returns 1.0 without data", eq.scale(0) == 1.0)
for t, nrm in ((0, 1.0), (1, 1.0), (2, 2.0), (3, 1.0)): eq.update(t, nrm)
check("(g') 2x-median timestep scaled by ~0.5", abs(eq.scale(2) - 0.5) < 1e-3 and abs(eq.scale(0) - 1.0) < 1e-3, f"{eq.scale(2):.3f}, {eq.scale(0):.3f}")
from h3rl.rewards.combine import WORST     # a gated rollout gets r = 0 and does not compress the rest of its group
plain = T.group_r([1.0, 0.9, 0.5, 0.1, 0.0], 0.05); gated = T.group_r([1.0, 0.9, 0.5, 0.1, 0.0, WORST], 0.05)
check("(d2) gated member -> 0, others unchanged", gated[-1] == 0.0 and all(abs(p - g) < 1e-12 for p, g in zip(plain, gated[:-1])))
check("(d3) gated / missing in a tiny group", T.group_r([0.3, WORST, None], 0.05) == [0.5, 0.0, None])
print("ALL PASS" if ok_all else "FAILURES"); sys.exit(0 if ok_all else 1)
