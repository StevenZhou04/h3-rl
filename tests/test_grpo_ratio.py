#!/usr/bin/env python
"""Gate for the GRPO policy gradient. CPU-only, no checkpoint, seconds to run.

Why this exists
---------------
`grpo_loss_for_trajectory` used to call `h3_sde_step` WITHOUT `prev_sample=`. When that
argument is None, h3_sde_step draws a fresh sample and returns the log-prob OF THAT DRAW, so
the PPO ratio compared a random new sample against the recorded one. The resulting gradient is

    d/dmean [ -(mean.detach() + s*eps - mean)^2 / (2 s^2) ] = eps / s

i.e. advantage-weighted white noise with E[grad] = 0. Three runs (run 1, arm A, arm C)
random-walked the LoRA weights and learned nothing, and nothing in the pipeline could tell:
the loss was finite, the reward was fine, the curve was merely flat.

These assertions fail loudly on that bug. Run before every training launch:

    python grpo_ratio_selftest.py
"""
import math
import os
import sys

import torch

# resolve rl/ from this file, not a fixed root
from h3rl.core.sde import h3_sde_step

SHAPE = (3, 5, 7)          # [batch, ...]; log_prob reduces over all non-batch dims
SIGMA, SIGMA_NEXT, SIGMA_NEAR_MAX = 0.80, 0.63, 0.988
NOISE = 0.7
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def t(x):
    return torch.tensor(float(x), dtype=torch.float32)


def rollout(seed=0):
    """One recorded step: returns (state, velocity, action, logp_old)."""
    g = torch.Generator().manual_seed(seed)
    sample = torch.randn(SHAPE, generator=g)
    velocity = torch.randn(SHAPE, generator=g)
    action, logp, mean, std = h3_sde_step(
        sample, velocity, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX),
        noise_level=NOISE, generator=g,
    )
    return sample, velocity, action.detach(), logp.detach(), mean.detach(), std


print(__doc__.split("Why this exists")[0].strip())
print("\n=== A. identity: replaying the recorded action under an unchanged policy ===")
sample, velocity, action, logp_old, mean_old, std = rollout(0)

_, logp_replay, _, _ = h3_sde_step(
    sample, velocity, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX),
    noise_level=NOISE, prev_sample=action,
)
ratio = torch.exp(logp_replay - logp_old)
check("ratio == 1 exactly when the policy is unchanged",
      torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6),
      f"ratio={ratio.tolist()}")

# the bug, demonstrated: same call without prev_sample
_, logp_fresh, _, _ = h3_sde_step(
    sample, velocity, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX), noise_level=NOISE,
)
ratio_bug = torch.exp(logp_fresh - logp_old)
check("omitting prev_sample does NOT give ratio 1 (the bug is real)",
      not torch.allclose(ratio_bug, torch.ones_like(ratio_bug), atol=1e-3),
      f"ratio={[round(v, 4) for v in ratio_bug.tolist()]}")

print("\n=== B. log-prob matches a textbook Gaussian ===")
ref = torch.distributions.Normal(mean_old, std * math.sqrt(SIGMA - SIGMA_NEXT))
ref_lp = ref.log_prob(action).mean(dim=tuple(range(1, action.ndim)))
check("h3_sde_step log_prob == Normal(mean, step_std).log_prob(action).mean()",
      torch.allclose(ref_lp, logp_old, atol=1e-5),
      f"max|diff|={float((ref_lp - logp_old).abs().max()):.3e}")

print("\n=== C. gradient points toward the recorded action ===")
# A one-parameter policy: theta scales the velocity, so d(mean)/d(theta) > 0.
def logp_of(theta, prev_sample):
    v = velocity * theta
    _, lp, _, _ = h3_sde_step(
        sample, v, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX),
        noise_level=NOISE, prev_sample=prev_sample,
    )
    return lp

theta = torch.tensor(1.0, requires_grad=True)
lp = logp_of(theta, action).sum()
lp.backward()
g_fixed = float(theta.grad)

# finite difference on the same quantity
with torch.no_grad():
    eps = 1e-3
    fd = float((logp_of(torch.tensor(1.0 + eps), action).sum()
                - logp_of(torch.tensor(1.0 - eps), action).sum()) / (2 * eps))
check("analytic gradient matches finite difference",
      abs(g_fixed - fd) <= 1e-2 * max(1.0, abs(fd)),
      f"analytic={g_fixed:+.5f} fd={fd:+.5f}")

print("\n=== D. E[grad] is zero under the bug, non-zero after the fix ===")
grads_bug = []
for k in range(200):
    th = torch.tensor(1.0, requires_grad=True)
    v = velocity * th
    torch.manual_seed(1000 + k)                      # fresh draw each time, as the bug did
    _, lp_b, _, _ = h3_sde_step(sample, v, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX),
                                noise_level=NOISE)   # <-- no prev_sample
    lp_b.sum().backward()
    grads_bug.append(float(th.grad))
gb = torch.tensor(grads_bug)
mean_b, sem_b = float(gb.mean()), float(gb.std() / math.sqrt(len(gb)))
check("buggy gradient has mean indistinguishable from 0",
      abs(mean_b) < 3 * sem_b,
      f"mean={mean_b:+.4f} +/- {sem_b:.4f} (3sigma), sd={float(gb.std()):.4f}")
# Determinism: the fixed path must give the same gradient every time, because it evaluates a
# FIXED recorded action rather than a fresh draw.
theta2 = torch.tensor(1.0, requires_grad=True)
logp_of(theta2, action).sum().backward()
check("fixed gradient is exactly reproducible",
      float(theta2.grad) == g_fixed, f"{float(theta2.grad):+.6f} vs {g_fixed:+.6f}")
check("fixed gradient is non-zero", abs(g_fixed) > 1e-6, f"{g_fixed:+.6f}")
print(f"     [info] per-sample SNR: signal {abs(g_fixed):.4f} vs buggy noise sd "
      f"{float(gb.std()):.4f} -- the bug's noise is {float(gb.std())/abs(g_fixed):.1f}x the "
      f"signal, which is why the random walk dominated.")

print("\n=== E. GRPO step moves each member toward/away per its advantage ===")
# Each member gets its OWN parameter, so there is no conflict between members and a correct
# gradient must move ALL of them the right way. (A single shared scalar cannot satisfy eight
# conflicting objectives -- that would test the optimiser, not the gradient.)
G = 8
members = [rollout(seed=10 + i) for i in range(G)]
advs = [+1.0] * (G // 2) + [-1.0] * (G // 2)


def grpo_step_deltas(use_prev_sample: bool, lr: float = 0.05):
    """One SGD step per member; returns advantage-signed log-prob change for each."""
    deltas = []
    for (s_i, v_i, a_i, lp_i, _, _), adv in zip(members, advs):
        th = torch.tensor(1.0, requires_grad=True)
        kw = dict(prev_sample=a_i) if use_prev_sample else {}
        _, lp_new, _, _ = h3_sde_step(s_i, v_i * th, t(SIGMA), t(SIGMA_NEXT), t(SIGMA_NEAR_MAX),
                                      noise_level=NOISE, **kw)
        (-adv * torch.exp(lp_new - lp_i)).mean().backward()
        th_new = 1.0 - lr * float(th.grad)
        with torch.no_grad():   # always score the REAL action, to judge whether it improved
            _, lp_after, _, _ = h3_sde_step(s_i, v_i * th_new, t(SIGMA), t(SIGMA_NEXT),
                                            t(SIGMA_NEAR_MAX), noise_level=NOISE, prev_sample=a_i)
        deltas.append(float((lp_after - lp_i).mean()) * adv)
    return deltas


d_fixed = grpo_step_deltas(use_prev_sample=True)
n_ok = sum(d > 0 for d in d_fixed)
check("every member moves in its advantage's direction", n_ok == G, f"{n_ok}/{G}")
check("advantage-weighted log-prob change is positive", sum(d_fixed) > 0,
      f"sum={sum(d_fixed):+.3e}")

torch.manual_seed(7)
d_bug = grpo_step_deltas(use_prev_sample=False)
n_bug = sum(d > 0 for d in d_bug)
check("the buggy path does NOT reliably move members correctly (contrast)",
      n_bug < G, f"{n_bug}/{G} correct under the bug, vs {n_ok}/{G} fixed")

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    sys.exit(1)
print("ALL PASS -- the policy gradient carries reward signal.")
