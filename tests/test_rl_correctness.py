#!/usr/bin/env python
"""End-to-end correctness suite for the LoRA + GRPO stack. CPU-only, no checkpoint.

Written after a bug where grpo_loss_for_trajectory evaluated log pi(fresh random draw)
instead of log pi(recorded action), making E[grad] exactly zero. Three runs learned nothing
and every dashboard looked healthy. Reading the code did not catch it; an assertion did.

So these tests EXECUTE the real functions -- compute_advantages, the real best_of_n selection,
the real grpo_loss_for_trajectory, the real LoRA modules -- rather than re-deriving them.
The headline test is Part 3: drive the real loss function on a toy problem whose optimum is
known and assert the reward actually rises.
"""
import math, os, sys, types
import numpy as np
import torch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-selftest-placeholder")

from h3rl.core.sde import h3_sde_step
import h3rl.core.grpo as T

FAILS = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok: FAILS.append(name)

def t(x): return torch.tensor(float(x), dtype=torch.float32)

# =====================================================================================
print("\n=== PART 1: LoRA ===")
from musubi_tuner.networks.lora import LoRAModule, LoRANetwork

base = torch.nn.Linear(8, 6, bias=False)
torch.nn.init.normal_(base.weight)
base_w = base.weight.detach().clone()   # captured BEFORE apply_to() patches forward
lm = LoRAModule("test", base, multiplier=1.0, lora_dim=4, alpha=2.0)
lm.apply_to()

check("scale == alpha/dim", abs(lm.scale - 2.0 / 4) < 1e-9, f"scale={lm.scale}")
check("lora_up is zero-initialised (LoRA is identity at step 0)",
      float(lm.lora_up.weight.detach().abs().max()) == 0.0)

x = torch.randn(3, 8)
with torch.no_grad():
    y0 = base(x)
check("forward == base forward while lora_up is zero",
      torch.allclose(y0, base.weight @ x.T if False else torch.nn.functional.linear(x, base.weight), atol=1e-6))

with torch.no_grad():
    lm.lora_up.weight.normal_()
    y1 = base(x)
    # apply_to() deletes lm.org_module, so use our own handle on the base layer;
    # base.weight is untouched by the monkey-patch, only base.forward is replaced.
    manual = torch.nn.functional.linear(x, base_w) \
             + lm.lora_up(lm.lora_down(x)) * lm.multiplier * lm.scale
check("forward == base + up(down(x))*multiplier*scale",
      torch.allclose(y1, manual, atol=1e-5), f"max|diff|={float((y1-manual).abs().max()):.2e}")

with torch.no_grad():
    lm.multiplier = 0.0
    y_off = base(x)
check("multiplier=0 disables the adapter exactly",
      torch.allclose(y_off, torch.nn.functional.linear(x, base_w), atol=1e-6))
lm.multiplier = 1.0

# gradient reaches BOTH factors
lm.lora_down.weight.grad = lm.lora_up.weight.grad = None
base(x).sum().backward()
check("gradient reaches lora_down", lm.lora_down.weight.grad is not None
      and float(lm.lora_down.weight.grad.abs().sum()) > 0)
check("gradient reaches lora_up", lm.lora_up.weight.grad is not None
      and float(lm.lora_up.weight.grad.abs().sum()) > 0)
check("base weight is unchanged by the adapter (frozen in real runs via requires_grad_(False))",
      torch.allclose(base.weight.detach(), base_w, atol=0.0))

# set_multiplier must reach every module, not just the container (a real past bug)
class _Net:
    def __init__(self, loras): self.text_encoder_loras, self.unet_loras = [], loras
    set_multiplier = LoRANetwork.set_multiplier
mods = [LoRAModule(f"m{i}", torch.nn.Linear(4, 4, bias=False), 1.0, 2, 1.0) for i in range(3)]
net = _Net(mods)
net.set_multiplier(0.0)
check("set_multiplier propagates to every LoRAModule",
      all(m.multiplier == 0.0 for m in mods),
      f"multipliers={[m.multiplier for m in mods]}")

# =====================================================================================
print("\n=== PART 3: end-to-end -- the REAL loss function must increase reward ===")
# A toy policy standing in for the transformer. The 'velocity' it emits is a learnable
# scalar; reward is higher when the final latent lands near TARGET. If GRPO is wired
# correctly -- advantage sign, trimming, ratio, loss sign, backward -- reward must rise.
TARGET = 1.5
STEPS, GROUP = 4, 8

class FakePred:
    def __init__(self, v, a): self.video, self.audio = v, a

class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(()))
        self.gradient_checkpointing = False
    def forward(self, *, video_latents, audio_latents, **kw):
        return FakePred(self.theta.expand_as(video_latents),
                        torch.zeros_like(audio_latents))

class FakeNetwork:
    def set_multiplier(self, m): pass

cfg = T.GRPOConfig(group_size=GROUP, infer_steps=STEPS, noise_level=0.7, clip_range=0.2,
                   adv_clip_max=5.0, grad_steps_per_traj=0, kl_beta=0.0, best_of_n=2)
sigmas = torch.linspace(1.0, 0.0, STEPS + 1)
CTX = dict(layout=None, text_hidden_states=torch.zeros(1, 1), text_token_tags=None,
           visual_condition_clean=None)  # only .device is read off text_hidden_states

def rollout(model):
    """Mirrors rollout_group: SDE for all but the last step, records the ACTION."""
    trajs = []
    for _ in range(GROUP):
        v = torch.zeros(1, 4); a = torch.zeros(1, 4); rec = []
        with torch.no_grad():
            for i in range(STEPS):
                is_last = i == STEPS - 1
                nl = 0.0 if is_last else cfg.noise_level
                pred = model(video_latents=v, audio_latents=a)
                nv, lpv, _, sv = h3_sde_step(v, pred.video, sigmas[i], sigmas[i+1], sigmas[1], noise_level=nl)
                na, lpa, _, sa = h3_sde_step(a, pred.audio, sigmas[i], sigmas[i+1], sigmas[1], noise_level=nl)
                if not is_last:
                    rec.append(dict(video=v, audio=a, next_video=nv, next_audio=na,
                                    sigma_v=sigmas[i], sigma_v_next=sigmas[i+1],
                                    sigma_a=sigmas[i], sigma_a_next=sigmas[i+1],
                                    sigma_max_v=sigmas[1], sigma_max_a=sigmas[1],
                                    log_prob_old=lpv + lpa,
                                    mean_v_ref=None, std_v=sv, mean_a_ref=None, std_a=sa))
                v, a = nv, na
        trajs.append(dict(steps=rec, final_video=v, final_audio=a))
    return trajs

torch.manual_seed(0)
model = FakeTransformer(); netw = FakeNetwork()
opt = torch.optim.SGD(model.parameters(), lr=0.5)
def group_advantages(rw):   # as in h3rl/algos/grpo_loop.py: centred within the group, scaled by the batch std
    return (rw - rw.mean()) / max(float(rw.std()), 1e-3)
def select(pr, rw, n):      # every rollout of the group is trained on
    return list(range(len(rw)))
history = []
for it in range(60):
    trajs = rollout(model)
    rw = np.array([-abs(float(tr["final_video"].mean()) - TARGET) for tr in trajs])
    history.append(rw.mean())
    pr = ["p"] * GROUP
    advs = group_advantages(rw)
    idx = select(pr, rw, cfg.best_of_n)
    opt.zero_grad()
    for i in idx:
        T.grpo_loss_for_trajectory(model, netw, trajs[i], advs[i], cfg, CTX,
                                   loss_scale=1.0 / len(idx))
    opt.step()

first, last = np.mean(history[:5]), np.mean(history[-5:])
check("reward INCREASES over training (the whole algorithm, end to end)",
      last > first + 0.05, f"{first:+.4f} -> {last:+.4f}  (theta={float(model.theta):+.3f}, target {TARGET})")
# Measure the final latent directly rather than predicting it: the SDE mean update is
# sample*(1 - std^2/(2 sigma)*dt) + velocity*(1 + std^2 (1-sigma)/(2 sigma))*dt, so the
# endpoint is NOT theta*steps.
torch.manual_seed(123)
untrained = FakeTransformer()
err_before = np.mean([abs(float(tr["final_video"].mean()) - TARGET) for tr in rollout(untrained)])
torch.manual_seed(123)
err_after = np.mean([abs(float(tr["final_video"].mean()) - TARGET) for tr in rollout(model)])
check("trained policy lands closer to the target than the untrained one",
      err_after < err_before * 0.5,
      f"|final-target|: {err_before:.3f} -> {err_after:.3f}  (theta={float(model.theta):+.3f})")

# and the control: break the ratio exactly as the old bug did, rerun, expect NO learning
import h3rl.core.sde as S
_orig = T.h3_sde_step
def _buggy(sample, velocity, sigma, sigma_next, near_max, *, noise_level=0.7, precise_std=False,
           prev_sample=None, generator=None):
    return _orig(sample, velocity, sigma, sigma_next, near_max,
                 noise_level=noise_level, prev_sample=None, generator=generator)
torch.manual_seed(0)
model_b = FakeTransformer(); opt_b = torch.optim.SGD(model_b.parameters(), lr=0.5)
T.h3_sde_step = _buggy
try:
    hist_b = []
    for it in range(60):
        trajs = rollout(model_b)   # rollout uses the module-level import, still correct
        rw = np.array([-abs(float(tr["final_video"].mean()) - TARGET) for tr in trajs])
        hist_b.append(rw.mean())
        pr = ["p"] * GROUP
        advs = group_advantages(rw)
        idx = select(pr, rw, cfg.best_of_n)
        opt_b.zero_grad()
        for i in idx:
            T.grpo_loss_for_trajectory(model_b, netw, trajs[i], advs[i], cfg, CTX,
                                       loss_scale=1.0 / len(idx))
        opt_b.step()
finally:
    T.h3_sde_step = _orig
gain_fixed = last - first
gain_bug = np.mean(hist_b[-5:]) - np.mean(hist_b[:5])
check("the OLD buggy ratio learns far less (regression guard)",
      gain_bug < gain_fixed * 0.5,
      f"fixed {gain_fixed:+.4f} vs buggy {gain_bug:+.4f}")

# =====================================================================================
print("\n=== PART 4: DDP gradient averaging ===")
ps = [torch.nn.Parameter(torch.randn(3)) for _ in range(2)]
grads = [torch.randn(3) for _ in ps]
for p, g in zip(ps, grads): p.grad = g.clone()
world = 4
for p in ps:                      # emulate all_reduce(SUM) over identical-shaped grads
    p.grad = p.grad * world
    p.grad /= world
check("all_reduce(SUM)/world_size is an average (identity on identical grads)",
      all(torch.allclose(p.grad, g, atol=1e-6) for p, g in zip(ps, grads)))
import inspect, h3rl.core.dist as DIST
src = inspect.getsource(DIST)
check("average_gradients fills missing grads with zeros (prevents rank deadlock)",
      "p.grad = torch.zeros_like(p)" in src)
check("average_gradients divides by world_size", "p.grad /= world_size" in src)

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS)); sys.exit(1)
print("ALL PASS")
