"""Stochastic (SDE) reformulation of MiniMax-H3's deterministic rectified-flow
Euler sampler, needed to get per-step Gaussian transition log-probabilities for
GRPO. H3's sampler (musubi_tuner/minimax_h3/sampling.py) is a plain deterministic ODE step:

    video_delta = sigma[i] - sigma[i+1]            # positive, sigma decreases 1->0
    video = video + video_delta * prediction.video  # prediction.video = v = x0 - eps (dataward)

GRPO needs stochastic sampling (so different rollouts in a group actually differ,
and so each step has a computable transition probability for the policy-gradient
ratio) -- the standard technique (DDPO for diffusion, Flow-GRPO for flow matching)
converts the ODE into a family of SDEs with the same marginals, parameterized by
a noise_level, and derives that SDE's per-step Gaussian log-prob analytically.

The Flow-GRPO reference implementation (github.com/yifan123/flow_grpo,
flow_grpo/diffusers_patch/sd3_sde_with_logprob.py, cloned and read directly --
not reproduced from memory) implements this for diffusers' FlowMatchEulerDiscreteScheduler,
whose velocity convention is v = eps - x0 (NOISEWARD) and whose Euler step is

    dt = sigma_prev - sigma          # NEGATIVE (sigma_prev < sigma)
    prev_sample = sample + dt * model_output   (deterministic case)

That is the opposite sign convention from H3's own documented v = x0 - eps
(DATAWARD; confirmed via the x0_callback comment in sampling.py: "the model
predicts the dataward velocity v = x0 - eps, so x0_hat = x_t + sigma*v"). Substituting
model_output_theirs = -v_h3 and dt_theirs = -dt_h3 (dt_h3 := sigma - sigma_prev,
H3's own positive `video_delta`) into their formula and simplifying gives the
h3_sde_step below. This substitution is checked, not assumed: h3_sde_step at
noise_level=0 must reproduce H3's real deterministic step bit-for-bit (see
_selftest_matches_deterministic_step at the bottom -- run this on the actual
GPU machine against the real sample_joint_av before trusting this file for
training).
"""
import math

import torch


def std_dev_t_for_sigma(sigma: torch.Tensor, sigma_near_max: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Same std_dev_t(sigma) as flow_grpo's sde_step_with_logprob -- independent of
    velocity sign convention, so ported unchanged: `torch.sqrt(sigma / (1 -
    torch.where(sigma == 1, sigma_max, sigma))) * noise_level`, where their
    `sigma_max = self.sigmas[1].item()` is NOT the true maximum (1.0) but the
    schedule's SECOND-highest sigma -- substituting the true max (1.0) back in
    would leave the denominator at 1-1=0, the exact singularity being avoided.
    `sigma_near_max` here must be that second-highest value (e.g. schedule.video[1]
    for H3, since schedule.video[0] is always exactly 1.0), never schedule.video[0]."""
    denom = torch.where(sigma == 1.0, sigma_near_max, sigma)
    return torch.sqrt(sigma / (1.0 - denom)) * noise_level


def precise_step_std(sigma: torch.Tensor, sigma_next: torch.Tensor, sigma_near_max: torch.Tensor, noise_level: float) -> torch.Tensor:
    """SAGE-GRPO's integrated ("precise manifold-aware") step std. Flow-GRPO's first-order
    std_dev_t*sqrt(dt) = eta*sqrt(dt*sigma/(1-sigma)) over-injects noise at high sigma; integrating the
    variance over the step gives Sigma = eta^2 [ -(sigma-sigma_next) + log((1-sigma_next)/(1-sigma)) ],
    which reduces to the first-order form as dt->0 (log(1+x) ~ x) and is strictly smaller otherwise.
    At sigma == 1 the integral diverges (log(1 - sigma)); patching sigma to the second-highest
    grid value as std_dev_t_for_sigma does is NOT enough here, because on H3's grid that value IS
    sigma_next, so the integrated variance collapses to exactly 0 and the log-prob becomes
    log(0) = NaN (this poisoned every SAGE update at the first step). The first step therefore
    falls back to the first-order Flow-GRPO std, std_dev_t(sigma_near_max) * sqrt(dt), which is
    finite and positive; every later step uses the integrated form."""
    sig = torch.where(sigma == 1.0, sigma_near_max, sigma)
    var = (noise_level ** 2) * (-(sig - sigma_next) + torch.log((1.0 - sigma_next) / (1.0 - sig)))
    precise = torch.sqrt(torch.clamp(var, min=0.0))
    first_order = std_dev_t_for_sigma(sigma, sigma_near_max, noise_level) * torch.sqrt(torch.clamp(sigma - sigma_next, min=0.0))
    return torch.where(sigma == 1.0, first_order, precise)


def h3_sde_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_near_max: torch.Tensor,
    *,
    noise_level: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    precise_std: bool = False,
):
    """One stochastic step from `sample` (at noise level `sigma`) to the next
    latent (at noise level `sigma_next`, sigma_next < sigma), using H3's own
    dataward velocity convention (`velocity` = x0 - eps, exactly what
    transformer(...) returns as prediction.video / prediction.audio in
    sampling.py). Returns (next_sample, log_prob, mean, std_dev_t).

    `sigma_near_max` is the schedule's second-highest sigma (schedule.video[1]),
    used only to patch the sigma==1 singularity in std_dev_t -- see
    std_dev_t_for_sigma's docstring. Passing the true max (schedule.video[0] == 1.0)
    here does NOT patch the singularity (it leaves the denominator at 1-1=0).

    noise_level=0 must exactly reproduce H3's deterministic sample_joint_av
    Euler step -- see the self-test below.

    THE FINAL STEP MUST BE TAKEN WITH noise_level=0. H3's shifted schedule ends
    with a very large drop (8 linspace steps: ... 0.8 -> 0.632 -> 0.0; HyperFlow: 0.469 -> 0.0), so at the last
    step std_dev_t = sqrt(0.632/0.368)*0.7 = 0.92 and the injected noise is
    std 0.73 -- added to the FINAL latent with no remaining step to denoise it.
    Measured with an oracle velocity (sde_oracle_test.py): integrating all steps
    stochastically leaves a relative error vs the true x0 of 0.87 at 8 steps /
    noise_level=0.7, i.e. the output IS noise, which is exactly what the first
    real rollouts decoded to ("abstract mosaic noise" per the reward judge).
    Taking only the last step deterministically gives 0.0000 error at every step
    count and noise level tested. This mirrors the "no noise at the last step"
    rule in DDPM/DDIM samplers.
    """
    sample = sample.float()
    velocity = velocity.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    dt = sigma - sigma_next  # H3's own `video_delta` / `audio_delta`: positive
    std_dev_t = std_dev_t_for_sigma(sigma, sigma_near_max, noise_level)

    # Derived by substituting model_output_theirs=-velocity, dt_theirs=-dt into
    # flow_grpo's `prev_sample_mean = sample*(1+std^2/(2*sigma)*dt) +
    # model_output*(1+std^2*(1-sigma)/(2*sigma))*dt` (sde_type='sde' branch) --
    # see module docstring for the full derivation.
    step_std = std_dev_t * torch.sqrt(dt)  # dt > 0 here (opposite of their dt<0), so sqrt(dt) directly
    if precise_std and noise_level > 0.0:
        step_std = precise_step_std(sigma, sigma_next, sigma_near_max, noise_level)   # SAGE-GRPO (backbone B)
    # mean = x + v dt + (var / 2) * score, score = (-x + (1 - sigma) v) / sigma. With var = std_dev_t^2 dt this is flow_grpo's
    # first-order mean; with the precise variance it is SAGE-GRPO Eq. 7 (score term and noise use the same variance).
    var = step_std**2
    mean = sample * (1.0 - var / (2.0 * sigma)) + velocity * (dt + var * (1.0 - sigma) / (2.0 * sigma))

    if prev_sample is None:
        noise = torch.randn(velocity.shape, generator=generator, device=velocity.device, dtype=velocity.dtype)
        prev_sample = mean + step_std * noise

    if noise_level == 0.0:
        # Deterministic step: step_std is exactly 0, so the Gaussian log-prob below is 0/0 and
        # log(0) = -inf. Callers use this path for the FINAL sampling step (see the last-step
        # note in rollout_group) and must not record it as a policy-gradient step, so return a
        # finite placeholder instead of letting NaNs loose in the graph.
        return prev_sample, torch.zeros(velocity.shape[0], device=velocity.device), mean, std_dev_t

    log_prob = (
        -((prev_sample.detach() - mean) ** 2) / (2.0 * (step_std**2))
        - torch.log(step_std)
        - torch.log(torch.sqrt(2.0 * torch.as_tensor(math.pi, device=step_std.device)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, mean, std_dev_t


@torch.no_grad()
def _selftest_matches_deterministic_step():
    """Run this on the training GPU (needs the real H3 checkpoint/transformer)
    before trusting h3_sde_step for real training. Verifies noise_level=0 gives
    the IDENTICAL next-sample as H3's real sample_joint_av deterministic Euler
    step, on real (not synthetic) shapes, sigmas, and a real transformer forward.

    from musubi_tuner.minimax_h3.sampling import build_shifted_schedule
    schedule = build_shifted_schedule(steps=8)  # match --infer_steps used elsewhere
    sigma_max = schedule.video[0]
    for i in range(len(schedule.video) - 1):
        sigma, sigma_next = schedule.video[i], schedule.video[i + 1]
        v = transformer(...).video  # same real forward call as in sample_joint_av
        deterministic_next = video + (sigma - sigma_next) * v
        sde_next, _, mean, std = h3_sde_step(video, v, sigma, sigma_next, sigma_max, noise_level=0.0)
        assert std.abs().max() < 1e-6, "std_dev_t should be ~0 at noise_level=0"
        assert torch.allclose(sde_next, deterministic_next, atol=1e-4), "SDE step diverges from H3's own ODE step at noise_level=0 -- sign convention bug, do not train with this"
        video = deterministic_next
    """
    raise NotImplementedError("Run this body on the GPU machine with a real H3 transformer loaded -- see docstring.")
