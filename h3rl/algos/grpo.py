"""Flow-GRPO with optional SAGE-GRPO switches (arXiv 2603.21872).
Rollouts: the ODE sampler turned into an SDE (noise_level), recording each step's Gaussian transition log-prob.
Update: clipped-ratio policy gradient with group-centred advantages (scaled by the batch std), backpropagated through
`grad_steps` of the recorded steps. SAGE switches: sde_precise (exact step variance), grad_equalizer (per-timestep
gradient-norm equalisation), tr_pos_beta / tr_vel_beta (trust regions to a periodic LoRA snapshot and to the previous step)."""
from __future__ import annotations
import numpy as np, torch
from h3rl.algos.base import Algorithm, register
from h3rl.core.grpo import GRPOConfig, rollout_group, grpo_loss_for_trajectory, GradNormEqualizer
from h3rl.core.dist import average_gradients
from h3rl.rewards.combine import WORST


@register("grpo")
class GRPO(Algorithm):
    def __init__(self, acfg, T):
        super().__init__(acfg, T)
        self.cfg = GRPOConfig(group_size=self.group_size(), prompts_per_step=self.prompts_per_step(), infer_steps=T.infer_steps,
                              noise_level=float(acfg.get("noise_level", 0.7)), clip_range=float(acfg.get("clip_range", 0.2)),
                              kl_beta=float(acfg.get("kl_beta", 0.0)), lr=float(acfg["lr"]), grad_steps_per_traj=int(acfg.get("grad_steps", 0)),
                              shared_init_noise=True, global_std=True, sde_precise=bool(acfg.get("sde_precise", False)),
                              grad_equalizer=bool(acfg.get("grad_equalizer", False)), tr_pos_beta=float(acfg.get("tr_pos_beta", 0.0)),
                              tr_vel_beta=float(acfg.get("tr_vel_beta", 0.0)), ref_refresh_every=int(acfg.get("ref_refresh_every", 20)))
        self.max_grad_norm = float(acfg.get("max_grad_norm", 1.0))
        self.opt = torch.optim.AdamW(T.params, lr=self.cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.eq = GradNormEqualizer(self.cfg.infer_steps - 1) if self.cfg.grad_equalizer else None
        self.ref_pos = [p.detach().clone() for p in T.params] if self.cfg.tr_pos_beta > 0 else None
        self.ref_vel = None; self.iteration = None

    def _canvas(self):
        c = self.T.canvas; self.cfg.frame_count, self.cfg.height, self.cfg.width = c["frames"], c["height"], c["width"]

    def rollout(self, ctx, prompt, seed):
        self._canvas(); T = self.T
        trust = {"pos": self.ref_pos, "vel": self.ref_vel} if (self.ref_pos is not None or self.ref_vel is not None) else None
        torch.manual_seed(seed)
        trajs = rollout_group(T.transformer, T.network, prompt=prompt["prompt"], ctx=ctx, cfg=self.cfg, device=T.device, schedule=T.schedule, trust_refs=trust)
        return [dict(tr, video=tr["final_video"], audio=tr["final_audio"]) for tr in trajs]

    def update(self, samples, iteration):
        T, cfg = self.T, self.cfg
        # missing reward (timeout, failed worker): advantage 0; gated or broken: fixed -1; the rest are normalised
        # within their group, and only they enter the mean and std
        rewards = np.array([np.nan if s["R"]["video"] is None and s["R"]["audio"] is None else (s["R"]["video"] or 0.0) + (s["R"]["audio"] or 0.0)
                            for s in samples]); adv = np.zeros_like(rewards)
        bad = np.array([s["R"]["video"] == WORST for s in samples]); ok = ~np.isnan(rewards) & ~bad
        for g in sorted({s["group"] for s in samples}):
            idx = np.array([i for i, s in enumerate(samples) if s["group"] == g and ok[i]], dtype=int)
            if len(idx) >= 2: adv[idx] = rewards[idx] - rewards[idx].mean()
        adv = adv / max(float(rewards[ok].std()) if ok.sum() >= 2 else 0.0, 1e-3); adv[bad] = -1.0
        T.transformer.train(); self.opt.zero_grad(set_to_none=True); losses, ratios = [], []
        prev = [p.detach().clone() for p in T.params]
        for i, s in enumerate(samples):
            l, rl = grpo_loss_for_trajectory(T.transformer, T.network, s, float(adv[i]), cfg, s["ctx"], loss_scale=1.0 / len(samples), equalizer=self.eq)
            losses.append(l); ratios.extend(rl)
        if T.world > 1: average_gradients(T.params, T.world)
        gn = float(torch.nn.utils.clip_grad_norm_(T.params, self.max_grad_norm)); self.opt.step(); self.opt.zero_grad(set_to_none=True)
        if cfg.tr_vel_beta > 0: self.ref_vel = prev
        if self.ref_pos is not None and (iteration + 1) % cfg.ref_refresh_every == 0: self.ref_pos = [p.detach().clone() for p in T.params]
        return dict(loss=float(np.mean(losses)), grad_norm=gn, reward_std=float(rewards[ok].std()) if ok.any() else 0.0,
                    ratio_dev_p95=float(np.percentile(np.abs(np.array(ratios) - 1.0), 95)) if ratios else 0.0)
