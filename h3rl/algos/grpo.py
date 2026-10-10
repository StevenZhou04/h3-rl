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
import torch.distributed as dist
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
        self.ref_vel = None

    def _canvas(self):
        c = self.T.canvas; self.cfg.frame_count, self.cfg.height, self.cfg.width = c["frames"], c["height"], c["width"]

    def rollout(self, ctx, prompt, seed, members=None):
        if members is not None and list(members) != list(range(self.cfg.group_size)):
            raise NotImplementedError("GRPO samples whole groups on one rank (ranks_per_group must be 1)")
        self._canvas(); T = self.T
        trust = {"pos": self.ref_pos, "vel": self.ref_vel} if (self.ref_pos is not None or self.ref_vel is not None) else None
        torch.manual_seed(seed)
        trajs = rollout_group(T.transformer, T.network, prompt=prompt["prompt"], ctx=ctx, cfg=self.cfg, device=T.device, schedule=T.schedule, trust_refs=trust)
        return [dict(tr, video=tr["final_video"], audio=tr["final_audio"]) for tr in trajs]

    def update(self, samples, iteration):
        T, cfg = self.T, self.cfg
        # missing reward (timeout, failed worker): advantage 0; gated or broken: below the group's worst; the rest are normalised
        # within their group, and only they enter the mean and std
        rewards = np.array([np.nan if s["R"].get("total") is None else s["R"]["total"] for s in samples]); adv = np.zeros_like(rewards)
        bad = np.array([s["R"]["video"] == WORST for s in samples]); ok = ~np.isnan(rewards) & ~bad
        for g in sorted({s["group"] for s in samples}):
            idx = np.array([i for i, s in enumerate(samples) if s["group"] == g and ok[i]], dtype=int)
            if len(idx) >= 2: adv[idx] = rewards[idx] - rewards[idx].mean()
        valid = [float(x) for x in rewards[ok]]
        if T.world > 1:                                           # one std over every rank's rollouts (global_std)
            parts = [None] * T.world; dist.all_gather_object(parts, valid); valid = [x for p in parts for x in p]
        sd = float(np.std(valid)) if len(valid) >= 2 else 0.0
        adv = adv / max(sd, 1e-3)
        for g in sorted({s["group"] for s in samples}):            # gated / broken: strictly the group's worst
            gi = [i for i, s in enumerate(samples) if s["group"] == g]; floor = min([-1.0] + [adv[i] for i in gi if ok[i]])
            for i in gi:
                if bad[i]: adv[i] = floor - 0.5
        T.transformer.train(); self.opt.zero_grad(set_to_none=True); losses, ratios = [], []
        prev = [p.detach().clone() for p in T.params]
        for i, s in enumerate(samples):
            l, rl = grpo_loss_for_trajectory(T.transformer, T.network, s, float(adv[i]), cfg, s["ctx"], loss_scale=1.0 / len(samples), equalizer=self.eq)
            losses.append(l); ratios.extend(rl)
        if T.world > 1: average_gradients(T.params, T.world)
        gn = float(torch.nn.utils.clip_grad_norm_(T.params, self.max_grad_norm)); self.opt.step(); self.opt.zero_grad(set_to_none=True)
        if cfg.tr_vel_beta > 0: self.ref_vel = prev
        if self.ref_pos is not None and (iteration + 1) % cfg.ref_refresh_every == 0: self.ref_pos = [p.detach().clone() for p in T.params]
        return dict(loss=float(np.mean(losses)), grad_norm=gn, reward_std=sd,
                    ratio_dev_p95=float(np.percentile(np.abs(np.array(ratios) - 1.0), 95)) if ratios else 0.0)

    def state_tensors(self):
        cpu = lambda ps: None if ps is None else [p.detach().cpu() for p in ps]
        return {"opt": self.opt.state_dict(), "ref_pos": cpu(self.ref_pos), "ref_vel": cpu(self.ref_vel), "eq": self.eq.n if self.eq else None}

    def load_tensors(self, s):
        dev = self.T.device
        self.opt.load_state_dict(s["opt"])
        if s.get("ref_pos") is not None: self.ref_pos = [p.to(dev) for p in s["ref_pos"]]
        if s.get("ref_vel") is not None: self.ref_vel = [p.to(dev) for p in s["ref_vel"]]
        if self.eq is not None and s.get("eq") is not None: self.eq.n = list(s["eq"])
