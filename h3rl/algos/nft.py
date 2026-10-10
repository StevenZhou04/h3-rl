"""DiffusionNFT (Zheng et al., 2025): forward-process, likelihood-free RL on the plain ODE sampler.
Rollouts: deterministic ODE samples with independent initial noise. Update: the reward-weighted flow-matching loss
    v+ = (1-b) v_old + b v_theta,  v- = (1+b) v_old - b v_theta,  L = r ||v+ - v||^2 + (1-r) ||v- - v||^2
with r in [0,1] the group-normalised reward (per modality: video and audio branches), v_old an EMA copy of the LoRA.
As in the reference implementation, rollouts are sampled with v_old and the EMA is updated once per iteration, after
the gradient steps: old <- eta old + (1 - eta) theta, eta = min(ema_slope * optimizer steps so far, ema_max)."""
from __future__ import annotations
import numpy as np, torch
from h3rl.algos.base import Algorithm, register
from h3rl.core.nft import NFTConfig, iter_samples, group_r, nft_loss, ema_update, swapped_params
from h3rl.core.dist import average_gradients


@register("nft")
class NFT(Algorithm):
    def __init__(self, acfg, T):
        super().__init__(acfg, T)
        self.cfg = NFTConfig(group_size=self.group_size(), prompts_per_step=self.prompts_per_step(), infer_steps=T.infer_steps,
                             beta=float(acfg.get("beta", 0.1)), lr=float(acfg["lr"]), epochs_per_batch=int(acfg.get("epochs_per_batch", 2)),
                             grad_accum=int(acfg.get("grad_accum", 4)), max_grad_norm=float(acfg.get("max_grad_norm", 1.0)),
                             ema_slope=float(acfg.get("ema_slope", 0.001)), ema_max=float(acfg.get("ema_max", 0.5)),
                             sigma_jitter=float(acfg.get("sigma_jitter", 0.5)), audio_loss_weight=float(acfg.get("audio_loss_weight", 1.0)),
                             z_floor=float(acfg.get("z_floor", 0.05)), adv_clip_max=float(acfg.get("adv_clip_max", 5.0)),
                             kl_beta=float(acfg.get("kl_beta", 1e-4)), timesteps_per_sample=int(acfg.get("timesteps_per_sample", 1)))
        self.global_std = bool(acfg.get("global_std", True))
        self.old = [p.detach().clone() for p in T.params]
        self.opt = torch.optim.AdamW(T.params, lr=self.cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.n_updates = 0

    def _canvas(self):
        c = self.T.canvas; self.cfg.frame_count, self.cfg.height, self.cfg.width = c["frames"], c["height"], c["width"]

    def supports_group_split(self) -> bool: return True

    def rollout(self, ctx, prompt, seed, members=None):
        """Yields the group's samples (or this rank's `members` of it) one by one: the caller decodes and submits each for
        reward while the next samples. Member k is seeded seed + k on whichever rank makes it. The old policy stays swapped
        in until the generator is exhausted or closed; consume it fully before updating."""
        self._canvas(); ks = list(range(self.cfg.group_size)) if members is None else list(members)
        with swapped_params(self.T.params, self.old):                 # data collection uses the old policy
            for k, smp in zip(ks, iter_samples(self.T.transformer, self.T.network, ctx, self.cfg, self.T.schedule, self.T.device,
                                               [seed + k for k in ks])):
                yield dict(smp, member=k)

    def _step(self):
        T = self.T
        if T.world > 1: average_gradients(T.params, T.world)
        gn = float(torch.nn.utils.clip_grad_norm_(T.params, self.cfg.max_grad_norm)); self.opt.step(); self.opt.zero_grad(set_to_none=True)
        self.n_updates += 1; return gn

    def update(self, samples, iteration):
        T, cfg = self.T, self.cfg; self._canvas()
        sd = {k: self._global_sd([s["R"][k] for s in samples]) if self.global_std else None for k in ("video", "audio")}
        groups = self._groups(samples)          # gid -> every member's rewards, gathered from all ranks sharing the group
        for gid, members in groups.items():
            ks = sorted(members)
            rv = group_r([members[k][0] for k in ks], cfg.z_floor, sd["video"], clip=cfg.adv_clip_max)
            ra = group_r([members[k][1] for k in ks], cfg.z_floor, sd["audio"], clip=cfg.adv_clip_max)
            r_of = {k: (v, u) for k, v, u in zip(ks, rv, ra)}
            for s in samples:
                if s["gid"] == gid: s["r_video"], s["r_audio"] = r_of[s["member"]]
        T.transformer.train(); T.network.set_multiplier(1.0); self.opt.zero_grad(set_to_none=True)
        parts, n, gn = [], 0, 0.0
        ga = cfg.grad_accum // self.ranks_per_group()   # R ranks each hold 1/R of a group: same samples per optimizer step overall
        for _ in range(cfg.epochs_per_batch):
            order = list(range(len(samples))); T.rng.shuffle(order)
            for i in order:
                s = samples[i]
                parts.append(nft_loss(T.transformer, T.network, T.params, self.old, s, s["ctx"], s["r_video"], s["r_audio"], cfg,
                                      T.schedule, T.device, T.rng, loss_scale=1.0 / ga)); n += 1
                if n % ga == 0: gn = self._step()
        if n % ga: gn = self._step()
        ema_update(self.old, T.params, min(cfg.ema_slope * self.n_updates, cfg.ema_max))
        rv = [s["r_video"] for s in samples if s["r_video"] is not None]
        return dict(loss=float(np.mean([p["loss"] for p in parts])), fm_video=float(np.mean([p["fm_video"] for p in parts])),
                    r_video_mean=float(np.mean(rv)) if rv else 0.0, grad_norm=gn, n_updates=float(self.n_updates))

    def _groups(self, samples):
        """{gid: {member: (R video, R audio)}} for this rank's groups, with the members other ranks sampled (ranks_per_group > 1)."""
        mine = [(s["gid"], s["member"], s["R"]["video"], s["R"]["audio"]) for s in samples]
        if self.ranks_per_group() > 1 and self.T.world > 1:
            import torch.distributed as dist
            parts = [None] * self.T.world; dist.all_gather_object(parts, mine); rows = [x for p in parts for x in p]
        else: rows = mine
        local = {g for g, _, _, _ in mine}; out = {}
        for g, k, v, u in rows:
            if g in local: out.setdefault(g, {})[k] = (v, u)
        n = self.cfg.group_size
        for g, m in out.items():
            if sorted(m) != list(range(n)): raise RuntimeError(f"group {g}: members {sorted(m)}, expected 0..{n - 1}")
        return out

    def _global_sd(self, values):
        """Std of every valid (not missing, not gated) reward of this iteration over all ranks (reference global_std)."""
        from h3rl.rewards.combine import WORST
        v = [float(x) for x in values if x is not None and x > WORST]
        if self.T.world > 1:
            import torch.distributed as dist
            parts = [None] * self.T.world; dist.all_gather_object(parts, v); v = [x for p in parts for x in p]
        return float(np.std(v)) if len(v) >= 2 else None

    def state(self): return {"n_updates": self.n_updates}
    def load(self, s): self.n_updates = int(s.get("n_updates", 0))
    def state_tensors(self): return {"old": [o.detach().cpu() for o in self.old], "opt": self.opt.state_dict()}

    def load_tensors(self, s):
        with torch.no_grad():
            for o, v in zip(self.old, s["old"], strict=True): o.copy_(v.to(o.device, o.dtype))
        self.opt.load_state_dict(s["opt"])
