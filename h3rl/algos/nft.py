"""DiffusionNFT (Zheng et al., 2025): forward-process, likelihood-free RL on the plain ODE sampler.
Rollouts: deterministic ODE samples with independent initial noise. Update: the reward-weighted flow-matching loss
    v+ = (1-b) v_old + b v_theta,  v- = (1+b) v_old - b v_theta,  L = r ||v+ - v||^2 + (1-r) ||v- - v||^2
with r in [0,1] the group-normalised reward (per modality: video and audio branches), v_old an EMA copy of the LoRA.
As in the reference implementation, rollouts are sampled with v_old and the EMA is updated once per iteration, after
the gradient steps: old <- eta old + (1 - eta) theta, eta = min(ema_slope * optimizer steps so far, ema_max)."""
from __future__ import annotations
import numpy as np, torch
from h3rl.algos.base import Algorithm, register
from h3rl.core.nft import NFTConfig, sample_group, group_r, nft_loss, ema_update, swapped_params
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
                             z_floor=float(acfg.get("z_floor", 0.05)))
        self.old = [p.detach().clone() for p in T.params]
        self.opt = torch.optim.AdamW(T.params, lr=self.cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.n_updates = 0

    def _canvas(self):
        c = self.T.canvas; self.cfg.frame_count, self.cfg.height, self.cfg.width = c["frames"], c["height"], c["width"]

    def rollout(self, ctx, prompt, seed):
        self._canvas()
        with swapped_params(self.T.params, self.old):                 # data collection uses the old policy
            return sample_group(self.T.transformer, self.T.network, ctx, self.cfg, self.T.schedule, self.T.device,
                                [seed + k for k in range(self.cfg.group_size)])

    def _step(self):
        T = self.T
        if T.world > 1: average_gradients(T.params, T.world)
        gn = float(torch.nn.utils.clip_grad_norm_(T.params, self.cfg.max_grad_norm)); self.opt.step(); self.opt.zero_grad(set_to_none=True)
        self.n_updates += 1; return gn

    def update(self, samples, iteration):
        T, cfg = self.T, self.cfg; self._canvas()
        for g in sorted({s["group"] for s in samples}):
            grp = [s for s in samples if s["group"] == g]
            rv = group_r([s["R"]["video"] for s in grp], cfg.z_floor); ra = group_r([s["R"]["audio"] for s in grp], cfg.z_floor)
            for s, v, u in zip(grp, rv, ra): s["r_video"], s["r_audio"] = v, u
        T.transformer.train(); T.network.set_multiplier(1.0); self.opt.zero_grad(set_to_none=True)
        parts, n, gn = [], 0, 0.0
        for _ in range(cfg.epochs_per_batch):
            order = list(range(len(samples))); T.rng.shuffle(order)
            for i in order:
                s = samples[i]
                parts.append(nft_loss(T.transformer, T.network, T.params, self.old, s, s["ctx"], s["r_video"], s["r_audio"], cfg,
                                      T.schedule, T.device, T.rng, loss_scale=1.0 / cfg.grad_accum)); n += 1
                if n % cfg.grad_accum == 0: gn = self._step()
        if n % cfg.grad_accum: gn = self._step()
        ema_update(self.old, T.params, min(cfg.ema_slope * self.n_updates, cfg.ema_max))
        rv = [s["r_video"] for s in samples if s["r_video"] is not None]
        return dict(loss=float(np.mean([p["loss"] for p in parts])), fm_video=float(np.mean([p["fm_video"] for p in parts])),
                    r_video_mean=float(np.mean(rv)) if rv else 0.0, grad_norm=gn, n_updates=float(self.n_updates))

    def state(self): return {"n_updates": self.n_updates}
    def load(self, s): self.n_updates = int(s.get("n_updates", 0))
    def state_tensors(self): return {"old": [o.detach().cpu() for o in self.old], "opt": self.opt.state_dict()}

    def load_tensors(self, s):
        with torch.no_grad():
            for o, v in zip(self.old, s["old"], strict=True): o.copy_(v.to(o.device, o.dtype))
        self.opt.load_state_dict(s["opt"])
