"""DiffusionNFT-on-grid for MiniMax-H3 + HyperFlow (Stage 2 backbone A).

Rollouts are the ordinary 8-step ODE sampler (HyperFlow grid, no injected noise), independent initial
noise per sample. Only the clean final latents are kept. The update is the reward-weighted forward-process
(flow-matching) objective of DiffusionNFT (Zheng et al., ICLR 2026):

    v+ = (1-b) v_old + b v_theta,   v- = (1+b) v_old - b v_theta
    L  = r ||v+ - v||^2 + (1-r) ||v- - v||^2,   v = x0 - eps,  x_t = (1-s) x0 + s eps

with r in [0,1] the group-normalised optimality of the sample and v_old the EMA copy of the LoRA. H3
predicts video and audio velocities separately, so r is per modality (OmniNFT's routing): the video
branch is weighted by the video/sync reward, the audio branch by the audio/sync reward.

Deviation from the paper ("on-grid"): the loss noise level s is drawn from HyperFlow's own 8 sigmas
(jittered), not uniformly, so the model is only supervised at noise levels its sampler visits.
"""
from __future__ import annotations
import os, random, subprocess
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np, torch

from h3rl.core.grpo import GRPOConfig, decode_rollout_to_video
from h3rl.core.sde import h3_sde_step
from h3rl.core.video_metrics import compute_video_metrics


@dataclass
class NFTConfig:
    group_size: int = 8              # K rollouts per prompt (NFT: 24; ours cost ~10x more per sample)
    prompts_per_step: int = 2        # B prompts per rank per iteration
    infer_steps: int = 8
    width: int = 960
    height: int = 544
    frame_count: int = 124
    beta: float = 0.1                # NFT guidance strength (0.1 for multi-reward)
    lr: float = 1e-4
    network_dim: int = 32
    network_alpha: float = 16.0
    epochs_per_batch: int = 2        # off-policy reuse of each rollout batch
    grad_accum: int = 4              # samples per optimizer step (per rank)
    max_grad_norm: float = 1.0
    ema_slope: float = 0.001         # eta_i = min(ema_slope * i, ema_max)
    ema_max: float = 0.5
    sigma_jitter: float = 0.5        # fraction of the inter-sigma gap used as uniform jitter
    audio_loss_weight: float = 1.0
    z_floor: float = 0.05            # floor on the reward scale Z (max-group-std idea)

    def as_grpo(self) -> GRPOConfig:
        return GRPOConfig(group_size=self.group_size, prompts_per_step=self.prompts_per_step, infer_steps=self.infer_steps, noise_level=0.0,
                          kl_beta=0.0, shared_init_noise=False, lr=self.lr, network_dim=self.network_dim, network_alpha=self.network_alpha,
                          width=self.width, height=self.height, frame_count=self.frame_count)


# ----------------------------------------------------------------------------- rollouts
@torch.no_grad()
def sample_group(transformer, network, ctx: dict, cfg: NFTConfig, schedule, device: torch.device, seeds: list[int]) -> list[dict]:
    """K deterministic ODE samples (HyperFlow grid) with independent initial noise; returns clean latents on CPU."""
    from musubi_tuner.minimax_h3.sampling import initialize_target_latents, augment_condition_latents
    from musubi_tuner.minimax_h3.packing import VIDEO_CHANNELS, AUDIO_CHANNELS, STEREO_CHANNELS
    layout = ctx["layout"]; out = []
    network.set_multiplier(1.0)
    for seed in seeds:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        vis_cond, aud_cond = augment_condition_latents(ctx["raw_visual_conditions"], ctx["audio_condition_latents"], generator=gen,
                                                       visual_clean=ctx["visual_condition_clean"], device=device)
        video, audio = initialize_target_latents(
            video_shape=(1, VIDEO_CHANNELS, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
            audio_shape=(1, AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames), generator=gen, device=device)
        for i in range(cfg.infer_steps):
            sv, svn = schedule.video[i], schedule.video[i + 1]; sa, san = schedule.audio[i], schedule.audio[i + 1]
            pred = transformer(video_latents=video, audio_latents=audio, text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                               layout=layout, model_t_video=1.0 - sv, model_t_audio=1.0 - sa, visual_condition_latents=vis_cond,
                               audio_condition_latents=aud_cond, visual_condition_clean=ctx["visual_condition_clean"])
            video, _, _, _ = h3_sde_step(video, pred.video, sv, svn, schedule.video[1], noise_level=0.0)
            audio, _, _, _ = h3_sde_step(audio, pred.audio, sa, san, schedule.audio[1], noise_level=0.0)
            video, audio = video.to(torch.bfloat16), audio.to(torch.bfloat16)
        out.append(dict(seed=int(seed), video=video.cpu(), audio=audio.cpu(),
                        vis_cond=tuple(t.cpu() for t in vis_cond), aud_cond=tuple(t.cpu() for t in aud_cond)))
    return out


def decode_and_write(models: dict, video_lat: torch.Tensor, audio_lat: torch.Tensor, path: str, device: torch.device, fps: int = 24) -> dict:
    """Decode video (+ audio) and mux to mp4; returns the pixel-space guardrail metrics."""
    from musubi_tuner.minimax_h3.sampling import write_audio_wav
    vid = decode_rollout_to_video(video_lat.to(device), audio_lat.to(device), models)          # [F,H,W,3] uint8
    metrics = compute_video_metrics(vid)
    with torch.no_grad():
        wav = models["audio_vae"].decode(audio_lat.to(device=device, dtype=torch.float32)).cpu()
    wav_path = path[:-4] + ".wav"; write_audio_wav(wav[0].float() if wav.ndim == 3 else wav.float(), wav_path, sample_rate=models["audio_vae"].output_sample_rate)   # decode returns [B,2,L]
    F, H, W, _ = vid.shape
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-i", wav_path, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", path],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.stdin.write(np.ascontiguousarray(vid).tobytes()); p.stdin.close(); p.wait()
    try: os.remove(wav_path)
    except OSError: pass
    return metrics


# ----------------------------------------------------------------------------- group-normalised reward r




def group_r(values: list, z_floor: float, global_sd: float | None = None) -> list:
    """NFT eq.: r = 0.5 + 0.5 clip((R - mean_group) / Z, -1, 1); Z = max(global sd, group sd, floor)."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) < 2: return [0.5 if v is not None else None for v in values]
    Z = max(float(arr.std()), z_floor, float(global_sd) if global_sd else 0.0); m = float(arr.mean())
    return [None if v is None else float(0.5 + 0.5 * np.clip((v - m) / Z, -1.0, 1.0)) for v in values]


# ----------------------------------------------------------------------------- update
@contextmanager
def swapped_params(params: list, other: list):
    """Temporarily load `other` tensors into the live LoRA params (for the v_old forward)."""
    backup = [p.data.clone() for p in params]
    with torch.no_grad():
        for p, o in zip(params, other): p.data.copy_(o)
    try: yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup): p.data.copy_(b)


def ema_update(old: list, params: list, eta: float):
    with torch.no_grad():
        for o, p in zip(old, params): o.mul_(eta).add_(p.data, alpha=1.0 - eta)


def draw_sigma(grid: list[float], jitter: float, rng: random.Random) -> float:
    """One of the sampler's interior sigmas, jittered within half the gap to its neighbours."""
    interior = [float(s) for s in grid[:-1]]                      # exclude the final 0.0
    i = rng.randrange(len(interior)); s = interior[i]
    lo = (interior[i + 1] if i + 1 < len(interior) else 0.0); hi = (interior[i - 1] if i > 0 else 1.0)
    s = s + rng.uniform(-jitter, jitter) * min(s - lo, hi - s)
    return float(min(max(s, 0.02), 0.995))


def nft_loss(transformer, network, params: list, old_params: list, sample: dict, ctx: dict, r_video: float | None, r_audio: float | None,
             cfg: NFTConfig, schedule, device: torch.device, rng: random.Random, loss_scale: float = 1.0) -> dict:
    """One sample's reward-weighted FM loss; backward() is called here (grads accumulate into the LoRA)."""
    layout = ctx["layout"]
    x0v, x0a = sample["video"].to(device), sample["audio"].to(device)
    sv = draw_sigma(schedule.video, cfg.sigma_jitter, rng); sa = draw_sigma(schedule.audio, cfg.sigma_jitter, rng)
    ev, ea = torch.randn_like(x0v.float()), torch.randn_like(x0a.float())
    xtv = ((1.0 - sv) * x0v.float() + sv * ev).to(torch.bfloat16); xta = ((1.0 - sa) * x0a.float() + sa * ea).to(torch.bfloat16)
    tv, ta = (x0v.float() - ev), (x0a.float() - ea)                                      # musubi's H3 target: latents - noise
    vis = tuple(t.to(device) for t in sample["vis_cond"]); aud = tuple(t.to(device) for t in sample["aud_cond"])
    fwd = lambda: transformer(video_latents=xtv, audio_latents=xta, text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                              layout=layout, model_t_video=1.0 - sv, model_t_audio=1.0 - sa, visual_condition_latents=vis, audio_condition_latents=aud,
                              visual_condition_clean=ctx["visual_condition_clean"])
    network.set_multiplier(1.0)
    with torch.no_grad(), swapped_params(params, old_params):
        old = fwd(); vo_v, vo_a = old.video.float(), old.audio.float()
    if getattr(transformer, "gradient_checkpointing", False):
        xtv.requires_grad_(True); xta.requires_grad_(True)
    pred = fwd(); vt_v, vt_a = pred.video.float(), pred.audio.float()
    b = cfg.beta
    def branch(vo, vt, target, r):
        vp = (1 - b) * vo + b * vt; vn = (1 + b) * vo - b * vt
        return r * ((vp - target) ** 2).mean() + (1 - r) * ((vn - target) ** 2).mean()
    loss = torch.zeros((), device=device); parts = {}
    if r_video is not None: lv = branch(vo_v, vt_v, tv, r_video); loss = loss + lv; parts["loss_video"] = float(lv)
    if r_audio is not None: la = branch(vo_a, vt_a, ta, r_audio); loss = loss + cfg.audio_loss_weight * la; parts["loss_audio"] = float(la)
    if loss.requires_grad: (loss * loss_scale).backward()
    parts.update(loss=float(loss), sigma_v=sv, sigma_a=sa, fm_video=float(((vt_v - tv) ** 2).mean()))
    del pred, old, vo_v, vo_a, vt_v, vt_a, loss
    return parts
