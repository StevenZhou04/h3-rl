"""Flow-GRPO core for MiniMax-H3: model loading for RL, sampling schedules, conditioning context, SDE rollouts with
per-step log-probs (h3rl.core.sde.h3_sde_step), the clipped-ratio policy loss and the SAGE-GRPO switches (precise step
variance, per-timestep gradient equalizer, dual trust region). Trains a LoRA on the frozen transformer.

Policy / reference: one transformer + one LoRA network gives both. Each LoRA module reads its own multiplier at forward
time, so the switch must go through LoRANetwork.set_multiplier() (1.0 = policy, 0.0 = frozen base); assigning
network.multiplier alone leaves the modules untouched and silently makes the reference identical to the policy.
The clipped-ratio + group-relative-advantage structure follows Flow-GRPO (github.com/yifan123/flow_grpo), adapted to
H3's joint video + audio output.
"""
from __future__ import annotations

import dataclasses
import json
import json
from pathlib import Path
from typing import Callable

import math
import numpy as np
import torch

from h3rl.core.sde import h3_sde_step


@dataclasses.dataclass
class GRPOConfig:
    group_size: int = 8          # rollouts per unique prompt (G in GRPO)
    prompts_per_step: int = 4    # unique prompts per training step (B)
    infer_steps: int = 8         # H3's --infer_steps; confirm the real default/value used elsewhere before trusting
    noise_level: float = 0.7     # SDE stochasticity; flow_grpo's own default, ported as-is
    clip_range: float = 0.2      # PPO clip epsilon
    adv_clip_max: float = 5.0
    grad_steps_per_traj: int = 0  # backprop through only k of the recorded denoising steps (0 = all)
    shared_init_noise: bool = True  # one initial latent per group; diversity from SDE noise only
    global_std: bool = True      # normalize advantages by the BATCH std, not the within-group std
    best_of_n: int = 0           # keep top-k and bottom-k of each group for the update (0 = keep all)
    dimension_wise: bool = False # z-score each reward axis within the group, then combine
    dim_std_floor: float = 0.25  # an axis whose within-group spread is below this is ignored
    kl_beta: float = 0.01        # 0 disables the KL-to-reference term
    lr: float = 1e-4
    network_dim: int = 32        # LoRA rank -- untuned default, not verified against this workload
    network_alpha: float = 16.0
    width: int = 896
    height: int = 512
    frame_count: int = 73        # matches one of the camera-adapter dataset's real bucket sizes
    reward_weights: dict | None = None  # per-axis weights into RewardScore.scalar()
    # --- SAGE-GRPO (backbone B) switches; all off = the validated Flow-GRPO path
    sde_precise: bool = False          # integrated step std instead of first-order
    grad_equalizer: bool = False       # per-timestep loss scale S_t = median(N)/(N_t+eps) from running grad norms
    tr_pos_beta: float = 0.0           # KL to a LoRA snapshot refreshed every ref_refresh_every steps (position)
    tr_vel_beta: float = 0.0           # KL to the previous step's LoRA (velocity)
    ref_refresh_every: int = 20


# ---------------------------------------------------------------------------
# Model loading / attach -- grounded in minimax_h3_train_network.py's
# load_transformer() and minimax_h3_generate_video.py's LoRA attach pattern.
# ---------------------------------------------------------------------------

def read_hyperflow_metadata(path: str) -> dict:
    """The HyperFlow distilled 8-step adapter stores its sampling contract in the safetensors
    header, and the model card is explicit that the step count and sigma grid come from the
    weights file -- passing a different count is an error. So we read them rather than assume.

    Its grid is NOT H3's shift formula: build_shifted_schedule(8, video_shift=12) gives
    [1.0, .988, .973, .952, .923, .878, .800, .632, 0] -- seven of eight steps above 0.8 and then a
    cliff -- while HyperFlow ships a symmetric, evenly spread
    [1.0, .932, .839, .704, .500, .297, .161, .068, 0]. That difference matters for RL beyond
    sample quality: our SDE exploration noise scales as sqrt(sigma/(1-sigma)), which is ~9.9 at
    sigma=0.99 and ~1.0 at sigma=0.5, so the stock grid injects enormous noise at nearly every
    step while this grid stays well conditioned throughout.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    if not meta.get("hyperflow"):
        raise ValueError(f"{path} does not carry HyperFlow metadata (key 'hyperflow')")
    sigmas = json.loads(meta["hyperflow_sigmas"])
    return dict(
        sigmas=[float(x) for x in sigmas],
        steps=len(sigmas) - 1,
        video_shift=float(meta.get("hyperflow_video_shift", 12.0)),
        audio_shift=float(meta.get("hyperflow_audio_shift", 3.0)),
        rank=int(meta.get("lora_rank", 0) or 0),
        version=meta.get("hyperflow_version", "?"),
    )


# MiniMax-H3's released duration range, and the frame-count lattice the layout builder enforces.
# TARGET_FPS is 24, so frames / 24 must land inside [5, 15] seconds.
H3_MIN_SECONDS, H3_MAX_SECONDS = 5.0, 15.0


def validate_clip_duration(frame_count: int, *, fps: int = 24) -> float:
    """Fail loudly if a config would train outside H3's supported clip length.

    This exists because the RL loop calls build_generation_layout DIRECTLY and therefore never
    passes through `validate_generation_request`, which is where the generation CLI enforces this.
    Run 1 trained 37 steps at 73 frames = 3.04s as a result -- outside the released 5-15s range,
    and only noticed when the CLI refused to render a comparison video of the same config
    ("duration 3.042s is outside the released 5-15s range; pass --allow_experimental_duration").
    A silent out-of-range run wastes the whole budget optimizing a regime nobody deploys, so this
    is a hard error rather than a warning.
    """
    if (frame_count - 5) % 17 != 0 or frame_count < 5:
        raise ValueError(
            f"MiniMax-H3 frame count must satisfy 17n+5 (5, 22, 39, 56, 73, 90, 107, 124, ...), got {frame_count}")
    seconds = frame_count / fps
    if not (H3_MIN_SECONDS <= seconds <= H3_MAX_SECONDS):
        nearest = [n for n in range(5, 400, 17) if H3_MIN_SECONDS <= n / fps <= H3_MAX_SECONDS]
        raise ValueError(
            f"MiniMax-H3 clip duration {seconds:.2f}s (frame_count={frame_count}) is outside the released "
            f"{H3_MIN_SECONDS:.0f}-{H3_MAX_SECONDS:.0f}s range. Valid frame counts here: "
            f"{nearest[0]} ({nearest[0]/fps:.2f}s) .. {nearest[-1]} ({nearest[-1]/fps:.2f}s)")
    return seconds


def make_schedule(cfg: GRPOConfig, device: torch.device, hyperflow: dict | None = None):
    """The sigma schedule for rollouts AND eval. One place, so the two can never diverge."""
    from musubi_tuner.minimax_h3.sampling import build_shifted_schedule, shift_sigma, H3SigmaSchedule

    if hyperflow is None:
        return build_shifted_schedule(cfg.infer_steps, device=device)

    if cfg.infer_steps != hyperflow["steps"]:
        raise ValueError(
            f"HyperFlow fixes the step count at {hyperflow['steps']} (its sigma grid is stored in the "
            f"weights); --infer_steps {cfg.infer_steps} would sample off that grid")
    video = torch.tensor(hyperflow["sigmas"], dtype=torch.float64, device=device)
    # Audio keeps H3's own shifted schedule at the audio shift recorded in the adapter: the file
    # ships ONE sigma grid and separately records audio_shift=3.0, which is H3's stock default,
    # so the adapter appears to retime the video branch only. Audio is not judged by our reward,
    # so an error here costs little -- but it is an assumption, not a documented fact.
    base = torch.linspace(1.0, 0.0, hyperflow["steps"] + 1, dtype=torch.float64, device=device)
    return H3SigmaSchedule(base=base, video=video, audio=shift_sigma(base, hyperflow["audio_shift"]))


def load_h3_for_rl(*, dit_path: str, text_encoder_path: str, video_vae_path: str, audio_vae_path: str,
                    device: torch.device, cfg: GRPOConfig, attn_mode: str | None = None, split_attn: bool = False,
                    skip_text_encoder: bool = False, base_lora_paths: list | None = None):
    from musubi_tuner.minimax_h3.model import load_h3_transformer
    from musubi_tuner.minimax_h3.text_encoder import load_h3_processor, load_h3_text_encoder
    from musubi_tuner.minimax_h3.video_vae import load_video_vae
    from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
    from musubi_tuner.networks import lora_minimax_h3

    transformer = load_h3_transformer(
        dit_path, device=device, dtype=torch.bfloat16, attn_mode=attn_mode, split_attn=split_attn,
    )
    transformer.requires_grad_(False)  # frozen base -- the same automatic-freezing pattern used for the camera adapter

    # Frozen adapters that are part of the BASE policy (e.g. the HyperFlow 8-step distillation)
    # are attached, not merged: this checkpoint is pre-quantized INT8, whose storage grid would
    # round away a merged delta. attach_lora_weights wraps each Linear's forward, and our own
    # trainable LoRA is applied afterwards so it stacks on top -- which also means
    # network.set_multiplier(0.0) for the KL reference zeroes only OUR adapter and correctly
    # leaves HyperFlow in the reference policy, where it belongs.
    base_networks = []
    if base_lora_paths:
        from musubi_tuner.utils.lora_utils import attach_lora_weights

        base_networks = attach_lora_weights(
            lora_minimax_h3, transformer, list(base_lora_paths), [1.0] * len(base_lora_paths),
            None, None, device,
        )
        for net in base_networks:
            net.requires_grad_(False)
            net.eval()

    network = lora_minimax_h3.create_arch_network(
        1.0, cfg.network_dim, cfg.network_alpha, vae=None, text_encoders=None, unet=transformer,
    )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.to(device)
    # Standard LoRANetwork (unlike the camera adapter's custom wrapper): its trainable
    # parameters live only in the LoRAModule objects inside `network`, attached to
    # `transformer`'s Linears via monkey-patched forward, never as real nn.Module
    # submodules of `transformer` itself -- so it does NOT hit the DDP double-registration
    # bug the camera adapter needed a workaround for. Safe to accelerator.prepare()
    # transformer and network separately, same as every other LoRA training path in this repo.

    # skip_text_encoder: the driver encodes its whole prompt set up front and frees the 32B
    # encoder before this call, so loading it again here would just re-reserve ~16 GiB that
    # nothing in the training loop reads.
    processor = None if skip_text_encoder else load_h3_processor()
    text_encoder = None if skip_text_encoder else load_h3_text_encoder(text_encoder_path, device=device, dtype=torch.bfloat16)
    video_vae = load_video_vae(video_vae_path, device=device)
    audio_vae = load_audio_vae(audio_vae_path, device=device)

    return dict(
        transformer=transformer, network=network, processor=processor,
        text_encoder=text_encoder, video_vae=video_vae, audio_vae=audio_vae,
        base_networks=base_networks,
    )


# ---------------------------------------------------------------------------
# Per-prompt layout + text conditioning -- grounded in generation_inputs.py /
# text_encoder.py's real t2va path (build_presentation only reads
# record.caption for t2va; no visual/audio conditions for this task).
# ---------------------------------------------------------------------------

def encode_prompt(prompt: str, processor, text_encoder, device: torch.device, *,
                  task: str = "t2va", condition_frames: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Text conditioning -- split out from context construction so the 32B text encoder can be
    loaded, used on every prompt once, and freed BEFORE the transformer is loaded. Holding both
    at once is what makes this not fit on a shared GPU.

    For task="fl2va" the first frame goes through the encoder's VISION branch as well: H3 builds
    a "<Picture 1>: <image>" + caption presentation, so the image-conditioned hidden states are a
    different length and different content from the same prompt's t2va encoding. The two are not
    interchangeable -- a prompt trained in both modes needs both encodings cached separately.
    `condition_frames` is a uint8 [1,H,W,3] frame on the target canvas (load_image_frames)."""
    from musubi_tuner.minimax_h3.media import H3Record
    from musubi_tuner.minimax_h3.text_encoder import build_presentation, encode_h3_presentation, H3TextVisual

    record = H3Record(video_path=Path("."), caption=prompt, references=())
    if task == "fl2va":
        if condition_frames is None:
            raise ValueError("MiniMax-H3 FL2VA needs a first frame -- it cannot be conditioned on text alone")
        # "first" is one of packing.FL_CONDITION_ROLES; a lone first frame is the I2VA case.
        visuals = {"first": H3TextVisual(condition_frames)}
        presentation = build_presentation(record, task="fl2va", visuals=visuals)
    else:
        presentation = build_presentation(record, task="t2va")
    hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
    return hidden_states.to(torch.bfloat16).unsqueeze(0).to(device), token_tags.unsqueeze(0).to(device)


def encode_condition_latent(condition_frames: torch.Tensor, video_vae, device: torch.device):
    """VAE-encode the FL2VA first frame -> (latent, H3VideoGeometry). Runs in phase B because it
    needs the video VAE, while the vision-branch text encoding above runs in phase A; the same
    image feeds both."""
    from musubi_tuner.minimax_h3.media import prepare_pixels
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition
    from musubi_tuner.minimax_h3.packing import H3VideoGeometry

    with torch.no_grad():
        # The video VAE is an fp16 checkpoint (VIDEO_VAE_DECODE_DTYPE), NOT the transformer's
        # bfloat16 -- feeding it bf16 pixels raises "Input type (c10::BFloat16) and bias type
        # (c10::Half) should be the same" inside its first conv. Take the dtype from the module
        # itself rather than hardcoding either one, so a differently-cast VAE still works.
        vae_dtype = next(video_vae.parameters()).dtype
        pixels = prepare_pixels(condition_frames).to(device=device, dtype=vae_dtype)
        latent = encode_video_condition(video_vae, pixels)
    # the transformer consumes conditions in its own dtype
    latent = latent.to(torch.bfloat16)
    return latent, H3VideoGeometry(*latent.shape[2:])


def build_context(prompt: str, hidden_states: torch.Tensor, token_tags: torch.Tensor,
                  cfg: GRPOConfig, device: torch.device, *, task: str = "t2va",
                  condition_latent: torch.Tensor | None = None,
                  condition_geometry=None, condition_path: str = "first_frame.png",
                  condition_frame=None) -> dict:
    """Layout + conditioning bundle for one prompt. Pure bookkeeping -- no model needed, so this
    runs per training step against cached text embeddings and (for FL2VA) a cached first-frame
    latent.

    FL2VA note: build_generation_layout derives the anchor roles from the request's condition
    IMAGE PATHS (fl_condition_entries reads --first_frame / --last_frame), not from the latents,
    so `condition_path` only has to be a non-empty string for the "first" role to exist. The
    geometry of the encoded latent is what actually sizes the condition segment."""
    from musubi_tuner.minimax_h3.generation_inputs import H3GenerationRequest, build_generation_layout
    from musubi_tuner.minimax_h3.sampling import DEFAULT_VIDEO_SHIFT, DEFAULT_AUDIO_SHIFT, DEFAULT_VISUAL_CONDITION_CLEAN, DEFAULT_AUDIO_CONDITION_CLEAN

    validate_clip_duration(cfg.frame_count)
    hidden_states = hidden_states.to(device)
    token_tags = token_tags.to(device)
    if task == "fl2va" and (condition_latent is None or condition_geometry is None):
        raise ValueError("MiniMax-H3 FL2VA context needs the encoded first-frame latent and its geometry")

    request = H3GenerationRequest(
        task=task, prompt=prompt, width=cfg.width, height=cfg.height, frame_count=cfg.frame_count,
        steps=cfg.infer_steps, h3_shift_video=DEFAULT_VIDEO_SHIFT, h3_shift_audio=DEFAULT_AUDIO_SHIFT,
        h3_visual_cond_clean=DEFAULT_VISUAL_CONDITION_CLEAN, h3_audio_cond_clean=DEFAULT_AUDIO_CONDITION_CLEAN,
        **({"first_frame": condition_path} if task == "fl2va" else {}),
    )
    layout = build_generation_layout(
        request, text_length=hidden_states.shape[1],
        visual_geometries=(condition_geometry,) if task == "fl2va" else (),
    )

    return dict(
        task=task, layout=layout, text_hidden_states=hidden_states, text_token_tags=token_tags,
        # The conditioning image as PIXELS, for the reward judge. The latent above is what the
        # model consumes; the judge needs to see what the model was asked to continue from,
        # otherwise first_frame_fidelity has nothing to compare against and stays inert on the
        # 300 of 560 prompts where matching the given first frame IS the task.
        condition_frame=condition_frame,
        # the RAW (un-augmented) condition latent; rollout_group augments it per rollout, exactly
        # as sample_joint_av_latents does before its sampling loop
        raw_visual_conditions=(condition_latent,) if task == "fl2va" else (),
        visual_condition_clean=DEFAULT_VISUAL_CONDITION_CLEAN,
        audio_condition_latents=(),
    )


def build_layout_and_conditions(prompt: str, models: dict, cfg: GRPOConfig, device: torch.device) -> dict:
    """Encode + build in one call (t2va only). Convenience path for when the text encoder is
    resident; the FL2VA path goes through the driver, which caches image encodings in phase A."""
    hidden_states, token_tags = encode_prompt(prompt, models["processor"], models["text_encoder"], device)
    return build_context(prompt, hidden_states, token_tags, cfg, device)


def decode_rollout_to_video(video_latents: torch.Tensor, audio_latents: torch.Tensor, models: dict) -> np.ndarray:
    """Returns [F,H,W,3] uint8 -- exactly what the reward workers read.
    Mirrors minimax_h3_generate_video.py's _decode_and_save (video_vae.decode +
    decoded_video_to_uint8); audio is decoded too (available if a future reward
    axis judges audio) but not sent to the current video-only judge."""
    from musubi_tuner.minimax_h3.sampling import decoded_video_to_uint8
    from musubi_tuner.minimax_h3.video_vae import VIDEO_VAE_DECODE_DTYPE

    with torch.no_grad():
        # VIDEO_VAE_DECODE_DTYPE is float16, not the transformer's bfloat16 -- the VAE is an
        # fp16 checkpoint and minimax_h3_generate_video.py casts to this same constant.
        decoded_video = models["video_vae"].decode(video_latents.to(dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
    frame_limit = decoded_video.shape[2]
    return decoded_video_to_uint8(decoded_video, frame_limit=frame_limit).numpy()


# ---------------------------------------------------------------------------
# GRPO mechanics
# ---------------------------------------------------------------------------





def rollout_group(transformer, network, *, prompt: str, ctx: dict, cfg: GRPOConfig, device: torch.device,
                  schedule=None, trust_refs: dict | None = None):
    """Generate cfg.group_size stochastic rollouts for one prompt at the CURRENT
    policy (network.set_multiplier(1.0)), also recording the reference model's
    (network.set_multiplier(0.0)) step means for the KL term. No gradients here --
    matches DDPO/GRPO practice of only tracking gradients during the separate
    update pass (grpo_loss_for_trajectory)."""
    from musubi_tuner.minimax_h3.sampling import initialize_target_latents
    from musubi_tuner.minimax_h3.packing import VIDEO_CHANNELS, AUDIO_CHANNELS, STEREO_CHANNELS

    layout = ctx["layout"]
    if schedule is None:
        schedule = make_schedule(cfg, device)
    # second-highest sigma (NOT schedule.video[0]/audio[0], which are exactly 1.0) --
    # see h3_sde_sampling.std_dev_t_for_sigma's docstring for why.
    sigma_max_v, sigma_max_a = schedule.video[1], schedule.audio[1]

    from musubi_tuner.minimax_h3.sampling import augment_condition_latents

    # SHARED INITIAL NOISE across the group. GRPO's advantage is only meaningful if the reward
    # differences it measures are CAUSED by the policy's own stochastic choices -- the SDE noise
    # injected at each denoising step, which enters the log-probability and therefore the
    # score-function gradient. Giving each group member a different initial latent adds variance
    # from a variable the log-prob knows nothing about: the advantage then partly measures "which
    # starting noise was luckier", and that component of the gradient points in a random
    # direction. The observable signature is exactly what this run produced -- healthy within-group
    # reward spread (0.05), zero_std_ratio 0.00, real weight movement (4.4% pixel change by step
    # 80), and a held-out eval that never leaves +/-1 sigma.
    # DanceGRPO states it directly for video: "assigning different noise vectors to samples with
    # the same prompts always leads to reward hacking phenomena in video generation, including
    # training instability" (arXiv 2505.07818). flow_grpo exposes the same switch as `same_latent`.
    # The condition augmentation is shared for the same reason: it is drawn outside the policy too.
    shared_latents = shared_conditions = None
    if cfg.shared_init_noise:
        shared_conditions = augment_condition_latents(
            ctx["raw_visual_conditions"], ctx["audio_condition_latents"],
            generator=None, visual_clean=ctx["visual_condition_clean"], device=device,
        )
        shared_latents = initialize_target_latents(
            video_shape=(1, VIDEO_CHANNELS, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
            audio_shape=(1, AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames),
            generator=torch.Generator(device="cpu"), device=device,
        )

    trajectories = []
    for _ in range(cfg.group_size):
        # Condition augmentation is per ROLLOUT, not per step: sample_joint_av_latents blends
        # noise into the condition latents once before its sampling loop (clean*x + (1-clean)*eps)
        # and then holds them fixed. Re-augmenting per step would make the conditioning
        # non-stationary within a trajectory and break the log-prob ratio the PPO term depends on.
        # generator=None draws from the global RNG, which is what training does.
        if cfg.shared_init_noise:
            vis_cond, aud_cond = shared_conditions
            video, audio = (shared_latents[0].clone(), shared_latents[1].clone())
        else:
            vis_cond, aud_cond = augment_condition_latents(
                ctx["raw_visual_conditions"], ctx["audio_condition_latents"],
                generator=None, visual_clean=ctx["visual_condition_clean"], device=device,
            )
            video, audio = initialize_target_latents(
                video_shape=(1, VIDEO_CHANNELS, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
                audio_shape=(1, AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames),
                generator=torch.Generator(device="cpu"),
                device=device,
            )
        steps_record = []
        with torch.no_grad():
            for i in range(cfg.infer_steps):
                # The final step is taken DETERMINISTICALLY and is not recorded. H3's schedule
                # ends with a huge drop to sigma=0, so a stochastic last step leaves ~0.73 std
                # of noise on the final latent with nothing left to denoise it -- measured to
                # produce pure noise (see h3_sde_sampling's docstring). A deterministic step has
                # no transition probability, so it carries no policy gradient either; excluding
                # it from steps_record removes it from both the ratio and the KL term.
                is_last = i == cfg.infer_steps - 1
                step_noise = 0.0 if is_last else cfg.noise_level

                sigma_v, sigma_v_next = schedule.video[i], schedule.video[i + 1]
                sigma_a, sigma_a_next = schedule.audio[i], schedule.audio[i + 1]

                network.set_multiplier(1.0)
                pred_policy = transformer(
                    video_latents=video, audio_latents=audio,
                    text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                    layout=layout, model_t_video=1.0 - sigma_v, model_t_audio=1.0 - sigma_a,
                    visual_condition_latents=vis_cond, audio_condition_latents=aud_cond,
                    visual_condition_clean=ctx["visual_condition_clean"],
                )
                next_video, logp_v, mean_v, std_v = h3_sde_step(
                    video, pred_policy.video, sigma_v, sigma_v_next, sigma_max_v, noise_level=step_noise, precise_std=cfg.sde_precise
                )
                next_audio, logp_a, mean_a, std_a = h3_sde_step(
                    audio, pred_policy.audio, sigma_a, sigma_a_next, sigma_max_a, noise_level=step_noise, precise_std=cfg.sde_precise
                )

                # Skip the reference-model forward entirely when KL-to-reference is
                # disabled -- it's a second full transformer forward per step, doubling
                # rollout cost, and mean_v_ref/mean_a_ref are only ever read by the KL
                # term in grpo_loss_for_trajectory. Confirmed via rollout_benchmark.py
                # that a single forward is ~8.8s/step, so this matters a lot at scale.
                mean_v_ref = mean_a_ref = None
                if cfg.kl_beta > 0 and not is_last:
                    network.set_multiplier(0.0)  # reference pass: same transformer, LoRA contribution zeroed
                    pred_ref = transformer(
                        video_latents=video, audio_latents=audio,
                        text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                        layout=layout, model_t_video=1.0 - sigma_v, model_t_audio=1.0 - sigma_a,
                        visual_condition_latents=vis_cond, audio_condition_latents=aud_cond,
                        visual_condition_clean=ctx["visual_condition_clean"],
                    )
                    _, _, mean_v_ref, _ = h3_sde_step(video, pred_ref.video, sigma_v, sigma_v_next, sigma_max_v, noise_level=step_noise, prev_sample=next_video, precise_std=cfg.sde_precise)
                    _, _, mean_a_ref, _ = h3_sde_step(audio, pred_ref.audio, sigma_a, sigma_a_next, sigma_max_a, noise_level=step_noise, prev_sample=next_audio, precise_std=cfg.sde_precise)
                    network.set_multiplier(1.0)
                # SAGE-GRPO dual trust region: reference means under the periodic LoRA snapshot (position) and the
                # previous step's LoRA (velocity). trust_refs = {"pos": [tensors], "vel": [tensors]} or None.
                tr_means = {}
                if trust_refs and not is_last:
                    live = [q for q in network.parameters() if q.requires_grad]
                    for name, ref in trust_refs.items():
                        if ref is None: continue
                        backup = [q.data.clone() for q in live]
                        with torch.no_grad():
                            for q, o in zip(live, ref): q.data.copy_(o)
                            pred_tr = transformer(
                                video_latents=video, audio_latents=audio,
                                text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                                layout=layout, model_t_video=1.0 - sigma_v, model_t_audio=1.0 - sigma_a,
                                visual_condition_latents=vis_cond, audio_condition_latents=aud_cond,
                                visual_condition_clean=ctx["visual_condition_clean"],
                            )
                            for q, b in zip(live, backup): q.data.copy_(b)
                        _, _, mv, _ = h3_sde_step(video, pred_tr.video, sigma_v, sigma_v_next, sigma_max_v, noise_level=step_noise, prev_sample=next_video, precise_std=cfg.sde_precise)
                        _, _, ma, _ = h3_sde_step(audio, pred_tr.audio, sigma_a, sigma_a_next, sigma_max_a, noise_level=step_noise, prev_sample=next_audio, precise_std=cfg.sde_precise)
                        tr_means[name] = (mv.cpu(), ma.cpu())

                if is_last:
                    video, audio = next_video, next_audio
                    continue
                # .cpu() on the stored inputs: with group_size x (infer_steps-1) records alive
                # until the update pass, these dominate resident memory, and each is read back
                # exactly once. The transfer is microseconds against an ~8.8s forward.
                steps_record.append(dict(
                    video=video.cpu(), audio=audio.cpu(),
                    # The ACTION taken at this step. Without it the update pass has nothing to
                    # evaluate the new policy against and h3_sde_step silently draws a fresh
                    # sample instead -- which makes E[grad] exactly zero. See the comment in
                    # grpo_loss_for_trajectory. CPU like the rest of the record; these are host
                    # tensors, so the extra copy costs RAM, not VRAM.
                    next_video=next_video.cpu(), next_audio=next_audio.cpu(),
                    sigma_v=sigma_v, sigma_v_next=sigma_v_next, sigma_a=sigma_a, sigma_a_next=sigma_a_next,
                    sigma_max_v=sigma_max_v, sigma_max_a=sigma_max_a,
                    log_prob_old=(logp_v + logp_a),
                    mean_v_ref=None if mean_v_ref is None else mean_v_ref.cpu(), std_v=std_v,
                    mean_a_ref=None if mean_a_ref is None else mean_a_ref.cpu(), std_a=std_a,
                    tr_means=tr_means, step_index=i,
                ))
                video, audio = next_video, next_audio

        trajectories.append(dict(
            prompt=prompt, steps=steps_record, final_video=video, final_audio=audio,
            # the SAME augmented conditions must be replayed in grpo_loss_for_trajectory: the
            # recomputed log-probs are only comparable to log_prob_old under identical conditioning
            visual_conditions=tuple(t.cpu() for t in vis_cond),
            audio_conditions=tuple(t.cpu() for t in aud_cond),
        ))
    network.set_multiplier(1.0)
    return trajectories


class GradNormEqualizer:
    """SAGE-GRPO's per-timestep loss scale S_t = median_tau(N_tau) / (N_t + eps), N_t = running mean of the
    parameter-gradient norm produced by timestep t's policy loss. Measured per step as the norm of the
    gradient increment (grads accumulate across steps), so it costs one grad copy per step."""
    def __init__(self, n_steps: int, momentum: float = 0.9, eps: float = 1e-6):
        self.n = [None] * n_steps; self.momentum = momentum; self.eps = eps
    def scale(self, t: int) -> float:
        known = [x for x in self.n if x is not None]
        if len(known) < 2 or self.n[t] is None: return 1.0
        return float(np.median(known) / (self.n[t] + self.eps))
    def update(self, t: int, norm: float):
        self.n[t] = norm if self.n[t] is None else self.momentum * self.n[t] + (1 - self.momentum) * norm


def grpo_loss_for_trajectory(transformer, network, traj: dict, advantage: float, cfg: GRPOConfig, ctx: dict,
                             *, loss_scale: float = 1.0, equalizer: "GradNormEqualizer | None" = None):
    """Recompute log-probs WITH gradients for one trajectory's steps and backward EACH STEP
    immediately, returning the loss as a float.

    The previous version accumulated `total_loss` as a live tensor across every step and let the
    caller call .backward() once -- which kept the autograd graph of all infer_steps forwards
    alive simultaneously, the exact opposite of what its own docstring claimed and the cause of
    a measured OOM (63.6 GiB in use, allocation of 1.09 GiB refused) on the first real step.
    Backward-per-step keeps at most one forward's activations resident, at no cost in
    correctness: gradients accumulate into .grad across steps and trajectories, so the caller
    still does one optimizer.step() per GRPO step. `loss_scale` folds in the 1/num_trajectories
    averaging the caller used to apply.

    Matches flow_grpo's per-step loop (train_sd3.py:886-902)."""
    device = ctx["text_hidden_states"].device
    advantage_t = torch.clamp(torch.as_tensor(advantage, device=device), -cfg.adv_clip_max, cfg.adv_clip_max)
    layout = ctx["layout"]
    recorded = traj["steps"]

    # Denoising-step subsampling. The per-trajectory loss is a MEAN over its steps, and the mean
    # over a uniform random subset is an unbiased estimator of it -- so backpropagating through k
    # of the T recorded steps costs k/T of the update pass for the same expected gradient, just
    # with more variance. The update pass is roughly half of a GRPO step's wall clock here, so
    # k=3 of 7 is a ~1.5x end-to-end speedup that does NOT cost optimizer steps.
    # Each step is still sampled independently per trajectory, so across a group every step index
    # gets covered.
    if cfg.grad_steps_per_traj and cfg.grad_steps_per_traj < len(recorded):
        pick = torch.randperm(len(recorded))[: cfg.grad_steps_per_traj].tolist()
        steps_used = [recorded[i] for i in sorted(pick)]
    else:
        steps_used = recorded
    n_steps = len(steps_used)
    total_loss = 0.0
    # Ratio statistics are the only window onto whether the policy gradient is doing anything.
    # They are also how clip_range gets set: the log-prob is reduced with .mean() over every
    # latent element (h3_sde_sampling.py), so the ratio exponent is scaled by 1/D and sits very
    # close to 1 -- a PPO epsilon of 0.2 can never bind. DanceGRPO uses 1e-4 for this reason.
    # Measure |ratio-1| here, then set cfg.clip_range from its upper percentile.
    ratio_log: list[float] = []

    network.set_multiplier(1.0)
    for step in steps_used:
        # recorded latents live on the CPU (see rollout_group); bring back one step at a time
        video = step["video"].to(device)
        audio = step["audio"].to(device)
        if getattr(transformer, "gradient_checkpointing", False):
            # torch.utils.checkpoint recomputes a block only if at least one INPUT requires grad;
            # our trainable parameters live inside the blocks, so without this the checkpointed
            # blocks contribute no gradient at all. Same requires_grad_ dance as
            # minimax_h3_train_network.py:1333.
            video.requires_grad_(True)
            audio.requires_grad_(True)
        pred = transformer(
            video_latents=video, audio_latents=audio,
            text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
            layout=layout, model_t_video=1.0 - step["sigma_v"], model_t_audio=1.0 - step["sigma_a"],
            visual_condition_latents=tuple(t.to(device) for t in traj.get("visual_conditions", ())),
            audio_condition_latents=tuple(t.to(device) for t in traj.get("audio_conditions", ())),
            visual_condition_clean=ctx["visual_condition_clean"],
        )
        # `prev_sample=` is load-bearing and its absence is silent. h3_sde_step draws a fresh
        # sample when prev_sample is None, so omitting it evaluates log pi(new random draw)
        # instead of log pi(action actually taken). The gradient then becomes
        #     d/dmean [-(mean.detach() + s*eps - mean)^2 / 2s^2] = eps/s
        # i.e. advantage-weighted white noise with E[grad] = 0 -- the policy random-walks and
        # the reward never reaches the weights. That is what run 1, arm A and arm C all did.
        # The rollout's reference pass always did this correctly (prev_sample=next_video above).
        _, logp_v_new, mean_v_new, _ = h3_sde_step(
            video, pred.video, step["sigma_v"], step["sigma_v_next"], step["sigma_max_v"],
            noise_level=cfg.noise_level, prev_sample=step["next_video"].to(device), precise_std=cfg.sde_precise,
        )
        _, logp_a_new, mean_a_new, _ = h3_sde_step(
            audio, pred.audio, step["sigma_a"], step["sigma_a_next"], step["sigma_max_a"],
            noise_level=cfg.noise_level, prev_sample=step["next_audio"].to(device), precise_std=cfg.sde_precise,
        )

        ratio = torch.exp((logp_v_new + logp_a_new) - step["log_prob_old"])
        ratio_log.append(float(ratio.detach().mean()))
        unclipped = -advantage_t * ratio
        clipped = -advantage_t * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
        policy_loss = torch.maximum(unclipped, clipped).mean()

        loss = policy_loss
        if cfg.kl_beta > 0 and step["mean_v_ref"] is not None:
            kl_v = ((mean_v_new - step["mean_v_ref"].to(device)) ** 2).mean() / (2.0 * step["std_v"] ** 2 + 1e-8)
            kl_a = ((mean_a_new - step["mean_a_ref"].to(device)) ** 2).mean() / (2.0 * step["std_a"] ** 2 + 1e-8)
            loss = loss + cfg.kl_beta * (kl_v + kl_a)
        for name, beta in (("pos", cfg.tr_pos_beta), ("vel", cfg.tr_vel_beta)):      # SAGE-GRPO dual trust region
            ref = step.get("tr_means", {}).get(name)
            if beta > 0 and ref is not None:
                kl_v = ((mean_v_new - ref[0].to(device)) ** 2).mean() / (2.0 * step["std_v"] ** 2 + 1e-8)
                kl_a = ((mean_a_new - ref[1].to(device)) ** 2).mean() / (2.0 * step["std_a"] ** 2 + 1e-8)
                loss = loss + beta * (kl_v + kl_a)
        scale = loss_scale / n_steps
        if equalizer is not None:                                                    # SAGE-GRPO gradient-norm equalizer
            t_idx = int(step.get("step_index", 0)); scale = scale * equalizer.scale(t_idx)
            live = [q for q in network.parameters() if q.requires_grad]
            before = [None if q.grad is None else q.grad.detach().clone() for q in live]
            (loss * scale).backward()
            inc = 0.0
            for q, b in zip(live, before):
                if q.grad is None: continue
                d = q.grad if b is None else (q.grad - b); inc += float((d.float() ** 2).sum())
            equalizer.update(t_idx, math.sqrt(inc) / max(scale, 1e-12))
            del before
        else:
            (loss * scale).backward()
        total_loss += float(loss.detach())
        del pred, loss, policy_loss, logp_v_new, logp_a_new, mean_v_new, mean_a_new, video, audio
    return total_loss / n_steps, ratio_log






