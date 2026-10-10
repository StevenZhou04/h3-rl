"""Cheap, objective, pixel-space video metrics computed locally on a decoded rollout.

Why these exist alongside the Claude judge: a VLM sees six sampled frames, so it structurally
cannot see a one-frame black flash, a strobing luminance swing, or a contrast collapse between
the frames it was shown -- and those are exactly the failure modes an early-RL video model
produces (the camera-motion LoRA run earlier in this project produced intermittent black frames
that no frame-sampled judge would reliably catch). They are computed on EVERY decoded frame, cost
nothing, and cannot be argued with, which also makes them harder to reward-hack than a rubric.

Each metric named in GUARDRAIL_KEYS returns a number in [0, 1] where 1.0 means "clean". is_broken() turns them into a
verdict that ranks the rollout last in its group (rewards.combine) rather than averaging them with the judge axes: a video
with black frames is broken regardless of how well it followed the prompt, and averaging would let a high
prompt_following score hide it.

Deliberately NOT a guardrail: "the video has motion". A prompt in this set may legitimately ask
the camera and subject to hold perfectly still, so penalizing stillness unconditionally would
teach the model to ignore those prompts. Motion statistics are therefore reported for logging
and left to the judge's motion_coherence axis, which reads the prompt.
"""
from __future__ import annotations

import numpy as np

# Thresholds are on 0-255 luminance. They are deliberately loose: these flag broken output,
# not merely dark or low-contrast artistic output, so a moody night scene must pass cleanly.
BLACK_FRAME_LUMA = 10.0       # mean luma below this is a dead frame, not a dark scene
LOW_CONTRAST_STD = 3.0        # per-frame luma std below this is grey mush (diagnostic only --
                              # see GUARDRAIL_KEYS for why this one does not gate reward)
FLICKER_LUMA_JUMP = 28.0      # frame-to-frame mean-luma jump above this reads as a strobe


def _luma(frames: np.ndarray) -> np.ndarray:
    """[F,H,W,3] uint8 RGB -> [F,H,W] float32 luma (Rec. 601)."""
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def compute_video_metrics(frames: np.ndarray) -> dict[str, float]:
    """frames: [F,H,W,3] uint8, the full decoded rollout (not the judge's 6-frame sample)."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [F,H,W,3] uint8 frames, got {frames.shape}")

    luma = _luma(frames)
    frame_mean = luma.mean(axis=(1, 2))
    frame_std = luma.std(axis=(1, 2))
    n = float(len(frame_mean))

    black_fraction = float((frame_mean < BLACK_FRAME_LUMA).mean())
    low_contrast_fraction = float((frame_std < LOW_CONTRAST_STD).mean())

    if len(frame_mean) > 1:
        luma_jumps = np.abs(np.diff(frame_mean))
        flicker_fraction = float((luma_jumps > FLICKER_LUMA_JUMP).mean())
        # Mean absolute pixel difference between consecutive frames: the crudest possible
        # motion-energy proxy. Reported, never penalized -- see the module docstring.
        diffs = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
        motion_mean = float(diffs.mean())
        # Coefficient of variation of that motion: a smooth move has near-constant energy,
        # a teleport/stutter has spikes. Also report-only, since a prompted accelerating
        # move is legitimately non-uniform.
        motion_cv = float(diffs.std() / (diffs.mean() + 1e-6))
    else:
        flicker_fraction = motion_mean = motion_cv = 0.0

    return {
        # guardrails, 1.0 = clean
        "black_frame_free": 1.0 - black_fraction,
        "contrast_ok": 1.0 - low_contrast_fraction,
        "flicker_free": 1.0 - flicker_fraction,
        # diagnostics, logged only
        "luma_mean": float(frame_mean.mean()),
        "motion_mean": motion_mean,
        "motion_cv": motion_cv,
        "frame_count": n,
    }


# contrast_ok is deliberately NOT a reward-gating guardrail. Low per-frame contrast is
# ambiguous: fog, snow, night interiors, and a subject on a plain white background are all
# legitimately low-contrast, so gating reward on it would crush real prompts in this set (a
# synthetic flat-color test clip tripped it at std=2.4 during bring-up). Black frames and
# strobing have no legitimate reading, so those two gate; contrast is logged for diagnosis.
GUARDRAIL_KEYS = ("black_frame_free", "flicker_free")

# Measured on 6.8k HyperFlow rollouts (2026-10-10): 2% have at least one dead frame (mostly a run of them: a fade or cut
# to black); single luma jumps are common (27% of clips, cuts and lighting changes) but never exceed 5% of transitions.
MAX_BLACK_FRAMES = 0          # any dead frame is broken: a one-frame black flash is the failure this exists to catch
MAX_FLICKER_FRACTION = 0.05   # strobing = luma jumps on more than 5% of transitions; isolated jumps are left to the judges


def is_broken(metrics: dict) -> bool:
    """True if the rollout has a dead (black) frame or strobes. `metrics` is compute_video_metrics' output."""
    frac = 1.0 - metrics.get("black_frame_free", 1.0); n = int(metrics.get("frame_count", 0) or 0)
    black = round(frac * n) if n else (1 if frac > 1e-9 else 0)        # without a frame count: any black fraction counts
    return black > MAX_BLACK_FRAMES or (1.0 - metrics.get("flicker_free", 1.0)) > MAX_FLICKER_FRACTION + 1e-9

