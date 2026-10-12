"""Optical-flow motion reward: real pixel movement (Farneback flow), not raw frame difference, so shimmer/grain
does not count. Added for nft_flow_mix to counter the drift toward near-static video seen in the NFT runs.

  flow_motion = log1p(min(flow, CAP) * min(coherence, COH_MAX)**3 * consistency**2 / FLOOR)
  flow        = mean flow magnitude over frame pairs (t, t+2), px with the long side at 480 px (aspect kept)
  coherence   = per 8-frame window, grid points are followed through the four 2-frame flows: |net displacement| /
                path length (~1 for real motion at any speed the 2-frame flow tracks, ~0.25 for camera shake);
                median over windows
  consistency = per window, 1 - mean |s_k - s_k+1| / (s_k + s_k+1 + 0.3) over the four 2-frame speeds s_k (~0.97 for
                real motion, ~0.4 for stutter, i.e. repeated frames)
Frames are luma-normalised first (flicker and fades are not motion), and windows whose flow spikes above 3x the median
(hard cuts) are left out. Version 2 (2026-10-12) replaced a single 8-frame flow in the coherence, which the flow could
not track for fast motion (a smooth 12 px/frame pan scored like shake), squashed vertical clips to landscape, and
rewarded stutter; tests/test_flow_motion.py builds those cases.
CAP and COH_MAX can be set per run in the reward config: worker_env: {FLOW_CAP: 3.0, FLOW_COH_MAX: 1.0}.
Capped so motion beyond CAP earns nothing; the coherence and consistency factors stop shake/stutter/noise from counting
as motion (audit: with no coherence factor the combined reward preferred shaken clips 20/24 and stuttered clips 18/24).
Prompts that ask for a fully static shot (static scene AND static camera) get a constant score, which carries no
ranking signal inside a group, so stillness is never penalised where the prompt wants it.
  python -m h3rl.rewards.workers.flowmotion --queue Q   (CPU only)"""
from __future__ import annotations
import os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from h3rl.rewards.worker_base import Worker

# px/frame @480w. Was 1.5 (base HyperFlow held-out mean 0.655 on the old prompts), but the camera-move prompts of
# example_pool.jsonl average ~1.7 for the base model and half of the NFT rollouts sat above 1.5, so within a group the
# term ignored motion and only ranked coherence (and slightly penalised motion: within-group r = -0.18). At 3.0, 16% of
# rollouts saturate and the within-group correlation with motion turns positive.
CAP = float(os.environ.get("FLOW_CAP", "3.0"))
# coherence above 1 is a Farneback artefact (the 8-frame flow cannot exceed the sum of its 2-frame parts for real motion;
# 7% of NFT rollouts), and coherence**3 turned it into up to a 2.7x bonus
COH_MAX = float(os.environ.get("FLOW_COH_MAX", "1.0"))
FLOOR = 0.1
SPEED_EPS = 0.3   # px: speed differences below this are noise, so slow real motion is not called inconsistent
VERSION = 2
STATIC_SCORE = float(np.log1p(CAP / FLOOR))   # constant for fully static prompts (any constant works within a group)

SCENE_STATIC = re.compile(r"(nothing (in the scene )?moves|(remain|remains|stays|stay) (completely |perfectly )?motionless|"
                          r"completely motionless|entirely still scene|静止不动|纹丝不动)", re.I)
CAMERA_STATIC = re.compile(r"(locked[- ]off|on a tripod|static camera|fixed camera|camera (holds|remains|stays|is) "
                           r"(completely |perfectly )?(still|fixed|static|stationary))", re.I)
CAMERA_MOVES = re.compile(r"\b(then|pans?|panning|push(es)?[- ]in|pull(s)?[- ]back|dolly|orbit|crane|tracks?|tracking|"
                          r"zoom(s)?|tilt(s)?|follows?|sweeps?|glides?|rises?|descends?)\b|推|拉|摇|移|跟拍|环绕|升|降", re.I)


def wants_static(prompt: str) -> bool:
    """Static scene AND static camera, with no later camera move ('at first the camera holds still, then ...')."""
    if not SCENE_STATIC.search(prompt): return False
    m = CAMERA_STATIC.search(prompt)
    if not m: return False
    return not CAMERA_MOVES.search(prompt[m.end():])


def flow_score(flow: float, coherence: float, consistency: float = 1.0) -> float:
    return float(np.log1p(min(flow, CAP) * min(coherence, COH_MAX) ** 3 * min(max(consistency, 0.0), 1.0) ** 2 / FLOOR))


def read_frames(path: str) -> list:
    """Grayscale frames, long side 480 px (aspect kept), each normalised to zero mean / unit spread of luma."""
    import cv2
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); h, w = g.shape
        size = (480, max(2, round(480 * h / w / 2) * 2)) if w >= h else (max(2, round(480 * w / h / 2) * 2), 480)
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        fr.append(np.clip((g - g.mean()) / (g.std() + 1e-3) * 40 + 128, 0, 255).astype(np.uint8))
    cap.release(); return fr


def clip_flow(path: str, step: int = 8) -> tuple[float, float, float]:
    """(flow, coherence, consistency) of one clip; see the module docstring."""
    import cv2
    fr = read_frames(path)
    if len(fr) < 9: raise ValueError(f"{path}: {len(fr)} frames decoded, need at least 9")   # an unreadable clip is an error, not a static one
    h, w = fr[0].shape
    gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step].astype(np.float32)
    wins = []                                                      # (flow, coherence, consistency) per 8-frame window
    for t in range(0, len(fr) - 8, 8):
        x, y = gx.copy(), gy.copy(); path_len = np.zeros_like(x); inside = np.ones(x.shape, bool); speeds = []
        for k in (0, 2, 4, 6):
            F = cv2.calcOpticalFlowFarneback(fr[t + k], fr[t + k + 2], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            speeds.append(float(np.linalg.norm(F, axis=2).mean()))
            dx = cv2.remap(F[..., 0], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dy = cv2.remap(F[..., 1], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            x, y = x + dx, y + dy; path_len += np.hypot(dx, dy)
            inside &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        net = np.hypot(x - gx, y - gy)
        coh = float(net[inside].sum() / (path_len[inside].sum() + 1e-6)) if inside.any() else 0.0
        con = 1.0 - float(np.mean([abs(a - b) / (a + b + SPEED_EPS) for a, b in zip(speeds, speeds[1:])]))
        wins.append((float(np.median(speeds)), coh, con))
    fl, coh, con = (np.array(v) for v in zip(*wins))
    keep = fl <= 3 * np.median(fl) + 0.5                         # a hard cut makes its window's flow spike
    return float(fl[keep].mean()), float(np.median(coh[keep])), float(np.median(con[keep]))


class FlowMotionWorker(Worker):
    name = "flowmotion"
    batch = 32

    def load(self):
        self.pool = ProcessPoolExecutor(int(os.environ.get("FLOW_PROCS", "16")))
        print(f"flow_motion v{VERSION}: CAP {CAP} COH_MAX {COH_MAX} FLOOR {FLOOR}", flush=True)

    def score(self, requests):
        futs = [self.pool.submit(clip_flow, r["mp4"]) for r in requests]
        out = []
        for r, fu in zip(requests, futs):
            try: fl, coh, con = fu.result()
            except Exception as e: out.append({"key": r["key"], "scores": {}, "error": f"{type(e).__name__}: {e}"}); continue
            static = wants_static(r.get("prompt") or "")
            s = STATIC_SCORE if static else flow_score(fl, coh, con)
            out.append({"key": r["key"], "scores": {"flow_motion": s, "flow_raw": fl, "flow_coherence": coh, "flow_consistency": con,
                                                    "flow_static_prompt": float(static)}})
        return out


if __name__ == "__main__":
    FlowMotionWorker().run()
