"""Optical-flow motion reward: real pixel movement (Farneback flow), not raw frame difference, so shimmer/grain
does not count. Added for nft_flow_mix to counter the drift toward near-static video seen in the NFT runs.

  flow_motion = log1p(min(flow, CAP) * coherence**3 / FLOOR)
  flow      = mean flow magnitude over frame pairs (t, t+2), px at 480 px width
  coherence = median over 8-frame windows of |F(t,t+8)| / sum of the four |F(t+k,t+k+2)|  (~0.94 for real motion,
              0.14 for added camera shake, 0.52 for temporal noise, 0.70 for stuttered frames)
Capped so motion beyond CAP earns nothing; the coherence factor stops shake/stutter/noise from counting as motion
(audit: with the uncorrected term the combined reward preferred shaken clips 20/24 and stuttered clips 18/24 over the clean ones).
Prompts that ask for a fully static shot (static scene AND static camera) get a constant score, which carries no
ranking signal inside a group, so stillness is never penalised where the prompt wants it.
  python -m h3rl.rewards.workers.flowmotion --queue Q   (CPU only)"""
from __future__ import annotations
import os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from h3rl.rewards.worker_base import Worker

CAP = 1.5      # px/frame @480w; base HyperFlow held-out mean is 0.655, held-out SpatialVID ~1.0
FLOOR = 0.1
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


def clip_flow(path: str) -> tuple[float, float]:
    """(mean short-range flow, median coherence) of one clip."""
    import cv2
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(cv2.resize(f, (480, 272)), cv2.COLOR_BGR2GRAY))
    flow = lambda a, b: np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2).mean()
    short, coh = [], []
    for t in range(0, len(fr) - 8, 8):
        s = [flow(fr[t + k], fr[t + k + 2]) for k in (0, 2, 4, 6)]
        short += s; coh.append(flow(fr[t], fr[t + 8]) / (sum(s) + 1e-6))
    return (float(np.mean(short)), float(np.median(coh))) if short else (0.0, 0.0)


class FlowMotionWorker(Worker):
    name = "flowmotion"
    batch = 32

    def load(self):
        self.pool = ProcessPoolExecutor(int(os.environ.get("FLOW_PROCS", "16")))

    def score(self, requests):
        stats = list(self.pool.map(clip_flow, [r["mp4"] for r in requests]))
        out = []
        for r, (fl, coh) in zip(requests, stats):
            static = wants_static(r.get("prompt") or "")
            s = STATIC_SCORE if static else float(np.log1p(min(fl, CAP) * coh ** 3 / FLOOR))
            out.append({"key": r["key"], "scores": {"flow_motion": s, "flow_raw": fl, "flow_coherence": coh, "flow_static_prompt": float(static)}})
        return out


if __name__ == "__main__":
    FlowMotionWorker().run()
