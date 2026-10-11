"""Sanity reward: mean frame brightness. Not a quality reward. It is noise-free, varies between rollouts of one prompt,
and its effect is visible at a glance, so a short run on it tests that the training loop can move the sampler at all
(configs/reward/sanity_bright.yaml).

  bright = mean luma over every 4th frame, 0..1 (frames resized to 240x136)
  python -m h3rl.rewards.workers.brightness --queue Q   (CPU only)"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from h3rl.rewards.worker_base import Worker


def clip_brightness(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path); vals = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % 4 == 0: vals.append(cv2.cvtColor(cv2.resize(f, (240, 136)), cv2.COLOR_BGR2GRAY).mean() / 255.0)
        i += 1
    if len(vals) < 3: raise ValueError(f"{path}: {i} frames decoded, need at least 9")   # unreadable is an error, not dark
    return float(np.mean(vals))


class BrightnessWorker(Worker):
    name = "brightness"
    batch = 32

    def load(self):
        self.pool = ProcessPoolExecutor(int(os.environ.get("BRIGHT_PROCS", "8")))

    def score(self, requests):
        futs = [self.pool.submit(clip_brightness, r["mp4"]) for r in requests]
        out = []
        for r, fu in zip(requests, futs):
            try: out.append({"key": r["key"], "scores": {"bright": fu.result()}})
            except Exception as e: out.append({"key": r["key"], "scores": {}, "error": f"{type(e).__name__}: {e}"})
        return out


if __name__ == "__main__":
    BrightnessWorker().run()
