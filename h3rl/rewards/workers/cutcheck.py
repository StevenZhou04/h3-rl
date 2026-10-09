"""Hard-cut penalty for prompts that ask for one continuous shot (一镜到底 / no cuts). Targets a common failure on complex
camera prompts: requested camera moves replaced by static shots joined by cuts.

  cut_free = -min(n_cuts, 3) if the prompt asks for a continuous shot, else 0 (constant: no ranking signal)
  a cut    = consecutive-frame pair whose mean luma difference is > 0.06 and > 4x its local median (+-6 pairs)
             AND whose HSV-histogram correlation is < 0.95
Validation (240x136): 0/200 real clips flagged, 0/40 shaken, 0/40 stuttered, 3/40 at 4x speed;
caught 39/40 A|B splices and 30-32/40 same-scene cuts to a 2x close-up. Time jumps inside one shot are not caught.
  python -m h3rl.rewards.workers.cutcheck --queue Q   (CPU only)"""
from __future__ import annotations
import os, re, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from h3rl.rewards.worker_base import Worker

CONTINUOUS = re.compile(r"(一镜到底|无剪辑|没有剪辑|不剪辑|no cuts?\b|without (a )?cuts?|one continuous|single continuous|continuous (unbroken )?shot)", re.I)


def clip_cuts(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(path); g, h = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (240, 136)); g.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
        x = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(x, x); h.append(x)
    d = np.array([np.abs(g[i + 1] - g[i]).mean() / 255 for i in range(len(g) - 1)])
    c = np.array([cv2.compareHist(h[i], h[i + 1], cv2.HISTCMP_CORREL) for i in range(len(h) - 1)])
    n = 0
    for i in range(len(d)):
        nb = np.concatenate([d[max(0, i - 6):i], d[i + 1:i + 7]])
        if d[i] > 0.06 and d[i] > 4.0 * ((np.median(nb) if len(nb) else 0) + 0.005) and c[i] < 0.95: n += 1
    return n


class CutCheckWorker(Worker):
    name = "cutcheck"
    batch = 32

    def load(self):
        self.pool = ProcessPoolExecutor(int(os.environ.get("CUT_PROCS", "8")))

    def score(self, requests):
        cuts = list(self.pool.map(clip_cuts, [r["mp4"] for r in requests]))
        out = []
        for r, n in zip(requests, cuts):
            cont = bool(CONTINUOUS.search(r.get("prompt") or ""))
            out.append({"key": r["key"], "scores": {"cut_free": -float(min(n, 3)) if cont else 0.0, "n_cuts": float(n), "cut_continuous_prompt": float(cont)}})
        return out


if __name__ == "__main__":
    CutCheckWorker().run()
