"""flow_motion settings (CPU, seconds):
 (a) the score: motion counts up to CAP, coherence is clamped at COH_MAX, then log1p(flow * coh**3 / FLOOR);
 (b) FLOW_CAP / FLOW_COH_MAX in the environment change the worker's settings;
 (c) the reward config's worker_env reaches every worker process Workers starts (so all nodes score alike);
 (d) behaviour on synthetic clips: real motion of any speed and orientation scores well, shake, stutter, flicker, hard
     cuts and frozen clips score low (the cases that broke version 1)."""
import os, subprocess, sys, tempfile, math
from unittest import mock
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

for k in ("FLOW_CAP", "FLOW_COH_MAX"): os.environ.pop(k, None)
import h3rl.rewards.workers.flowmotion as F
check("(a) defaults CAP 3.0, COH_MAX 1.0", (F.CAP, F.COH_MAX) == (3.0, 1.0), f"{F.CAP} {F.COH_MAX}")
check("(a) motion below the cap counts", F.flow_score(2.0, 0.9) > F.flow_score(1.5, 0.9))
check("(a) motion above the cap earns nothing more", F.flow_score(5.0, 0.9) == F.flow_score(3.0, 0.9))
check("(a) coherence above 1 earns no bonus", F.flow_score(2.0, 1.3) == F.flow_score(2.0, 1.0))
check("(a) formula", abs(F.flow_score(2.0, 0.8) - math.log1p(2.0 * 0.8 ** 3 / 0.1)) < 1e-12)

code = "import h3rl.rewards.workers.flowmotion as F; print(F.CAP, F.COH_MAX)"
out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, FLOW_CAP="4.5", FLOW_COH_MAX="1.2"), capture_output=True, text=True)
check("(b) environment overrides", out.stdout.split() == ["4.5", "1.2"], out.stdout.strip() or out.stderr[-300:])

from h3rl.rewards.procs import Workers
seen = []
with mock.patch("subprocess.Popen", side_effect=lambda cmd, **kw: seen.append((cmd, kw["env"])) or mock.MagicMock()):
    Workers(["flowmotion", "hpspp"], tempfile.mkdtemp(), [0], env={"FLOW_CAP": 3.0, "FLOW_COH_MAX": 1})
check("(c) worker_env is set for every worker", len(seen) == 2 and all(e.get("FLOW_CAP") == "3.0" and e.get("FLOW_COH_MAX") == "1" for _, e in seen))
check("(c) workers keep their own GPU assignment", [e["CUDA_VISIBLE_DEVICES"] for _, e in seen] == ["", "0"])
from omegaconf import OmegaConf
rc = OmegaConf.to_container(OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "configs", "reward", "mix_v1.yaml")))
from h3rl.rewards.combine import make_combiner
check("(c) mix_v1 sets worker_env and the combiner ignores it", rc.get("worker_env") == {"FLOW_CAP": 3.0, "FLOW_COH_MAX": 1.0}
      and make_combiner(rc).terms() == {"va_ta", "soli_ta", "hps", "flow_motion"})

import cv2, numpy as np
rng = np.random.default_rng(0)
tex = cv2.GaussianBlur(rng.normal(0, 1, (900, 3000)).astype(np.float32), (0, 0), 3)
tex = cv2.normalize(tex + 0.3 * cv2.GaussianBlur(rng.normal(0, 1, (900, 3000)).astype(np.float32), (0, 0), 12), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR); TMP = tempfile.mkdtemp()
def clip(name, frames):
    p = f"{TMP}/{name}.avi"; h, w = frames[0].shape[:2]; vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 24, (w, h))
    for f in frames: vw.write(np.ascontiguousarray(f))
    vw.release(); return F.flow_score(*F.clip_flow(p))
W, H, N = 480, 272, 64
crop = lambda x, y, w=W, h=H: tex[int(round(y)):int(round(y)) + h, int(round(x)):int(round(x)) + w]
pan = lambda v, w=W, h=H: [crop(20 + t * v * w / 480 if w >= h else 20 + t * v * h / 480, 20, w, h) for t in range(N)]
shake = lambda px: [crop(60 + rng.normal(0, px), 60 + rng.normal(0, px)) for _ in range(N)]
S = {n: clip(n, fr) for n, fr in [("frozen", [crop(20, 20)] * N), ("pan0.25", pan(0.25)), ("pan0.5", pan(0.5)), ("pan1", pan(1)), ("pan2", pan(2)), ("pan4", pan(4)), ("pan8", pan(8)), ("pan12", pan(12)),
     ("shake3", shake(3)), ("shake6", shake(6)), ("stutter", [pan(2)[(i // 3) * 3] for i in range(N)]),
     ("flicker", [np.clip(crop(20, 20).astype(np.float32) * (1 + rng.uniform(-.15, .15)), 0, 255).astype(np.uint8) for _ in range(N)]),
     ("cut", [crop(20, 20)] * (N // 2) + [crop(1500, 400)] * (N // 2)), ("vert4", pan(4, 272, 480))]}
print("   ", {k: round(v, 2) for k, v in S.items()})
check("(d) frozen scores 0", S["frozen"] < 0.05)
check("(d) more motion scores higher below the cap", S["pan0.25"] < S["pan0.5"] < S["pan1"] < S["pan2"])
check("(d) fast pans keep their score (v1 scored 8-12 px/frame like shake)", min(S["pan8"], S["pan12"]) > 0.9 * S["pan4"])
check("(d) shake scores far below real motion", max(S["shake3"], S["shake6"]) < 0.35 * S["pan2"])
check("(d) stutter scores below the same motion played smoothly", S["stutter"] < 0.75 * S["pan2"])
check("(d) flicker on a still frame is not motion", S["flicker"] < 0.15)
check("(d) a hard cut between two still shots is not motion", S["cut"] < 0.15)
check("(d) vertical clips score like landscape ones (v1 squashed them)", abs(S["vert4"] - S["pan4"]) < 0.15 * S["pan4"])
if FAILS: sys.exit(1)
print("ALL PASS")
