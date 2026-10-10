"""flow_motion settings (CPU, seconds):
 (a) the score: motion counts up to CAP, coherence is clamped at COH_MAX, then log1p(flow * coh**3 / FLOOR);
 (b) FLOW_CAP / FLOW_COH_MAX in the environment change the worker's settings;
 (c) the reward config's worker_env reaches every worker process Workers starts (so all nodes score alike)."""
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
if FAILS: sys.exit(1)
print("ALL PASS")
