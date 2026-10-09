"""Starting reward-worker processes for a queue (used by the launcher and by the reward service)."""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
from h3rl import paths
from h3rl.rewards.registry import WORKERS, CPU_WORKERS

REPO = Path(__file__).resolve().parent.parent.parent


def reward_python(env: str) -> str:
    if env == "main": return sys.executable
    return os.path.expanduser(os.environ.get("H3RL_REWARD_PYTHON", f"{paths.HOME}/envs/reward/bin/python"))


def env_with_repo(**kw):
    return dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}", **kw)


class Workers:
    """Reward workers on one queue; GPU workers spread round-robin over `gpus`, CPU workers get no GPU.
    `replicas` > 1 starts that many copies of every GPU worker (they share the queue), for reward-only nodes."""
    def __init__(self, names, queue: Path, gpus, replicas: int = 1):
        self.names, self.queue, self.procs = [], Path(queue), []; (self.queue / "logs").mkdir(parents=True, exist_ok=True); gi = 0
        for w in names:
            for r in range(1 if w in CPU_WORKERS else replicas):
                gpu = "" if w in CPU_WORKERS else str(gpus[gi % len(gpus)]); gi += w not in CPU_WORKERS
                tag = w if r == 0 else f"{w}.{r}"; log = open(self.queue / "logs" / f"{tag}.log", "w")
                self.procs.append(subprocess.Popen([reward_python(WORKERS[w]["env"]), "-m", f"h3rl.rewards.workers.{w}", "--queue", str(self.queue)],
                                                   env=env_with_repo(CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                                   start_new_session=True))   # own process group: stop() also ends pool children
                self.names.append(tag)

    def wait_loaded(self, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            dead = [n for n, p in zip(self.names, self.procs) if p.poll() is not None]
            if dead: raise RuntimeError(f"reward workers exited: {dead} (see {self.queue}/logs)")
            if all("loaded" in (self.queue / "logs" / f"{n}.log").read_text() for n in self.names): return
            time.sleep(10)
        raise TimeoutError("reward workers did not load in time")

    def stop(self, grace: float = 10.0):
        """Terminate every worker together with its children (CPU workers run process pools), then kill stragglers."""
        import signal
        for p in self.procs:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        t0 = time.time()
        while time.time() - t0 < grace and any(p.poll() is None for p in self.procs): time.sleep(0.5)
        for p in self.procs:
            try: os.killpg(p.pid, signal.SIGKILL)                 # the group may outlive its leader
            except ProcessLookupError: pass
