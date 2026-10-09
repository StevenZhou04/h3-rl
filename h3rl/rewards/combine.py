"""Reward combiners: per-video reward-model scores -> one scalar per branch (video, audio) per rollout.

A reward config names a combiner (default `weighted_z`) and its options:

    combine: weighted_z
    video: {va_ta: 1.0, soli_ta: 0.5, hps: 0.3, flow_motion: 0.5}   # term: weight (negative = lower is better)
    audio: {}                                                         # audio-branch terms (used by NFT's audio loss)
    sync:  {}                                                         # added to both branches
    gates:  {cut_free: {min: 0}}         # a rollout whose term falls outside the bound ranks last in its group
    floors: {va_vq: 0.5}                 # penalise only drops below the term's start-of-training level
    floor_warmup: 64                     # rollouts used to fix each floor's reference level

Add a combiner by subclassing Combiner and registering it in COMBINERS.
"""
from __future__ import annotations
import math
from h3rl.core.video_metrics import GUARDRAIL_KEYS

WORST = -10.0   # broken or gated rollouts: always the group's worst


class RunningZ:
    """Per-term running mean/std (Welford) over everything this rank has scored, for z-scoring raw scores."""
    def __init__(self): self.n, self.mean, self.m2 = {}, {}, {}

    def update(self, k, x):
        n = self.n.get(k, 0) + 1; mu = self.mean.get(k, 0.0); d = x - mu; mu += d / n
        self.m2[k] = self.m2.get(k, 0.0) + d * (x - mu); self.n[k] = n; self.mean[k] = mu

    def sd(self, k): n = self.n.get(k, 0); return math.sqrt(max(self.m2[k] / (n - 1), 1e-12)) if n >= 2 else None

    def z(self, k, x):
        s = self.sd(k); return 0.0 if s is None else (x - self.mean[k]) / max(s, 1e-6)

    def state(self): return dict(n=self.n, mean=self.mean, m2=self.m2)
    def load(self, s): self.n, self.mean, self.m2 = dict(s["n"]), dict(s["mean"]), dict(s["m2"])


class Combiner:
    def __init__(self, rc: dict): self.rc = rc
    def terms(self) -> set[str]: raise NotImplementedError
    def __call__(self, axes: dict, guard: dict, has_audio: bool) -> dict: raise NotImplementedError   # -> {"video", "audio", "axes", "gated"}
    def state(self) -> dict: return {}
    def load(self, s: dict): pass


class WeightedZ(Combiner):
    """Each term z-scored over a running window, weighted average per branch; sync terms added to both branches;
    gates send a rollout to the bottom of its group; floors add a one-sided penalty below a frozen reference level."""
    def __init__(self, rc):
        super().__init__(rc); self.rz = RunningZ(); self.ref = {}; self.seen = {}
        self.gates = rc.get("gates") or {}; self.floors = rc.get("floors") or {}; self.warmup = int(rc.get("floor_warmup", 64))
        self.sync_conf_min = float(rc.get("sync_conf_min", 0.3))

    def terms(self):
        return set(self.rc.get("video") or {}) | set(self.rc.get("audio") or {}) | set(self.rc.get("sync") or {}) | set(self.gates) | set(self.floors)

    def _branch(self, weights, axes):
        tot = wsum = 0.0; n = 0
        for k, w in (weights or {}).items():
            if k in axes: tot += w * self.rz.z(k, axes[k]); wsum += abs(w); n += 1
        return (tot / wsum if wsum else 0.0), n, wsum

    def __call__(self, axes, guard, has_audio):
        axes = dict(axes)
        if "desync" in axes and axes.get("sync_conf", 0.0) < self.sync_conf_min: axes.pop("desync")   # uninformative estimate
        for k, v in axes.items(): self.rz.update(k, v)
        for k in self.floors:                                     # freeze each floor's reference after the warm-up
            if k in axes and self.seen.get(k, 0) < self.warmup:
                self.seen[k] = self.seen.get(k, 0) + 1; self.ref[k] = self.ref.get(k, 0.0) + (axes[k] - self.ref.get(k, 0.0)) / self.seen[k]
        Rv, nv, wv = self._branch(self.rc.get("video"), axes)
        fl = [(w, k) for k, w in self.floors.items() if k in axes and self.seen.get(k, 0) >= self.warmup and self.rz.sd(k)]
        if fl:
            pen = sum(w * min(0.0, (axes[k] - self.ref[k]) / self.rz.sd(k)) for w, k in fl); wf = sum(abs(w) for w, _ in fl)
            Rv = (Rv * wv + pen) / (wv + wf); nv += len(fl)
        Rs, ns, _ = self._branch(self.rc.get("sync"), axes)
        Ra, na, _ = self._branch(self.rc.get("audio"), axes) if has_audio else (0.0, 0, 0.0)
        if ns: Rv += Rs; Ra += Rs
        gated = any(k in axes and (("min" in b and axes[k] < b["min"]) or ("max" in b and axes[k] > b["max"])) for k, b in self.gates.items())
        broken = not all(guard.get(k, 1.0) >= 0.5 for k in GUARDRAIL_KEYS)
        if gated or broken: Rv = WORST
        return {"video": Rv if (nv or ns or gated or broken) else None, "audio": Ra if (has_audio and (na or ns)) else None,
                "axes": axes, "gated": float(gated), "broken": float(broken)}

    def state(self): return {"rz": self.rz.state(), "ref": self.ref, "seen": self.seen}
    def load(self, s): self.rz.load(s["rz"]); self.ref = dict(s.get("ref", {})); self.seen = dict(s.get("seen", {}))


COMBINERS = {"weighted_z": WeightedZ}


def make_combiner(rc: dict) -> Combiner:
    name = rc.get("combine", "weighted_z")
    if name not in COMBINERS: raise ValueError(f"unknown combiner {name!r}; known: {sorted(COMBINERS)}")
    return COMBINERS[name](rc)
