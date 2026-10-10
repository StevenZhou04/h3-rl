"""Reward combiners: per-video reward-model scores -> one scalar per branch (video, audio) per rollout.

A reward config names a combiner (default `weighted_z`) and its options:

    combine: weighted_z
    video: {va_ta: 1.0, soli_ta: 0.5, hps: 0.3, flow_motion: 0.5}   # term: weight (negative = lower is better)
    audio: {}                                                         # audio-branch terms (used by NFT's audio loss)
    sync:  {}                                                         # added to both branches
    gates:  {cut_free: {min: 0}}         # a rollout whose term falls outside the bound ranks last in its group (missing term: no reward)
    floors: {va_vq: 0.5}                 # penalise only drops below the term's start-of-training level
    floor_warmup: 64                     # rollouts used to fix each floor's reference level
    z_window_iters: 16                   # iterations (of each clip length) the running mean / sd forget over
    z_by_length: true                    # separate statistics per frame count (reward models score 5 s and 10 s clips on
                                         # different scales; pooled, the gap inflates the sd and shrinks the term's weight)

Add a combiner by subclassing Combiner and registering it in COMBINERS.
"""
from __future__ import annotations
import math
from h3rl.core.video_metrics import is_broken

WORST = -10.0   # broken or gated rollouts: always the group's worst


class RunningZ:
    """Per-key running mean / variance for z-scoring raw scores: exact (Welford) for the first `window` observations,
    then exponentially weighted with weight 1/window, so the statistics follow a term whose scale drifts during training
    (the policy improving, HPS's iter_step ramp) instead of averaging over the whole run."""
    def __init__(self, window: int | None = 128): self.window = window; self.n, self.mean, self.var = {}, {}, {}

    def update(self, k, x):
        n = self.n.get(k, 0) + 1; self.n[k] = n
        if n == 1: self.mean[k], self.var[k] = x, 0.0; return
        a = max(1.0 / n, 1.0 / self.window) if self.window else 1.0 / n; d = x - self.mean[k]
        self.mean[k] += a * d; self.var[k] = (1 - a) * (self.var[k] + a * d * d)   # exponentially weighted (Welford for a = 1/n)

    def sd(self, k):
        n = self.n.get(k, 0)
        return math.sqrt(max(self.var[k] * n / (n - 1) if n < (self.window or 1e18) else self.var[k], 1e-12)) if n >= 2 else None

    def z(self, k, x):
        s = self.sd(k); return 0.0 if s is None else (x - self.mean[k]) / max(s, 1e-6)

    def state(self): return dict(n=self.n, mean=self.mean, var=self.var)
    def load(self, s):
        self.n, self.mean = dict(s["n"]), dict(s["mean"])
        self.var = dict(s["var"]) if "var" in s else {k: s["m2"][k] / self.n[k] for k in s["m2"]}   # checkpoints before 2026-10-10


class Combiner:
    def __init__(self, rc: dict): self.rc = rc
    def terms(self) -> set[str]: raise NotImplementedError
    def __call__(self, axes: dict, guard: dict, has_audio: bool) -> dict: raise NotImplementedError   # -> {"video", "audio", "axes", "gated"}
    def score_batch(self, items: list, observe: list | None = None) -> list: return [self(*x) for x in items]   # items: (axes, guard, has_audio)
    def state(self) -> dict: return {}
    def load(self, s: dict): pass


class WeightedZ(Combiner):
    """Each term z-scored over a running window (per clip length), weighted average per branch; sync terms added to both branches;
    gates send a rollout to the bottom of its group; floors add a one-sided penalty below a frozen reference level."""
    def __init__(self, rc):
        super().__init__(rc); self.rz = RunningZ(None); self.ref = {}; self.seen = {}
        self.window_iters = int(rc.get("z_window_iters", 16))
        self.by_length = bool(rc.get("z_by_length", True))
        self.gates = rc.get("gates") or {}; self.floors = rc.get("floors") or {}; self.warmup = int(rc.get("floor_warmup", 64))
        self.sync_conf_min = float(rc.get("sync_conf_min", 0.3))

    def terms(self):
        return set(self.rc.get("video") or {}) | set(self.rc.get("audio") or {}) | set(self.rc.get("sync") or {}) | set(self.gates) | set(self.floors)

    def _key(self, k, guard):                                     # statistics key: the term, per clip length
        fc = guard.get("frame_count") if self.by_length else None
        return f"{k}@{int(fc)}" if fc else k

    def _branch(self, weights, axes, guard):
        tot = wsum = 0.0; n = 0
        for k, w in (weights or {}).items():
            if k in axes: tot += w * self.rz.z(self._key(k, guard), axes[k]); wsum += abs(w); n += 1
        return (tot / wsum if wsum else 0.0), n, wsum

    def _clean(self, axes):
        axes = dict(axes)
        if "desync" in axes and axes.get("sync_conf", 0.0) < self.sync_conf_min: axes.pop("desync")   # uninformative estimate
        return axes

    def _observe(self, axes, guard):
        for k, v in axes.items(): self.rz.update(self._key(k, guard), v)
        for k in self.floors:                                     # freeze each floor's reference after the warm-up
            sk = self._key(k, guard)
            if sk not in self.seen and k in self.seen and sk != k:   # state from before per-length keys: keep its reference
                self.seen[sk], self.ref[sk] = self.seen[k], self.ref[k]
            if k in axes and self.seen.get(sk, 0) < self.warmup:
                self.seen[sk] = self.seen.get(sk, 0) + 1; self.ref[sk] = self.ref.get(sk, 0.0) + (axes[k] - self.ref.get(sk, 0.0)) / self.seen[sk]

    def score_batch(self, items, observe=None):
        """Update the running statistics with the whole iteration first (`observe`: every rank's items, in rank order, so
        all ranks hold the same statistics; default `items`), then score `items` against the same stats."""
        obs = items if observe is None else observe
        if self.window_iters: self.rz.window = max(2, self.window_iters * len(obs))   # an iteration has one clip length
        for a, g, _ in obs: self._observe(self._clean(a), g)
        return [self._score(self._clean(a), g, h) for a, g, h in items]

    def __call__(self, axes, guard, has_audio):
        axes = self._clean(axes); self._observe(axes, guard); return self._score(axes, guard, has_audio)

    def _score(self, axes, guard, has_audio):
        Rv, nv, wv = self._branch(self.rc.get("video"), axes, guard)
        fl = [(w, k, self._key(k, guard)) for k, w in self.floors.items() if k in axes]
        fl = [(w, k, sk) for w, k, sk in fl if self.seen.get(sk, 0) >= self.warmup and self.rz.sd(sk)]
        if fl:
            pen = sum(w * min(0.0, (axes[k] - self.ref[sk]) / self.rz.sd(sk)) for w, k, sk in fl); wf = sum(abs(w) for w, _, _ in fl)
            Rv = (Rv * wv + pen) / (wv + wf); nv += len(fl)
        Rs, ns, _ = self._branch(self.rc.get("sync"), axes, guard)
        Ra, na, _ = self._branch(self.rc.get("audio"), axes, guard) if has_audio else (0.0, 0, 0.0)
        if ns: Rv += Rs; Ra += Rs
        gated = any(k in axes and (("min" in b and axes[k] < b["min"]) or ("max" in b and axes[k] > b["max"])) for k, b in self.gates.items())
        gate_unknown = not gated and any(k not in axes for k in self.gates)   # gate worker failed or timed out: no verdict
        broken = is_broken(guard)
        if gated or broken: Rv = WORST
        video = None if gate_unknown and not broken else (Rv if (nv or ns or gated or broken) else None)
        audio = (WORST if (gated or broken) else Ra) if (has_audio and (na or ns)) else None   # a gated/broken rollout is the worst on both branches
        # one scalar for algorithms with a single objective (GRPO): the sync terms are in both branches, count them once
        total = None if video is None and audio is None else (WORST if video == WORST else (video or 0.0) + (audio or 0.0) - (Rs if (ns and video is not None and audio is not None) else 0.0))
        return {"video": video, "audio": audio, "total": total, "axes": axes, "gated": float(gated), "broken": float(broken),
                "gate_unknown": float(gate_unknown)}

    def state(self): return {"rz": self.rz.state(), "ref": self.ref, "seen": self.seen}
    def load(self, s): self.rz.load(s["rz"]); self.ref = dict(s.get("ref", {})); self.seen = dict(s.get("seen", {}))


COMBINERS = {"weighted_z": WeightedZ}


def make_combiner(rc: dict) -> Combiner:
    name = rc.get("combine", "weighted_z")
    if name not in COMBINERS: raise ValueError(f"unknown combiner {name!r}; known: {sorted(COMBINERS)}")
    return COMBINERS[name](rc)
