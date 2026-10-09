"""Interface between the shared trainer and an RL algorithm.

The trainer owns the model, prompts, clip-length buckets, decoding, rewards, metrics and checkpoints. Per iteration
and per prompt it calls `rollout(ctx, prompt, seed)`, decodes each returned sample's final latents, scores them, and
then calls `update(samples)` once with every sample of the iteration (each carrying `R`, the combined reward, and
`group`, its prompt index). To add an algorithm: subclass Algorithm, register it, add configs/algo/<name>.yaml.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import random
import torch

ALGORITHMS: dict[str, type] = {}


def register(name: str):
    def deco(cls): ALGORITHMS[name] = cls; cls.name = name; return cls
    return deco


@dataclass
class TrainContext:
    """What the trainer hands an algorithm."""
    models: dict
    transformer: torch.nn.Module
    network: torch.nn.Module
    params: list
    schedule: object
    device: torch.device
    rank: int
    world: int
    rng: random.Random
    infer_steps: int
    canvas: dict = field(default_factory=dict)      # current {"frames", "height", "width"}, set by the trainer each iteration


class Algorithm:
    name = "base"

    def __init__(self, acfg: dict, T: TrainContext):
        self.a = acfg; self.T = T

    def group_size(self) -> int: return int(self.a.get("group_size", 8))
    def prompts_per_step(self) -> int: return int(self.a.get("prompts_per_step", 1))

    def rollout(self, ctx: dict, prompt: dict, seed: int) -> list[dict]:
        """Sample one group for one prompt. Each returned dict needs `video` and `audio` (final latents); anything else
        the algorithm wants back in update() can be added."""
        raise NotImplementedError

    def update(self, samples: list[dict], iteration: int) -> dict:
        """One policy update from the iteration's scored samples; returns float metrics."""
        raise NotImplementedError

    def state(self) -> dict: return {}
    def load(self, s: dict): pass
