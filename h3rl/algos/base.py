"""Interface between the shared trainer and an RL algorithm.

The trainer owns the model, prompts, clip-length buckets, decoding, rewards, metrics and checkpoints. Per iteration
and per prompt it calls `rollout(ctx, prompt, seed, members)`, decodes each returned sample's final latents, scores them, and
then calls `update(samples)` once with every sample of the iteration (each carrying `R`, the combined reward, `group`,
its prompt index on this rank, `gid`, the group's id across ranks, and `member`, its index in the group). To add an algorithm: subclass Algorithm, register it, add configs/algo/<name>.yaml.
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
    def ranks_per_group(self) -> int:
        """GPUs that share each prompt group (algo.ranks_per_group, default 1): each makes group_size / R of its rollouts, so
        more GPUs shorten an iteration instead of only enlarging the batch. Algorithms that support R > 1 override
        supports_group_split()."""
        return int(self.a.get("ranks_per_group", 1))
    def supports_group_split(self) -> bool: return False

    def rollout(self, ctx: dict, prompt: dict, seed: int, members=None):   # -> iterable of sample dicts (list or generator)
        """Sample one group (or, with ranks_per_group > 1, this rank's `members`: indices into the group, each with its own
        seed) for one prompt. Each returned dict needs `video` and `audio` (final latents); anything else the algorithm
        wants back in update() can be added."""
        raise NotImplementedError

    def update(self, samples: list[dict], iteration: int) -> dict:
        """One policy update from the iteration's scored samples; returns float metrics."""
        raise NotImplementedError

    def state(self) -> dict: return {}          # small JSON-able state (counters), saved next to every checkpoint
    def load(self, s: dict): pass
    def state_tensors(self) -> dict: return {}  # tensors and optimizer state for an exact resume (torch.save)
    def load_tensors(self, s: dict): pass
