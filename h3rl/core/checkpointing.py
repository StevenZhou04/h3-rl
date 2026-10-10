"""Partial gradient checkpointing for musubi's H3 transformer.

musubi checkpoints every DiT block when gradient checkpointing is on (MiniMaxH3Model.forward calls
torch.utils.checkpoint.checkpoint(block, ...) per block). Recomputing a block in backward costs about one extra forward
of it; with memory to spare, running some blocks plainly (activations kept) skips that recomputation. Gradients are the
same either way: checkpointing only trades memory for compute.
"""
from __future__ import annotations
import torch
import torch.utils.checkpoint

_ORIG = torch.utils.checkpoint.checkpoint


def _checkpoint(function, *args, **kwargs):
    if getattr(function, "_h3rl_plain", False): return function(*args)   # marked block: no checkpoint, keep activations
    return _ORIG(function, *args, **kwargs)


def limit_checkpointing(transformer, plain_blocks: int) -> int:
    """Run the last `plain_blocks` DiT blocks without checkpointing (the rest stay checkpointed). Returns how many."""
    blocks = list(transformer.blocks); n = max(0, min(int(plain_blocks), len(blocks)))
    for i, b in enumerate(blocks): b._h3rl_plain = i >= len(blocks) - n
    if n and torch.utils.checkpoint.checkpoint is not _checkpoint:
        torch.utils.checkpoint.checkpoint = _checkpoint                    # musubi looks it up on the module at call time
    return n
