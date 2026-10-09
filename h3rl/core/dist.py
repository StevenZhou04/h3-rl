"""Process-group setup and the hand-rolled DDP helpers used by every training loop."""
import os, re
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist


def dist_setup(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1, torch.device(device_arg)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # Ranks reach the gradient all-reduce at different times: rollout length varies with prompt
    # length and the judge's network latency, so skew of several minutes is normal. The default
    # NCCL collective timeout would abort a healthy run; one hour is longer than any plausible
    # single step (~20 min measured) and still bounded.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    return dist.get_rank(), world_size, torch.device(f"cuda:{local_rank}")


def average_gradients(params, world_size: int) -> None:
    """All-reduce-mean the LoRA gradients. Called after every trajectory's backward has
    accumulated into .grad and before optimizer.step(), which makes the effective batch
    world_size x prompts_per_step x group_size rollouts with one shared update.

    Done by hand rather than with DistributedDataParallel because the LoRA modules are not
    submodules of `transformer` -- they are attached by monkey-patching each Linear's forward
    -- so DDP has no module tree to hook, and wrapping the transformer would try to sync the
    frozen 21 GB base every step."""
    for p in params:
        if p.grad is None:
            # A parameter with no gradient on THIS rank still has to participate: all_reduce is
            # collective, and a rank that skips one deadlocks the others.
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def average_metrics(metrics: dict, world_size: int, device: torch.device) -> dict:
    """Mean every scalar in the metrics dict across ranks, so the logged reward is the real
    batch reward and not just rank 0's slice of it.

    Ranks may carry DIFFERENT key sets (e.g. geometry axes only when a rank's batch held a
    camera prompt, DeSync only when sync confidence passed the gate). A tensor all-reduce over
    the flattened dict therefore had mismatched sizes across ranks and hung until the NCCL
    watchdog aborted the run (NFT head-to-head, 2026-09-28). all_gather_object tolerates
    unequal payloads; each key is averaged over the ranks that reported it."""
    if world_size == 1:
        return metrics
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    flat[(key, sub)] = float(sub_value)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, flat)
    out = {k: v for k, v in metrics.items()}
    for k in sorted({kk for g in gathered for kk in g}, key=str):
        vals = [g[k] for g in gathered if k in g]
        mean = sum(vals) / len(vals)
        if isinstance(k, tuple):
            out.setdefault(k[0], {}); out[k[0]][k[1]] = mean
        else:
            out[k] = mean
    return out


def prune_checkpoints(out_dir: Path, keep_recent: int, keep_every: int) -> None:
    """Keep the newest `keep_recent` checkpoints plus every `keep_every`-th as a coarse history,
    and delete the rest, so an unattended run saving every few steps cannot fill a shared disk."""
    # Match ONLY the numbered checkpoints. The emergency saves -- h3-grpo-000030-budget,
    # -fatal, -degenerate -- also start with digits, so the old "[0-9]*" glob swept them in and
    # int(stem.rsplit("-",1)[1]) raised ValueError on "budget". The try below only wraps
    # unlink(), so that crashed the trainer at save time; a run that hit its spend ceiling and
    # was then resumed would die on its first checkpoint.
    ckpts = sorted(p for p in out_dir.glob("h3-grpo-*.safetensors")
                   if re.fullmatch(r"h3-grpo-\d+", p.stem))
    if len(ckpts) <= keep_recent:
        return
    for path in ckpts[:-keep_recent]:
        step = int(path.stem.rsplit("-", 1)[1])
        if keep_every and step % keep_every == 0:
            continue
        try:
            path.unlink()
        except OSError:
            pass


