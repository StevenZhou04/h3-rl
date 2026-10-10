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
    # Ranks reach the collectives at different times: rollout length varies with prompt length, and a rank can wait up
    # to run.reward_timeout_s (default 30 min) for missing rewards before it proceeds with advantage 0. The timeout
    # must exceed that plus a rollout (~20 min measured), or one slow reward worker aborts the whole job.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
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


def prune_checkpoints(out_dir: Path, name: str, keep_recent: int, keep_every: int) -> None:
    """Keep the newest `keep_recent` checkpoints plus every `keep_every`-th as a coarse history and delete the rest
    (all of <name>-NNNNN.safetensors / .state.json / .train.pt), so a long run cannot fill a shared disk.
    keep_recent <= 0 keeps everything."""
    if keep_recent <= 0: return
    steps = sorted(int(m.group(1)) for p in Path(out_dir).glob(f"{name}-*.safetensors")
                   if (m := re.fullmatch(rf"{re.escape(name)}-(\d+)", p.stem)))
    for step in steps[:-keep_recent]:
        if keep_every and step % keep_every == 0: continue
        for ext in (".safetensors", ".state.json", ".train.pt"):
            try: (Path(out_dir) / f"{name}-{step:05d}{ext}").unlink()
            except FileNotFoundError: pass


def broadcast_tree(obj, rank: int, device: torch.device):
    """Rank 0's `obj` (nested dicts/lists/tuples holding tensors) on every rank, tensors on the CPU. The structure goes as
    a small pickle; each tensor goes through its own NCCL broadcast, so a multi-GB optimizer state never becomes one
    pickled byte tensor on every GPU."""
    tensors = []
    def strip(x):
        if isinstance(x, torch.Tensor): tensors.append(x); return ("__tensor__", tuple(x.shape), str(x.dtype).replace("torch.", ""))
        if isinstance(x, dict): return {k: strip(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return type(x)(strip(v) for v in x)
        return x
    box = [strip(obj) if rank == 0 else None]; dist.broadcast_object_list(box, src=0); skel = box[0]
    it = iter(tensors)
    def fill(x):
        if isinstance(x, tuple) and len(x) == 3 and x[0] == "__tensor__":
            t = next(it).to(device) if rank == 0 else torch.empty(x[1], dtype=getattr(torch, x[2]), device=device)
            dist.broadcast(t, src=0); return t.cpu()
        if isinstance(x, dict): return {k: fill(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return type(x)(fill(v) for v in x)
        return x
    out = fill(skel); torch.cuda.empty_cache(); return out

