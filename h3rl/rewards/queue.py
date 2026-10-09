"""File-queue between the RL trainer and the reward workers.

Layout: <queue>/in/<worker>/<key>.json (request) -> <queue>/out/<worker>/<key>.json (result).
A request is {"key", "mp4", "prompt", "meta": {...}}; a result is {"key", "scores": {axis: float}, "worker",
"error": str|null}. Workers live in their own venvs/GPUs; the trainer only touches files. Requests are
written atomically (tmp + rename) so a worker never reads a half-written JSON."""
from __future__ import annotations
import json, os, time, glob

WORKERS = ("hpspp", "unifiedreward", "videoalign", "audiobox", "clap", "desync", "geometry")


def _atomic_write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def submit(queue: str, key: str, mp4: str, prompt: str, workers, meta: dict | None = None) -> None:
    for w in workers:
        os.makedirs(f"{queue}/in/{w}", exist_ok=True)
        _atomic_write(f"{queue}/in/{w}/{key}.json", {"key": key, "mp4": mp4, "prompt": prompt, "meta": meta or {}})


def collect(queue: str, keys, workers, timeout_s: float = 3600, poll_s: float = 2.0) -> dict:
    """Returns {key: {worker: result}}; missing results after timeout are reported as errors."""
    t0 = time.time(); out = {k: {} for k in keys}
    pending = {(k, w) for k in keys for w in workers}
    while pending and time.time() - t0 < timeout_s:
        for k, w in list(pending):
            p = f"{queue}/out/{w}/{k}.json"
            if os.path.exists(p):
                try:
                    out[k][w] = json.load(open(p)); pending.discard((k, w))
                except json.JSONDecodeError:
                    pass
        if pending: time.sleep(poll_s)
    for k, w in pending:
        out[k][w] = {"key": k, "worker": w, "scores": {}, "error": "timeout"}
    return out


def pending_requests(queue: str, worker: str, limit: int):
    """Claims up to `limit` requests by renaming them (atomic), so replicas sharing a queue never score the same one."""
    files, reqs = [], []
    for f in sorted(glob.glob(f"{queue}/in/{worker}/*.json")):
        if len(files) >= limit: break
        claimed = f"{f[:-5]}.{os.getpid()}.claimed"
        try: os.rename(f, claimed)
        except FileNotFoundError: continue                     # another replica took it
        try: reqs.append(json.load(open(claimed))); files.append(claimed)
        except json.JSONDecodeError: os.remove(claimed)
    return files, reqs


def write_results(queue: str, worker: str, results: list[dict], request_files: list[str]) -> None:
    os.makedirs(f"{queue}/out/{worker}", exist_ok=True)
    for r in results:
        r["worker"] = worker; r.setdefault("error", None)
        _atomic_write(f"{queue}/out/{worker}/{r['key']}.json", r)
    for f in request_files:
        try: os.remove(f)
        except FileNotFoundError: pass


# ---- group requests: one request per group of rollouts (pairwise / listwise judges)
def submit_group(queue: str, key: str, members: list[dict], prompt: str, worker: str, meta: dict | None = None) -> None:
    """members: [{"key", "mp4"}, ...] -- all rollouts of one prompt. The worker answers with per-member scores."""
    os.makedirs(f"{queue}/in/{worker}", exist_ok=True)
    _atomic_write(f"{queue}/in/{worker}/{key}.json", {"key": key, "members": members, "prompt": prompt, "meta": meta or {}})
