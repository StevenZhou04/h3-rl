"""Prompt-pool helpers shared by the launcher (encoding), the trainer and eval: per-length canvas, length eligibility,
and the text-cache key."""
from __future__ import annotations
import functools, hashlib, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def canvas_for(c: dict, frames: int) -> tuple[int, int]:
    """(height, width) for a clip length: data.frame_sizes[frames], else data.size."""
    d = c["data"]; fs = {int(k): v for k, v in (d.get("frame_sizes") or {}).items()}
    return tuple(fs.get(frames, d["size"]))


def eligible(pool, frames):
    return [r for r in pool if r.get("min_frames", 0) <= frames <= r.get("max_frames", 10 ** 9)]


@functools.lru_cache(maxsize=4096)
def _file_digest(path: str, size: int, mtime: float) -> str:
    with open(path, "rb") as f: return hashlib.sha1(f.read()).hexdigest()


def text_key(row: dict, height: int, width: int) -> str:
    """Cache name of a prompt's text encoding: the pid plus a fingerprint of everything the encoding depends on (task,
    prompt text; for fl2va the image bytes and the canvas the image is cropped to), so editing a prompt or reusing a pid
    in another pool can never pick up a stale encoding."""
    task = row.get("task", "t2va"); parts = [task, row["prompt"]]
    if task == "fl2va":
        img = row["image"] if os.path.isabs(row["image"]) else os.path.join(REPO, row["image"])   # trainer and encoder run in REPO
        st = os.stat(img); parts += [_file_digest(img, st.st_size, st.st_mtime), f"{height}x{width}"]
    return f"{row.get('pid', 'p')}-{hashlib.sha1(chr(0).join(parts).encode()).hexdigest()[:16]}"


def encodings_needed(c: dict, pool: list) -> list[dict]:
    """One row per (prompt, canvas) the run can draw: rows carry _key, _h, _w for h3rl.data.encode."""
    out, seen = [], set()
    for frames in c["data"]["frames"]:
        h, w = canvas_for(c, frames)
        for r in eligible(pool, frames):
            k = text_key(r, h, w)
            if k not in seen: seen.add(k); out.append(dict(r, _key=k, _h=h, _w=w))
    return out
