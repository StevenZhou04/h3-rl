"""Common loop for reward workers: poll the queue, score a batch, write results. Subclasses implement
`load()` and `score(requests) -> list[{"key", "scores": {...}}]`; any exception becomes an error result
for that batch (the trainer treats errors as missing axes, never as zeros)."""
from __future__ import annotations
from h3rl.paths import H3_ROOT
import argparse, os, sys, time, traceback, subprocess, tempfile, shutil
TMP_ROOT = f"{H3_ROOT}/worker_tmp"   # under H3RL_CACHE, not /tmp: frame dumps can be large
from h3rl.rewards.queue import pending_requests, write_results


def sample_frames(mp4: str, n: int, out_dir: str, short_side: int = 448) -> list[str]:
    """n uniformly spaced JPEG frames (ffmpeg), returns paths. Cheap and venv-independent."""
    os.makedirs(out_dir, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp4],
                               capture_output=True, text=True).stdout.strip() or 5.0)
    paths = []
    for k in range(n):
        t = (k + 0.5) / n * dur; p = f"{out_dir}/f{k:02d}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", mp4, "-frames:v", "1",
                        "-vf", f"scale=-2:{short_side}", "-q:v", "2", p], check=False, stdin=subprocess.DEVNULL)
        if os.path.exists(p): paths.append(p)
    return paths


def extract_wav(mp4: str, out_path: str, sr: int = 48000, mono: bool = True) -> str | None:
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", mp4, "-vn", "-ar", str(sr), "-ac", "1" if mono else "2", out_path], capture_output=True, stdin=subprocess.DEVNULL)
    return out_path if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000 else None


class Worker:
    name = "base"
    batch = 4

    def load(self): ...
    def score(self, requests: list[dict]) -> list[dict]: ...

    def run(self):
        ap = argparse.ArgumentParser(); ap.add_argument("--queue", required=True); ap.add_argument("--batch", type=int, default=self.batch)
        ap.add_argument("--once", action="store_true", help="process what is queued and exit (tests)")
        a = ap.parse_args(); self.args = a; os.makedirs(TMP_ROOT, exist_ok=True)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)   # child ffmpeg/decoders must never read our stdin (it can be a piped script)
        self.tmp = tempfile.mkdtemp(prefix=f"rw_{self.name}_", dir=TMP_ROOT)
        t0 = time.time(); self.load(); print(f"[{self.name}] loaded in {time.time()-t0:.0f}s", flush=True)
        idle = 0
        while True:
            files, reqs = pending_requests(a.queue, self.name, a.batch)
            if not reqs:
                if a.once and idle > 2: return
                idle += 1; time.sleep(2.0); continue
            idle = 0; t1 = time.time()
            try:
                res = self.score(reqs)
            except Exception:
                err = traceback.format_exc()[-1500:]; print(f"[{self.name}] batch error:\n{err}", flush=True)
                res = [{"key": r["key"], "scores": {}, "error": err} for r in reqs]
            write_results(a.queue, self.name, res, files)
            for r in reqs:   # per-request scratch (frames, wavs) is not needed after scoring
                shutil.rmtree(f"{self.tmp}/{r['key']}", ignore_errors=True)
                for ext in (".wav",): 
                    try: os.remove(f"{self.tmp}/{r['key']}{ext}")
                    except FileNotFoundError: pass
            print(f"[{self.name}] scored {len(res)} in {time.time()-t1:.1f}s", flush=True)


class GroupWorker(Worker):
    """A reward that needs a whole group of rollouts at once (pairwise or listwise judges, win rates).
    Requests carry {"key": group_key, "members": [{"key", "mp4"}, ...], "prompt"}; implement score_groups(requests) ->
    [{"key": group_key, "members": {member_key: {term: value}}}]. The trainer gives every member its own scores."""
    batch = 1

    def score_groups(self, requests: list[dict]) -> list[dict]: ...

    def score(self, requests):
        return [dict(r, scores={}) for r in self.score_groups(requests)]
