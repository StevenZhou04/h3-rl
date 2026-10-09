"""Where the trainer sends reward requests. Same three calls either way:
  LocalQueue(dir)   -- file queue on this node's disk, read by reward workers on this node (default)
  RemoteQueue(url)  -- HTTP client for h3rl.rewards.service running on dedicated reward nodes (uploads the videos)"""
from __future__ import annotations
import json, time, urllib.request, urllib.parse
from h3rl.rewards import queue as fq


class LocalQueue:
    def __init__(self, path: str): self.path = path
    def submit(self, key, mp4, prompt, workers, meta): fq.submit(self.path, key, mp4, prompt, workers, meta)
    def submit_group(self, key, members, prompt, worker, meta): fq.submit_group(self.path, key, members, prompt, worker, meta)
    def collect(self, keys, workers, timeout_s): return fq.collect(self.path, keys, workers, timeout_s=timeout_s)


class RemoteQueue:
    def __init__(self, url: str, retries: int = 5): self.url = url.rstrip("/"); self.retries = retries; self.sent = set()

    def _req(self, path, data: bytes | None = None, ctype="application/json", query=None):
        u = f"{self.url}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        for i in range(self.retries):
            try:
                req = urllib.request.Request(u, data=data, method="POST" if data is not None else "GET", headers={"Content-Type": ctype})
                with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read() or b"{}")
            except Exception:
                if i == self.retries - 1: raise
                time.sleep(2 ** i)

    def _upload(self, key, mp4):
        if key in self.sent: return
        with open(mp4, "rb") as f: self._req("/upload", f.read(), "application/octet-stream", {"key": key})
        self.sent.add(key)

    def submit(self, key, mp4, prompt, workers, meta):
        if not workers: return
        self._upload(key, mp4); self._req("/submit", json.dumps(dict(key=key, prompt=prompt, workers=list(workers), meta=meta)).encode())

    def submit_group(self, key, members, prompt, worker, meta):
        for m in members: self._upload(m["key"], m["mp4"])
        self._req("/submit_group", json.dumps(dict(key=key, members=[m["key"] for m in members], prompt=prompt, worker=worker, meta=meta)).encode())

    def collect(self, keys, workers, timeout_s):
        t0 = time.time(); out = {k: {} for k in keys}; pending = {(k, w) for k in keys for w in workers}
        while pending and time.time() - t0 < timeout_s:
            got = self._req("/results", json.dumps([list(p) for p in pending]).encode())
            for k, w, r in got.get("results", []): out[k][w] = r; pending.discard((k, w))
            if pending: time.sleep(3)
        for k, w in pending: out[k][w] = {"key": k, "worker": w, "scores": {}, "error": "timeout"}
        return out


def make_backend(run: dict, local_dir: str):
    return RemoteQueue(run["reward_service"]) if run.get("reward_service") else LocalQueue(local_dir)
