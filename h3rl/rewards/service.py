"""Reward service for multi-node training: reward models on dedicated nodes, trainers send videos over HTTP.

  python -m h3rl.rewards.service --reward configs/reward/mix_v1.yaml --gpus 0 1 2 3 4 5 6 7 --replicas 2 --port 8800

Starts the reward workers that the reward config needs (each GPU worker `--replicas` times, sharing one queue) and
serves:  POST /upload?key=K (raw mp4)   POST /submit {key, prompt, workers, meta}
         POST /submit_group {key, members, prompt, worker, meta}   POST /results [[key, worker], ...] -> found results
Trainers point at it with run.reward_service: http://<host>:8800. Uploaded videos are deleted after an hour.
"""
from __future__ import annotations
import argparse, json, os, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from omegaconf import OmegaConf
from h3rl import paths
from h3rl.rewards import queue as fq
from h3rl.rewards.combine import make_combiner
from h3rl.rewards.procs import Workers
from h3rl.rewards.registry import workers_for


def make_handler(root: Path):
    vids = root / "videos"; q = str(root / "queue"); vids.mkdir(parents=True, exist_ok=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._json({"ok": True}) if urlparse(self.path).path == "/health" else self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path); body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if u.path == "/upload":
                    key = parse_qs(u.query)["key"][0]; assert "/" not in key and ".." not in key
                    tmp = vids / f"{key}.mp4.part"; tmp.write_bytes(body); os.replace(tmp, vids / f"{key}.mp4"); return self._json({"ok": True})
                d = json.loads(body or b"null")
                if u.path == "/submit":
                    fq.submit(q, d["key"], str(vids / f"{d['key']}.mp4"), d["prompt"], d["workers"], d.get("meta")); return self._json({"ok": True})
                if u.path == "/submit_group":
                    fq.submit_group(q, d["key"], [{"key": k, "mp4": str(vids / f"{k}.mp4")} for k in d["members"]], d["prompt"], d["worker"], d.get("meta"))
                    return self._json({"ok": True})
                if u.path == "/results":
                    found = []
                    for k, w in d:
                        p = Path(q) / "out" / w / f"{k}.json"
                        if p.exists():
                            try: found.append([k, w, json.loads(p.read_text())])
                            except json.JSONDecodeError: pass
                    return self._json({"results": found})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
    return H


def janitor(root: Path, max_age: float):
    while True:
        cut = time.time() - max_age
        for p in list((root / "videos").glob("*.mp4")) + list((root / "queue" / "out").glob("*/*.json")):
            try:
                if p.stat().st_mtime < cut: p.unlink()
            except FileNotFoundError: pass
        time.sleep(300)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reward", required=True, help="reward config (decides which workers run)")
    ap.add_argument("--gpus", type=int, nargs="+", default=list(range(8))); ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--keep_s", type=float, default=4 * 3600, help="delete uploads and results older than this (keep above run.reward_timeout_s)")
    ap.add_argument("--port", type=int, default=8800); ap.add_argument("--root", default=f"{paths.CACHE}/reward_service")
    ap.add_argument("--workers", nargs="*", default=None, help="override the worker list (default: from the reward config)")
    a = ap.parse_args(); root = Path(a.root); (root / "queue").mkdir(parents=True, exist_ok=True)
    names = a.workers if a.workers is not None else workers_for(make_combiner(OmegaConf.to_container(OmegaConf.load(a.reward))).terms())
    w = Workers(names, root / "queue", a.gpus, a.replicas) if names else None
    if w: w.wait_loaded()
    threading.Thread(target=janitor, args=(root, a.keep_s), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(root))
    print(f"reward service on :{a.port} | workers {names} x{a.replicas} on GPUs {a.gpus} | root {root}", flush=True)
    try: srv.serve_forever()
    finally:
        if w: w.stop()


if __name__ == "__main__":
    main()
