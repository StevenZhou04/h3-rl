"""Reward service round trip on CPU: a trainer-side RemoteQueue uploads a video, submits per-video and group requests,
a fake worker answers through the service's queue, and collect() returns the results."""
import json, os, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer
from pathlib import Path
from h3rl.rewards.service import make_handler
from h3rl.rewards.backend import RemoteQueue
from h3rl.rewards import queue as fq
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

root = Path(tempfile.mkdtemp()); (root / "queue").mkdir()
srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root)); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"
def fake_worker(stop):            # answers every request in the service queue: per-video and group
    while not stop.is_set():
        for w in ("dummy", "dummy_group"):
            files, reqs = fq.pending_requests(str(root / "queue"), w, 16)
            res = []
            for r in reqs:
                if "members" in r: res.append({"key": r["key"], "scores": {}, "members": {m["key"]: {"g": float(i)} for i, m in enumerate(r["members"])}})
                else: res.append({"key": r["key"], "scores": {"size": float(os.path.getsize(r["mp4"]))}})
            if reqs: fq.write_results(str(root / "queue"), w, res, files)
        time.sleep(0.2)
stop = threading.Event(); threading.Thread(target=fake_worker, args=(stop,), daemon=True).start()
vids = []
for i in range(3):
    p = Path(tempfile.mkdtemp()) / f"v{i}.mp4"; p.write_bytes(os.urandom(1000 + i)); vids.append(p)
rq = RemoteQueue(url)
for i, p in enumerate(vids): rq.submit(f"k{i}", str(p), "a prompt", ["dummy"], {"m": 1})
rq.submit_group("g0", [{"key": f"k{i}", "mp4": str(p)} for i, p in enumerate(vids)], "a prompt", "dummy_group", {})
got = rq.collect([f"k{i}" for i in range(3)], ["dummy"], timeout_s=30); gg = rq.collect(["g0"], ["dummy_group"], timeout_s=30)
check("per-video results come back through HTTP", all(got[f"k{i}"]["dummy"]["scores"].get("size") == 1000 + i for i in range(3)), json.dumps({k: v["dummy"]["scores"] for k, v in got.items()}))
check("uploaded bytes are intact (size scored by the worker)", got["k2"]["dummy"]["scores"]["size"] == 1002)
check("group result carries per-member scores", gg["g0"]["dummy_group"].get("members", {}).get("k2") == {"g": 2.0})
check("each video uploaded once", len(rq.sent) == 3)
stop.set(); srv.shutdown()
if FAILS: print("FAILED:", FAILS); sys.exit(1)
print("ALL PASS")
