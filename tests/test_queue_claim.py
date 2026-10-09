"""Replicas sharing a file queue: every request is claimed by exactly one of them. CPU-only."""
import os, sys, tempfile, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3rl.rewards import queue as Q


def drain(qdir, out):
    got = []
    while True:
        files, reqs = Q.pending_requests(qdir, "w", 3)
        if not reqs: break
        got += [r["key"] for r in reqs]; Q.write_results(qdir, "w", [{"key": r["key"], "scores": {"x": 1.0}} for r in reqs], files)
    out.put(got)


if __name__ == "__main__":
    qdir = tempfile.mkdtemp()
    for i in range(200): Q.submit(qdir, f"k{i:03d}", "/dev/null", "p", ["w"])
    out = mp.Queue(); ps = [mp.Process(target=drain, args=(qdir, out)) for _ in range(4)]
    for p in ps: p.start()
    keys = [k for _ in ps for k in out.get()]
    for p in ps: p.join()
    assert sorted(keys) == [f"k{i:03d}" for i in range(200)], f"{len(keys)} scored, {len(set(keys))} distinct"
    assert not os.listdir(f"{qdir}/in/w"), os.listdir(f"{qdir}/in/w")[:3]
    res = Q.collect(qdir, [f"k{i:03d}" for i in range(200)], ["w"], timeout_s=5)
    assert all(res[k]["w"]["scores"] == {"x": 1.0} for k in res)
    print("PASS  4 replicas scored 200 requests exactly once")
