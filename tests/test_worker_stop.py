"""Stopping reward workers also ends the processes they spawned (CPU workers run process pools). CPU-only, Linux/macOS."""
import os, sys, tempfile, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import h3rl.rewards.procs as P

if __name__ == "__main__":
    q = tempfile.mkdtemp(); P.WORKERS["_fake"] = dict(env="main"); P.CPU_WORKERS.add("_fake")
    code = "import multiprocessing as mp, time\nif __name__ == '__main__':\n    ps = [mp.Process(target=time.sleep, args=(600,)) for _ in range(3)]\n    [p.start() for p in ps]; print('loaded', flush=True); time.sleep(600)\n"
    open(f"{q}/fake_worker.py", "w").write(code)
    w = P.Workers.__new__(P.Workers); w.names, w.queue, w.procs = ["_fake"], q, []
    w.procs.append(subprocess.Popen([sys.executable, f"{q}/fake_worker.py"], start_new_session=True))
    time.sleep(3); pg = w.procs[0].pid
    kids = subprocess.run(["pgrep", "-g", str(pg)], capture_output=True, text=True).stdout.split()
    assert len(kids) >= 4, kids
    w.stop(grace=3); time.sleep(1)
    left = subprocess.run(["pgrep", "-g", str(pg)], capture_output=True, text=True).stdout.split()
    assert not left, f"left behind: {left}"
    print(f"PASS  stop() ended the worker and its {len(kids) - 1} children")
