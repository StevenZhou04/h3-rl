"""One entry point for training:  python -m h3rl.launch configs/experiments/nft_mix.yaml [key=value ...]

Resolves the experiment config (algorithm + reward + data + run), encodes any prompt missing from the text cache,
starts exactly the reward workers the reward config needs, optionally runs a 2-iteration smoke test with pass/fail
checks, then trains with torchrun (h3rl.train). Overrides use dotted keys: run.iters=50 algo.lr=5e-5 run.train_gpus=[0,1].
Multi-node: start the same command on every node with run.nnodes=N run.node_rank=i run.master_addr=<node 0 IP>
(scripts/launch_multinode.sh does this over ssh). Node 0 writes run.out (metrics, checkpoints); node i > 0 writes run.out/node<i>,
so run.out and the text cache may sit on a filesystem the nodes share. Rewards run on each node by default; run.reward_service=http://host:8800
sends them to h3rl.rewards.service on dedicated reward nodes instead.
Nothing here is algorithm-specific: the algorithm's config section is passed to it unchanged.
"""
from __future__ import annotations
import json, math, os, subprocess, sys, threading, time
from pathlib import Path
from omegaconf import OmegaConf
from h3rl import paths
from h3rl.algos import ALGORITHMS
from h3rl.rewards.combine import make_combiner
from h3rl.rewards.registry import workers_for
from h3rl.rewards.procs import Workers, env_with_repo

REPO = Path(__file__).resolve().parent.parent


def load_config(path: str, overrides: list[str]) -> dict:
    over = OmegaConf.from_dotlist(overrides); cfg = OmegaConf.load(path)
    for part in ("algo", "reward"):                      # a sub-config given as a file path (in the file or as an override) is inlined
        src = over.get(part) if isinstance(over.get(part), str) else cfg.get(part)
        if isinstance(src, str):
            cfg[part] = OmegaConf.load(REPO / src)
            if part in over and isinstance(over[part], str): over.pop(part)
    c = OmegaConf.to_container(OmegaConf.merge(cfg, over), resolve=True)
    if c["algo"]["name"] not in ALGORITHMS: raise SystemExit(f"unknown algorithm {c['algo']['name']!r}; known: {sorted(ALGORITHMS)}")
    c["data"]["pool"] = str((REPO / c["data"]["pool"]).resolve()) if not os.path.isabs(c["data"]["pool"]) else c["data"]["pool"]
    return c


def encode_missing(c: dict, out: Path, gpus):
    pool = [json.loads(l) for l in open(c["data"]["pool"])]; tc = Path(paths.CACHE) / "text_cache"; tc.mkdir(parents=True, exist_ok=True)
    miss = [r for r in pool if not (tc / f"{r['pid']}.pt").exists()]
    if not miss: return
    print(f"encoding {len(miss)} prompts on GPUs {gpus}", flush=True)
    f = out / "to_encode.jsonl"; f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in miss))
    procs = [subprocess.Popen([sys.executable, "-m", "h3rl.data.encode", "--pool", str(f), "--out", str(tc), "--shard", f"{i}/{len(gpus)}"],
                              env=env_with_repo(CUDA_VISIBLE_DEVICES=str(g)), cwd=REPO) for i, g in enumerate(gpus)]
    if any([p.wait() for p in procs]): raise RuntimeError("prompt encoding failed")   # list: wait for every shard, not just up to the first failure


EFA_ENV = {   # NCCL over AWS EFA through the aws-ofi-nccl plugin shipped on AWS GPU images (run.efa: true)
    "FI_PROVIDER": "efa", "FI_EFA_USE_DEVICE_RDMA": "1", "NCCL_PROTO": "simple",
    "LD_LIBRARY_PATH": "/opt/amazon/ofi-nccl/lib64:/opt/amazon/efa/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
}


def train(c: dict, out: Path, iters: int, save_every: int, log_path: Path, smoke: bool = False) -> tuple[int, int]:
    r = c["run"]; gpus = r["train_gpus"]; out.mkdir(parents=True, exist_ok=True)
    net = dict(EFA_ENV) if r.get("efa") else {}
    (out / "config.json").write_text(json.dumps(c, indent=1, ensure_ascii=False))
    names = [] if r.get("reward_service") else workers_for(make_combiner(c["reward"]).terms())   # remote service: no local workers
    workers = Workers(names, out / "queue", r.get("reward_gpus") or [0]) if names else None; peak = [0]; stop = threading.Event()
    def watch():
        while not stop.is_set():
            q = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            for line in q.split(): peak[0] = max(peak[0], int(line))
            stop.wait(20)
    threading.Thread(target=watch, daemon=True).start()
    try:
        if workers: workers.wait_loaded()
        nodes = ["--nnodes", str(r.get("nnodes", 1)), "--node_rank", str(r.get("node_rank", 0)), "--master_addr", r["master_addr"]] if int(r.get("nnodes", 1)) > 1 else []
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpus)), *nodes, "--master_port", str(r["port"]), "-m", "h3rl.train",
               "--config", str(out / "config.json"), "--out", str(out), "--iters", str(iters), "--save_every", str(save_every)]
        if r.get("resume"): cmd += ["--resume", r["resume"]] + (["--start_iter", str(r["start_iter"])] if r.get("start_iter") is not None else [])
        if smoke: cmd += ["--smoke"]
        with open(log_path, "a") as lf:
            rc = subprocess.call(cmd, env=env_with_repo(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **net),
                                 stdout=lf, stderr=subprocess.STDOUT, cwd=REPO)
    finally:
        stop.set()
        if workers: workers.stop()
    return rc, peak[0]


def smoke_check(out: Path, c: dict, peak: int) -> tuple[bool, str]:
    try: rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    except FileNotFoundError: rows = []
    need = [f"axis/{t}" for t in make_combiner(c["reward"]).terms()]
    tot = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(c["run"]["train_gpus"][0])],
                             capture_output=True, text=True).stdout.strip() or 0)
    ok = len(rows) == 2 and all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 and all(k in r for k in need) for r in rows)
    ok = ok and (not tot or peak < 0.96 * tot)
    g = lambda x, n: float(format(x, f".{n}g"))
    fr = [r.get("frames") for r in rows]; lo = [g(r["loss"], 4) for r in rows]; gr = [g(r["grad_norm"], 3) for r in rows]
    ti = [round(r["t_iter"] / 60, 1) for r in rows]; tr = [round(r.get("t_reward", 0) / 60, 1) for r in rows]
    missing = [k for k in need if rows and k not in rows[-1]]
    msg = ("OK" if ok else "FAIL") + f" | frames {fr} | loss {lo} | grad {gr} | iter min {ti} | reward min {tr} | peak MiB {peak}/{tot} | missing terms {missing}"
    return ok, msg


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    c = load_config(sys.argv[1], sys.argv[2:]); out = Path(c["run"]["out"]).expanduser()
    if not out.is_absolute(): out = REPO / out
    rank = int(c["run"].get("node_rank", 0))
    if int(c["run"].get("nnodes", 1)) > 1 and rank > 0: out = out / f"node{rank}"   # own queue, logs and rollouts when run.out is on shared storage
    out.mkdir(parents=True, exist_ok=True); (out / "config.resolved.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(c)))
    encode_missing(c, out, c["run"]["train_gpus"])
    if c["run"].get("smoke", True) and not c["run"].get("resume"):
        rc, peak = train(c, out / "smoke", 2, 100, out / "smoke" / "train.log", smoke=True)
        if rank == 0: ok, msg = smoke_check(out / "smoke", c, peak)    # metrics live on the head node
        else:
            tot = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(c["run"]["train_gpus"][0])], capture_output=True, text=True).stdout.strip() or 0)
            ok = rc == 0 and (not tot or peak < 0.96 * tot); msg = f"{'OK' if ok else 'FAIL'} (node {c['run']['node_rank']}: training rc {rc}, peak MiB {peak}/{tot})"
        print(f"smoke: {msg}", flush=True); (out / "smoke" / "verdict.txt").write_text(msg + "\n")
        if not ok or rc: sys.exit(f"smoke failed (rc={rc}); see {out}/smoke/train.log")
        if c["run"].get("smoke_only"): return
    rc, _ = train(c, out, c["run"]["iters"], c["run"]["save_every"], out / "train.log")
    sys.exit(rc)


if __name__ == "__main__":
    main()
