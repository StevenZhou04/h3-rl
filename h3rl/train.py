"""Shared RL trainer for MiniMax-H3 (one process per GPU, started by h3rl.launch through torchrun).

  python -m torch.distributed.run --nproc_per_node N -m h3rl.train --config <run>/config.json --out <dir> --iters K

Per iteration: pick the clip-length bucket -> per prompt build the conditioning and let the algorithm sample a group
-> decode + write each rollout -> send per-video and per-group reward requests to the worker queue -> combine scores
(h3rl.rewards.combine) -> algorithm.update -> metrics, checkpoint. Algorithms only implement rollout() and update().
"""
from __future__ import annotations
import argparse, json, os, random, time
import numpy as np, torch, torch.distributed as dist
from h3rl import paths
from h3rl.algos import ALGORITHMS
from h3rl.algos.base import TrainContext
from h3rl.core.grpo import GRPOConfig, load_h3_for_rl, read_hyperflow_metadata, make_schedule, build_context, encode_condition_latent
from h3rl.core.nft import decode_and_write
from h3rl.core.dist import dist_setup, average_metrics
from h3rl.rewards.combine import make_combiner, WORST
from h3rl.rewards.backend import make_backend
from h3rl.rewards.registry import WORKERS, workers_for


def canvas_for(c: dict, frames: int) -> tuple[int, int]:
    d = c["data"]; fs = {int(k): v for k, v in (d.get("frame_sizes") or {}).items()}
    return tuple(fs.get(frames, d["size"]))


def eligible(pool, frames):
    return [r for r in pool if r.get("min_frames", 0) <= frames <= r.get("max_frames", 10 ** 9)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, required=True); ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--start_iter", type=int, default=0); ap.add_argument("--resume", default=None, help="checkpoint prefix (without .safetensors)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="after the run, check loss/grad/reward terms on rank 0 and exit 3 on every rank if they fail")
    a = ap.parse_args()
    c = json.load(open(a.config)); m, d, ac, rc, run = c.get("model", {}), c["data"], c["algo"], c["reward"], c["run"]
    rank, world, device = dist_setup(a.device); is_main = rank == 0
    out = a.out; Q = f"{out}/queue"; os.makedirs(f"{out}/rollouts", exist_ok=True)
    log = lambda s: print(f"[r{rank} {time.strftime('%H:%M:%S')}] {s}", flush=True) if is_main else None

    hyper_path = m.get("hyperflow") or ""
    if hyper_path == "default": hyper_path = paths.HYPERFLOW_DEFAULT
    if hyper_path and not os.path.exists(hyper_path): raise SystemExit(f"HyperFlow LoRA not found: {hyper_path} (scripts/download_weights.py --hyperflow)")
    infer_steps = int(m.get("infer_steps") or (8 if hyper_path else 30))
    lcfg = GRPOConfig(infer_steps=infer_steps, network_dim=int(m.get("lora_rank", 32)), network_alpha=float(m.get("lora_alpha", 16.0)),
                      height=d["size"][0], width=d["size"][1], frame_count=d["frames"][0])
    models = load_h3_for_rl(dit_path=m.get("dit") or paths.DIT_BF16, text_encoder_path="unused", video_vae_path=paths.VIDEO_VAE,
                            audio_vae_path=paths.AUDIO_VAE, device=device, cfg=lcfg, attn_mode="sdpa", skip_text_encoder=True,
                            base_lora_paths=([hyper_path] if hyper_path else []) + list(m.get("adapters") or []) or None)
    from h3rl.core.rope_fast import install_fast_rope; install_fast_rope()
    transformer, network = models["transformer"], models["network"]; transformer.enable_gradient_checkpointing()
    schedule = make_schedule(lcfg, device, read_hyperflow_metadata(hyper_path) if hyper_path else None)
    params = [p for p in network.parameters() if p.requires_grad]
    T = TrainContext(models=models, transformer=transformer, network=network, params=params, schedule=schedule, device=device,
                     rank=rank, world=world, rng=random.Random(int(run.get("seed", 0)) * 1000 + rank), infer_steps=infer_steps)
    algo = ALGORITHMS[ac["name"]](ac, T); comb = make_combiner(rc)
    terms = comb.terms(); workers = workers_for(terms)
    per_video = [w for w in workers if WORKERS[w].get("kind", "video") == "video"]; per_group = [w for w in workers if WORKERS[w].get("kind") == "group"]
    if a.resume:
        network.load_weights(a.resume + ".safetensors"); st = json.load(open(a.resume + ".state.json"))
        comb.load(st["combiner"]); algo.load(st["algo"]); log(f"resumed from {a.resume}")
    tc = f"{paths.CACHE}/text_cache"; pool = [json.loads(l) for l in open(d["pool"])]; pool = [r for r in pool if os.path.exists(f"{tc}/{r['pid']}.pt")]
    log(f"algo={ac['name']} infer_steps={infer_steps} workers={workers} pool={len(pool)} world={world}")
    timeout = float(run.get("reward_timeout_s", 5400)); rq = make_backend(run, Q); history = []

    for it in range(a.start_iter, a.iters):
        t0 = time.time(); network.set_multiplier(1.0); transformer.eval()
        frames = d["frames"][it % len(d["frames"])]; h, w = canvas_for(c, frames)              # same length on every rank
        T.canvas = dict(frames=frames, height=h, width=w); lcfg.frame_count, lcfg.height, lcfg.width = frames, h, w
        prompts = T.rng.sample(eligible(pool, frames), algo.prompts_per_step()); samples = []
        for j, pr in enumerate(prompts):
            hs, tags = torch.load(f"{tc}/{pr['pid']}.pt", map_location="cpu", weights_only=False)
            latent = geometry = cond_px = None
            if pr["task"] == "fl2va":
                from musubi_tuner.minimax_h3.generation_inputs import load_image_frames
                fr = load_image_frames(pr["image"], width=w, height=h); latent, geometry = encode_condition_latent(fr, models["video_vae"], device); cond_px = np.asarray(fr[0])
            ctx = build_context(pr["prompt"], hs, tags, lcfg, device, task=pr["task"], condition_latent=latent, condition_geometry=geometry,
                                condition_path=pr.get("image") or "first_frame.png", condition_frame=cond_px)
            for k, s in enumerate(algo.rollout(ctx, pr, it * 100003 + rank * 1009 + j * 101)):
                key = f"it{it:04d}_r{rank}_p{j}_k{k}"; mp4 = f"{out}/rollouts/it{it:04d}/{key}.mp4"; os.makedirs(os.path.dirname(mp4), exist_ok=True)
                s.update(key=key, mp4=mp4, guard=decode_and_write(models, s["video"], s["audio"], mp4, device), ctx=ctx, prompt=pr, group=j)
                samples.append(s); json.dump({"key": key, "pid": pr.get("pid"), "prompt": pr.get("prompt"), "guard": s["guard"]}, open(mp4[:-4] + ".json", "w"))
        t_roll = time.time() - t0
        # rewards: per-video workers get every rollout, group workers one request per prompt group
        rtext = lambda s: s["prompt"].get("reward_prompt") or s["prompt"]["prompt"]
        meta = lambda s: {"camera_move": s["prompt"].get("camera_move"), "audio_prompt": s["prompt"].get("audio_prompt"), "rl_progress": it / max(a.iters, 1)}
        for s in samples: rq.submit(s["key"], s["mp4"], rtext(s), per_video, meta(s))
        gkeys = {}
        for j in sorted({s["group"] for s in samples}):
            grp = [s for s in samples if s["group"] == j]; gk = f"it{it:04d}_r{rank}_p{j}"; gkeys[j] = gk
            for wk in per_group: rq.submit_group(gk, [{"key": s["key"], "mp4": s["mp4"]} for s in grp], rtext(grp[0]), wk, meta(grp[0]))
        got = rq.collect([s["key"] for s in samples], per_video, timeout) if per_video else {s["key"]: {} for s in samples}
        ggot = rq.collect(list(gkeys.values()), per_group, timeout) if per_group else {}
        for s in samples:
            axes = {}
            for res in got[s["key"]].values(): axes.update({k: float(v) for k, v in (res.get("scores") or {}).items()})
            for res in ggot.get(gkeys[s["group"]], {}).values(): axes.update({k: float(v) for k, v in ((res.get("members") or {}).get(s["key"]) or {}).items()})
            s["R"] = comb(axes, s["guard"], bool(s["prompt"].get("has_audio", True)))
        t_rew = time.time() - t0 - t_roll
        mt = algo.update(samples, it)
        Rv = [s["R"]["video"] for s in samples if s["R"]["video"] is not None]
        mt.update(R_video_mean=float(np.mean(Rv)) if Rv else 0.0, R_video_std=float(np.std(Rv)) if Rv else 0.0,
                  worst_frac=float(np.mean([s["R"]["video"] == WORST for s in samples])), gated=float(np.mean([s["R"]["gated"] for s in samples])),
                  frames=float(frames), height=float(h), t_rollout=t_roll, t_reward=t_rew, t_iter=time.time() - t0, n_samples=float(len(samples)))
        for ax in sorted({k for s in samples for k in s["R"]["axes"]}):
            mt[f"axis/{ax}"] = float(np.mean([s["R"]["axes"][ax] for s in samples if ax in s["R"]["axes"]]))
        if world > 1: mt = average_metrics({k: v for k, v in mt.items() if isinstance(v, float)}, world, device)
        mt["iter"] = it; history.append(mt)
        if is_main:
            log(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in mt.items()}))
            with open(f"{out}/metrics.jsonl", "a") as f: f.write(json.dumps(mt) + "\n")
            if (it + 1) % a.save_every == 0 or it + 1 == a.iters:
                p = f"{out}/{ac['name']}-{it + 1:05d}"; network.save_weights(p + ".safetensors", torch.bfloat16, {"iter": str(it + 1), "algo": ac["name"]})
                json.dump({"combiner": comb.state(), "algo": algo.state(), "iter": it + 1}, open(p + ".state.json", "w")); log(f"saved {p}")
        if world > 1: dist.barrier()
    if a.smoke:                                    # one verdict for every rank on every node (no shared disk needed)
        import math
        need = [f"axis/{t}" for t in terms]
        ok = len(history) == a.iters - a.start_iter and all(math.isfinite(m["loss"]) and math.isfinite(m["grad_norm"]) and m["grad_norm"] > 0
                                                            and all(k in m for k in need) for m in history)
        if world > 1:
            flag = torch.tensor([1.0 if ok else 0.0], device=device); dist.all_reduce(flag, op=dist.ReduceOp.MIN); ok = bool(flag.item())
        log(f"SMOKE {'OK' if ok else 'FAIL'} | missing terms {[k for k in need if history and k not in history[-1]]}")
        if world > 1: dist.destroy_process_group()
        if not ok: raise SystemExit(3)
    log("DONE")


if __name__ == "__main__":
    main()
