"""Benchmark partial gradient checkpointing on one GPU: for each number of un-checkpointed DiT blocks, the NFT loss step
(old + reference + policy forward, backward) on one real 5 s HyperFlow rollout: peak memory, time, and whether the
gradients equal the fully checkpointed ones. Pick run.plain_blocks from the largest count with headroom.

  CUDA_VISIBLE_DEVICES=4 python scripts/bench_checkpointing.py --plain 0 10 20 30 40 50
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3rl import paths
from h3rl.core.grpo import GRPOConfig, load_h3_for_rl, read_hyperflow_metadata, make_schedule, build_context
from h3rl.core.hyperflow import bind_schedule
from h3rl.core.nft import NFTConfig, iter_samples, nft_loss
from h3rl.core.checkpointing import limit_checkpointing
from h3rl.data.pool import text_key

ap = argparse.ArgumentParser(); ap.add_argument("--plain", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50])
ap.add_argument("--frames", type=int, default=124); ap.add_argument("--size", type=int, nargs=2, default=[544, 960]); a = ap.parse_args()
dev = torch.device("cuda:0"); hf = paths.HYPERFLOW_DEFAULT
lcfg = GRPOConfig(infer_steps=8, network_dim=32, network_alpha=16.0, height=a.size[0], width=a.size[1], frame_count=a.frames)
models = load_h3_for_rl(dit_path=paths.DIT_BF16, text_encoder_path="unused", video_vae_path=paths.VIDEO_VAE, audio_vae_path=paths.AUDIO_VAE,
                        device=dev, cfg=lcfg, attn_mode="sdpa", skip_text_encoder=True, base_lora_paths=[hf])
from h3rl.core.rope_fast import install_fast_rope; install_fast_rope()
tr, net = models["transformer"], models["network"]; tr.enable_gradient_checkpointing()
sch = make_schedule(lcfg, dev, read_hyperflow_metadata(hf)); bind_schedule(tr, sch)
params = [p for p in net.parameters() if p.requires_grad]
with torch.no_grad():                                   # a non-trivial LoRA so every parameter gets a gradient
    g = torch.Generator(device="cpu").manual_seed(0)
    for p in params: p.add_(1e-3 * torch.randn(p.shape, generator=g).to(p.device, p.dtype))
old = [p.detach().clone() for p in params]
pool = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "prompts/example_pool.jsonl")]
pr = next(r for r in pool if r.get("min_frames") == a.frames and r["task"] == "t2va")
hs, tags = torch.load(f"{paths.CACHE}/text_cache/{text_key(pr, *a.size)}.pt", map_location="cpu", weights_only=False)
ctx = build_context(pr["prompt"], hs, tags, lcfg, dev, task="t2va", condition_latent=None, condition_geometry=None,
                    condition_path="first_frame.png", condition_frame=None)
cfg = NFTConfig(infer_steps=8, frame_count=a.frames, height=a.size[0], width=a.size[1])
tr.eval(); sample = next(iter_samples(tr, net, ctx, cfg, sch, dev, [123])); tr.train()
print(f"blocks {len(list(tr.blocks))}; rollout ready", flush=True)
ref = None
print(f"{'plain':>6}{'peak GiB':>10}{'step s':>9}{'max |dgrad| / |grad|':>24}", flush=True)
for n in a.plain:
    limit_checkpointing(tr, n); times = []; reps = []
    for rep in range(2):                                  # rep 0 warms up; rep 1 is timed
        for p in params: p.grad = None
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        try: nft_loss(tr, net, params, old, sample, ctx, 0.8, None, cfg, sch, dev, random.Random(7), loss_scale=1.0)
        except torch.cuda.OutOfMemoryError: print(f"{n:>6}       OOM", flush=True); break
        torch.cuda.synchronize(); times.append(time.time() - t); reps.append([p.grad.detach().float().clone() for p in params])
    else:
        grads = reps[-1]
        if ref is None:
            ref = grads; noise = max(float((x - y).abs().max() / (y.abs().max() + 1e-12)) for x, y in zip(reps[0], reps[1]))
            print(f"  run-to-run gradient noise with full checkpointing (nondeterministic kernels): {noise:.2e}", flush=True)
        rel = max(float((x - y).abs().max() / (y.abs().max() + 1e-12)) for x, y in zip(grads, ref))
        print(f"{n:>6}{torch.cuda.max_memory_allocated() / 2**30:>10.1f}{times[-1]:>9.2f}{rel:>24.2e}", flush=True)
        continue
    torch.cuda.empty_cache()
