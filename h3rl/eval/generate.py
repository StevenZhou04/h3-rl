"""Judge-free eval generation: base (BF16) [+ HyperFlow] [+ adapters] on the frozen eval set, fixed
seeds, 8 steps, deterministic ODE. Writes one mp4 per (prompt, seed) plus a manifest for the scorers.
Text encodings come from the RL text cache when present; anything else (held-out SpatialVID captions,
short prompts) is encoded once with the NVFP4 text encoder, which is loaded and freed BEFORE the DiT."""
from h3rl.paths import H3_ROOT, H3_CKPTS, DIT_BF16, HYPERFLOW, HYPERFLOW_DEFAULT, TEXT_ENCODER
import argparse, json, os, sys, time, subprocess
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
from h3rl.core.grpo import (GRPOConfig, load_h3_for_rl, read_hyperflow_metadata, make_schedule, build_context,
                             encode_condition_latent, decode_rollout_to_video, encode_prompt)
from h3rl.core.sde import h3_sde_step
from h3rl.core.video_metrics import compute_video_metrics
from musubi_tuner.minimax_h3.generation_inputs import load_image_frames

ap = argparse.ArgumentParser()
ap.add_argument("--eval_set", nargs="+", required=True, help="jsonl files with eid/prompt/task/image")
ap.add_argument("--out", required=True)
ap.add_argument("--dit", default=DIT_BF16)
ap.add_argument("--hyperflow", default="auto", help="'' for the base model; 'default' or a path for HyperFlow; auto: HyperFlow if converted, "
               "but with --adapters you must choose (a LoRA trained on the base model must not get HyperFlow added)")
ap.add_argument("--adapters", nargs="*", default=[], help="extra LoRA safetensors (e.g. the SFT LoRA), attached after HyperFlow")
ap.add_argument("--steps", type=int, default=None, help="default: 8 with HyperFlow, 30 on the base model")
ap.add_argument("--frames", type=int, default=124, help="clip length for prompts without min_frames (17n+5 at 24 fps)")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--text_cache", default=f"{H3_ROOT}/eval_text_cache.pt")
ap.add_argument("--text_encoder", default=TEXT_ENCODER)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed_mode", choices=["shared", "prompt"], default="shared", help="shared: one noise per seed index for every prompt (legacy); prompt: independent noise per prompt")
ap.add_argument("--size", type=int, nargs=2, default=[544, 960], metavar=("H", "W"), help="eval canvas (default for every length)")
ap.add_argument("--frame_sizes", nargs="*", default=["243:480x832"], help="per-length canvas FRAMES:HxW, as the experiments' data.frame_sizes")
a = ap.parse_args()
if a.hyperflow == "auto":
    if a.adapters: sys.exit("--adapters given: pass --hyperflow default (the LoRA was trained on HyperFlow) or --hyperflow '' (base model)")
    a.hyperflow = HYPERFLOW or (HYPERFLOW_DEFAULT if os.path.exists(HYPERFLOW_DEFAULT) else "")
elif a.hyperflow == "default": a.hyperflow = HYPERFLOW_DEFAULT
if a.hyperflow and not os.path.exists(a.hyperflow): sys.exit(f"HyperFlow LoRA not found: {a.hyperflow} (scripts/download_weights.py --hyperflow)")
if a.hyperflow:
    from h3rl.core.hyperflow import is_current
    if not is_current(a.hyperflow): sys.exit(f"{a.hyperflow} is from an older, incorrect conversion; rerun scripts/download_weights.py --hyperflow")
FS = {int(k): tuple(int(x) for x in v.split("x")) for k, v in (x.split(":") for x in a.frame_sizes)}
def canvas(r): return FS.get(int(r.get("min_frames") or a.frames), tuple(a.size))   # (H, W) for this row's clip length
CK = H3_CKPTS; os.makedirs(a.out, exist_ok=True); device = torch.device("cuda:0")
if a.steps is None: a.steps = 8 if a.hyperflow else 30
cfg = GRPOConfig(infer_steps=a.steps, width=a.size[1], height=a.size[0], frame_count=a.frames)
rows = [json.loads(l) for f in a.eval_set for l in open(f)]
if a.limit: rows = rows[:a.limit]
done = {json.loads(l)["key"] for l in open(f"{a.out}/manifest.jsonl")} if os.path.exists(f"{a.out}/manifest.jsonl") else set()
todo = [(r, s) for r in rows for s in a.seeds if f"{r['eid']}_s{s}" not in done]
print(f"{len(rows)} prompts x {len(a.seeds)} seeds; {len(todo)} to generate", flush=True)
if not todo: sys.exit(0)

# --- text encodings: cache hits + one pass of the text encoder for the rest
from h3rl.data.pool import text_key
tkey = lambda r: text_key(dict(r, pid=r["eid"]), *canvas(r))   # task + prompt (+ image and canvas for fl2va), not the prompt alone
tc = torch.load(a.text_cache, map_location="cpu", weights_only=False) if os.path.exists(a.text_cache) else {}; tc.pop("__tasks__", None)
missing = [r for r in rows if tkey(r) not in tc]
if missing:
    from musubi_tuner.minimax_h3.text_encoder import load_h3_processor, load_h3_text_encoder
    print(f"encoding {len(missing)} prompts with the text encoder", flush=True)
    proc, te = load_h3_processor(), load_h3_text_encoder(a.text_encoder, device=device, dtype=torch.bfloat16)
    for r in missing:
        H, W = canvas(r); fr = load_image_frames(r["image"], width=W, height=H) if r["task"] == "fl2va" else None
        h, t = encode_prompt(r["prompt"], proc, te, device, task=r["task"], condition_frames=fr)
        tc[tkey(r)] = (h.cpu(), t.cpu())
    del te; torch.cuda.empty_cache()
    tmp = f"{a.text_cache}.{os.getpid()}.tmp"; torch.save(tc, tmp); os.replace(tmp, a.text_cache)

hyper = read_hyperflow_metadata(a.hyperflow) if a.hyperflow else None
base_loras = ([a.hyperflow] if a.hyperflow else []) + list(a.adapters)
t0 = time.time()
models = load_h3_for_rl(dit_path=a.dit, text_encoder_path="unused", video_vae_path=f"{CK}/vae/minimax_h3_video_vae_fp16.safetensors",
                        audio_vae_path=f"{CK}/vae/minimax_h3_audio_vae_fp32.safetensors", device=device, cfg=cfg, attn_mode="sdpa",
                        skip_text_encoder=True, base_lora_paths=base_loras or None)
from h3rl.core.rope_fast import install_fast_rope; install_fast_rope()
transformer, network = models["transformer"], models["network"]; network.set_multiplier(0.0); transformer.eval()
schedule = make_schedule(cfg, device, hyper)
from h3rl.core.hyperflow import bind_schedule; bind_schedule(models["transformer"], schedule)
print(f"model loaded {time.time()-t0:.0f}s; adapters={base_loras}; sigmas={[round(float(x),3) for x in schedule.video]}", flush=True)
from musubi_tuner.minimax_h3.sampling import initialize_target_latents, augment_condition_latents
from musubi_tuner.minimax_h3.packing import VIDEO_CHANNELS, AUDIO_CHANNELS, STEREO_CHANNELS

def write_mp4(frames, path, fps=24):
    F, H, W, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes()); p.stdin.close(); p.wait()

man = open(f"{a.out}/manifest.jsonl", "a")
ctx_cache = {}
for n, (r, seed) in enumerate(todo):
    key = f"{r['eid']}_s{seed}"
    if r["eid"] not in ctx_cache:
        h, t = tc[tkey(r)]; latent = geometry = cond_px = None; cfg.frame_count = int(r.get("min_frames") or a.frames)
        cfg.height, cfg.width = canvas(r)                              # same canvas as training for this length
        if r["task"] == "fl2va":
            frames = load_image_frames(r["image"], width=cfg.width, height=cfg.height)
            latent, geometry = encode_condition_latent(frames, models["video_vae"], device); cond_px = np.asarray(frames[0])
        ctx_cache[r["eid"]] = build_context(r["prompt"], h, t, cfg, device, task=r["task"], condition_latent=latent, condition_geometry=geometry,
                                            condition_path=r.get("image") or "first_frame.png", condition_frame=cond_px)
    ctx = ctx_cache[r["eid"]]; layout = ctx["layout"]
    import zlib
    eff = seed if a.seed_mode == "shared" else (seed * 1000003 + zlib.crc32(r["eid"].encode())) % (2**31)
    gen = torch.Generator(device="cpu").manual_seed(eff)
    video, audio = initialize_target_latents(video_shape=(1, VIDEO_CHANNELS, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
                                             audio_shape=(1, AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames), generator=gen, device=device,
                                             video_dtype=torch.float32, audio_dtype=torch.float32)   # as sample_joint_av_latents: fp32, noise first
    vis_cond, aud_cond = augment_condition_latents(ctx["raw_visual_conditions"], ctx["audio_condition_latents"], generator=gen, visual_clean=ctx["visual_condition_clean"], device=device)
    ts = time.time()
    with torch.no_grad():
        for k in range(cfg.infer_steps):
            sv, svn = schedule.video[k], schedule.video[k + 1]; sa, san = schedule.audio[k], schedule.audio[k + 1]
            pred = transformer(video_latents=video, audio_latents=audio, text_hidden_states=ctx["text_hidden_states"], text_token_tags=ctx["text_token_tags"],
                               layout=layout, model_t_video=1.0 - sv, model_t_audio=1.0 - sa, visual_condition_latents=vis_cond,
                               audio_condition_latents=aud_cond, visual_condition_clean=ctx["visual_condition_clean"])
            video, _, _, _ = h3_sde_step(video, pred.video, sv, svn, schedule.video[1], noise_level=0.0)
            audio, _, _, _ = h3_sde_step(audio, pred.audio, sa, san, schedule.audio[1], noise_level=0.0)
        vid = decode_rollout_to_video(video, audio, models)
    m = compute_video_metrics(vid)
    write_mp4(vid, f"{a.out}/{key}.mp4")
    rec = dict(key=key, eid=r["eid"], seed=seed, prompt=r["prompt"], task=r["task"], image=r.get("image"), source=r.get("source"),
               camera_move=r.get("camera_move"), variant=r.get("variant"), mp4=f"{a.out}/{key}.mp4", gen_s=round(time.time() - ts, 2),
               **{k: round(float(v), 4) for k, v in m.items()})
    man.write(json.dumps(rec) + "\n"); man.flush()
    if n % 10 == 0: print(f"{n+1}/{len(todo)} {key} {rec['gen_s']}s", flush=True)
json.dump(dict(size=a.size, frame_sizes=a.frame_sizes, dit=a.dit, hyperflow=a.hyperflow, adapters=a.adapters, steps=a.steps, seeds=a.seeds, sigmas=[float(x) for x in schedule.video]), open(f"{a.out}/config.json", "w"), indent=1)
print("DONE", flush=True)
