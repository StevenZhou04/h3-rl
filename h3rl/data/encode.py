"""Encode pool prompts once with the NVFP4 text encoder into per-prompt files (text_cache/<key>.pt = (hidden_states,
token_tags) on CPU, key = h3rl.data.pool.text_key). Rows come from pool.encodings_needed (with _key, _h, _w: fl2va images
are cropped to the canvas the rollout uses); plain pool rows are keyed at --size. Resumable."""
from h3rl.paths import TEXT_ENCODER
import argparse, json, os, socket, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--text_encoder", default=TEXT_ENCODER)
ap.add_argument("--size", type=int, nargs=2, default=[544, 960]); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
si, sn = map(int, a.shard.split("/")); os.makedirs(a.out, exist_ok=True)
rows = [json.loads(l) for l in open(a.pool)][si::sn]
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from h3rl.data.pool import text_key
for r in rows:                                                     # plain pool rows: the trainer's key at --size
    r.setdefault("_h", a.size[0]); r.setdefault("_w", a.size[1]); r.setdefault("_key", text_key(r, r["_h"], r["_w"]))
todo = [r for r in rows if not os.path.exists(f"{a.out}/{r['_key']}.pt")]
print(f"shard {a.shard}: {len(todo)} of {len(rows)} to encode", flush=True)
if not todo: sys.exit(0)
from h3rl.core.grpo import encode_prompt
from musubi_tuner.minimax_h3.text_encoder import load_h3_processor, load_h3_text_encoder
from musubi_tuner.minimax_h3.generation_inputs import load_image_frames
dev = torch.device("cuda:0"); proc, te = load_h3_processor(), load_h3_text_encoder(a.text_encoder, device=dev, dtype=torch.bfloat16)
t0 = time.time()
for i, r in enumerate(todo):
    fr = load_image_frames(r["image"], width=r["_w"], height=r["_h"]) if r["task"] == "fl2va" else None
    h, t = encode_prompt(r["prompt"], proc, te, dev, task=r["task"], condition_frames=fr)
    tmp = f"{a.out}/{r['_key']}.pt.{socket.gethostname()}.{os.getpid()}.tmp"     # unique: nodes sharing the cache may encode the same prompt
    torch.save((h.cpu(), t.cpu()), tmp); os.replace(tmp, f"{a.out}/{r['_key']}.pt")
    if i % 200 == 0: print(f"{i}/{len(todo)} {(time.time()-t0)/(i+1):.2f}s/prompt", flush=True)
print("DONE", flush=True)
