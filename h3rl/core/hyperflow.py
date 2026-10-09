"""HyperFlow (videorebirth/hyperflow), the 8-step distilled MiniMax-H3 LoRA, converted from diffusers layout to musubi's.

  python -m h3rl.core.hyperflow --in minimax_h3_hyperflow_8step_v1.0.safetensors --out hyperflow_musubi.safetensors
  (scripts/download_weights.py --hyperflow downloads and converts it)

musubi only strips the `transformer.` prefix, so a diffusers LoRA attaches to 0 modules without this. Differences:
  1. module names: transformer_blocks -> blocks, ff.net.{0.proj,2} -> mlp.{fc1,fc2}, attn.to_out.0 -> attn.out_proj,
     token_refiner.refiner_blocks -> token_refiner.blocks, time_embedder.linear_{1,2} -> time_embedder.{proj_in,proj_out}
  2. fused QKV: musubi has one attn.qkv_proj (q, k, v in that order) where diffusers has three adapters. A sum of three
     low-rank updates is exactly A_fused = [A_q; A_k; A_v], B_fused = blockdiag(B_q, B_k, B_v); rank triples, so alpha
     triples too (musubi scales by alpha / rank).
  3. endpoint_time_embedder has no counterpart in musubi's H3 and is dropped. Harmless for text-to-video; for
     first-frame prompts validate before relying on it.
The file header (sampling sigma grid, shifts) is kept: the trainer reads the 8-step grid from it.
"""
from __future__ import annotations
import argparse, re
from collections import defaultdict
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

RENAMES = [
    (r"^transformer_blocks\.(\d+)\.attn\.to_out\.0$", r"blocks.\1.attn.out_proj"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj$", r"blocks.\1.mlp.fc1"),
    (r"^transformer_blocks\.(\d+)\.ff\.net\.2$", r"blocks.\1.mlp.fc2"),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0$", r"token_refiner.blocks.\1.attn.out_proj"),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj$", r"token_refiner.blocks.\1.mlp.fc1"),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2$", r"token_refiner.blocks.\1.mlp.fc2"),
    (r"^time_embedder\.linear_1$", r"time_embedder.proj_in"),
    (r"^time_embedder\.linear_2$", r"time_embedder.proj_out"),
]
QKV_PATTERNS = [
    (r"^transformer_blocks\.(\d+)\.attn\.to_([qkv])$", r"blocks.\1.attn.qkv_proj"),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_([qkv])$", r"token_refiner.blocks.\1.attn.qkv_proj"),
]
UNSUPPORTED = [r"^endpoint_time_embedder\."]


def musubi_key(module_path: str) -> str:
    return "lora_unet_" + module_path.replace(".", "_")


def convert(src: str, dst: str, verbose: bool = True) -> dict:
    sd = load_file(src)
    from safetensors import safe_open
    with safe_open(src, framework="pt") as f: meta = f.metadata() or {}
    if not meta.get("hyperflow"): raise ValueError(f"{src} has no HyperFlow header (key 'hyperflow'); is this the HyperFlow LoRA?")
    rank = int(meta.get("lora_rank", 256)); alpha = float(meta.get("lora_alpha", rank))
    out, dropped, direct = {}, defaultdict(int), 0
    qkv: dict = defaultdict(lambda: defaultdict(dict))
    for key, tensor in sd.items():
        if not key.startswith("transformer."): dropped["no transformer. prefix"] += 1; continue
        m = re.match(r"^(.*)\.lora_(A|B)\.weight$", key[len("transformer."):])
        if not m: dropped["not a lora_A/lora_B tensor"] += 1; continue
        module, ab = m.group(1), m.group(2)
        if any(re.match(p, module) for p in UNSUPPORTED): dropped["endpoint_time_embedder (absent in musubi H3)"] += 1; continue
        fused = None
        for pat, repl in QKV_PATTERNS:
            mm = re.match(pat, module)
            if mm: fused = re.sub(pat, repl, module); qkv[fused][mm.group(2)]["down" if ab == "A" else "up"] = tensor; break
        if fused: continue
        for pat, repl in RENAMES:
            if re.match(pat, module):
                t = re.sub(pat, repl, module)
                out[f"{musubi_key(t)}.lora_{'down' if ab == 'A' else 'up'}.weight"] = tensor
                out[f"{musubi_key(t)}.alpha"] = torch.tensor(alpha, dtype=torch.float32); direct += 1; break
        else:
            dropped[f"unmapped: {re.sub(r'[0-9]+', 'N', module)}"] += 1
    fused_n = 0
    for target, parts in qkv.items():
        if set(parts) != {"q", "k", "v"} or any(set(p) != {"down", "up"} for p in parts.values()):
            dropped[f"incomplete qkv: {re.sub(r'[0-9]+', 'N', target)}"] += 1; continue
        downs = [parts[x]["down"] for x in "qkv"]; ups = [parts[x]["up"] for x in "qkv"]; r = downs[0].shape[0]
        b = torch.zeros(sum(u.shape[0] for u in ups), 3 * r, dtype=ups[0].dtype); row = 0
        for i, u in enumerate(ups): b[row:row + u.shape[0], i * r:(i + 1) * r] = u; row += u.shape[0]
        out[f"{musubi_key(target)}.lora_down.weight"] = torch.cat(downs, dim=0); out[f"{musubi_key(target)}.lora_up.weight"] = b
        out[f"{musubi_key(target)}.alpha"] = torch.tensor(alpha * 3.0, dtype=torch.float32); fused_n += 1
    new_meta = dict(meta, converted_from="videorebirth/hyperflow diffusers layout", converted_rank_qkv=str(rank * 3), format="pt")
    Path(dst).parent.mkdir(parents=True, exist_ok=True); save_file(out, dst, metadata={k: str(v) for k, v in new_meta.items()})
    if verbose:
        print(f"wrote {dst}: {len(out)} tensors ({direct} direct, {fused_n} fused qkv modules)")
        for reason, n in sorted(dropped.items(), key=lambda x: -x[1]): print(f"   dropped {n:>4}  {reason}")
    return dict(direct=direct, fused=fused_n, dropped=dict(dropped))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="src", required=True); ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args(); convert(a.src, a.dst)
