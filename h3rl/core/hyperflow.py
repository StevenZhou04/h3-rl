"""HyperFlow (videorebirth/hyperflow), the 8-step distilled MiniMax-H3 LoRA, converted from diffusers layout to musubi's.

  python -m h3rl.core.hyperflow --in minimax_h3_hyperflow_8step_v1.0.safetensors --out hyperflow_musubi.safetensors
  (scripts/download_weights.py --hyperflow downloads and converts it)

musubi only strips the `transformer.` prefix, so a diffusers LoRA attaches to 0 modules without this. Differences:
  1. module names: transformer_blocks -> blocks, ff.net.{0.proj,2} -> mlp.{fc1,fc2}, attn.to_out.0 -> attn.out_proj,
     token_refiner.refiner_blocks -> token_refiner.blocks, {,endpoint_}time_embedder.linear_{1,2} -> .proj_{in,out}
  2. fused QKV: musubi has one attn.qkv_proj (q, k, v in that order) where diffusers has three adapters. A sum of three
     low-rank updates is exactly A_fused = [A_q; A_k; A_v], B_fused = blockdiag(B_q, B_k, B_v); rank triples, so alpha
     triples too (musubi scales by alpha / rank).
  3. SwiGLU halves: diffusers' ff.net.0.proj outputs [value; gate], musubi's mlp.fc1 [gate; value] (diffusers' own
     H3 converter swaps them), so the two row halves of fc1's lora_up swap places.
  4. Two-time conditioning: HyperFlow embeds every step as emb(t) + gate * (emb_r(r) - emb(t)), r = the step's endpoint
     (1 - next sigma; conditioning rows keep r = t), with emb_r an extra copy of the time embedder carrying its own LoRA
     (endpoint_time_embedder). musubi's H3 has no such module: install_two_time() adds it before the LoRA is attached
     and blends the two embeddings in a forward hook; bind_schedule() gives it the sigma grids that define r.
The file header (sampling sigma grid, shifts, gate) is kept: the trainer reads the 8-step grid from it.
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
    (r"^endpoint_time_embedder\.linear_1$", r"endpoint_time_embedder.proj_in"),
    (r"^endpoint_time_embedder\.linear_2$", r"endpoint_time_embedder.proj_out"),
]
SWIGLU_FC1 = r"(^|\.)mlp\.fc1$"     # diffusers [value; gate] -> musubi [gate; value]
QKV_PATTERNS = [
    (r"^transformer_blocks\.(\d+)\.attn\.to_([qkv])$", r"blocks.\1.attn.qkv_proj"),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_([qkv])$", r"token_refiner.blocks.\1.attn.qkv_proj"),
]
UNSUPPORTED: list[str] = []
CONVERSION_VERSION = 2      # 2: SwiGLU halves swapped, endpoint_time_embedder kept (1 = earlier conversion, wrong)


def is_current(path: str) -> bool:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f: return (f.metadata() or {}).get("h3rl_conversion") == str(CONVERSION_VERSION)


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
        if any(re.match(p, module) for p in UNSUPPORTED): dropped["unsupported"] += 1; continue
        fused = None
        for pat, repl in QKV_PATTERNS:
            mm = re.match(pat, module)
            if mm: fused = re.sub(pat, repl, module); qkv[fused][mm.group(2)]["down" if ab == "A" else "up"] = tensor; break
        if fused: continue
        for pat, repl in RENAMES:
            if re.match(pat, module):
                t = re.sub(pat, repl, module)
                if ab == "B" and re.search(SWIGLU_FC1, t):
                    value, gate = tensor.chunk(2, dim=0); tensor = torch.cat([gate, value], dim=0).contiguous()
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
    new_meta = dict(meta, converted_from="videorebirth/hyperflow diffusers layout", converted_rank_qkv=str(rank * 3), format="pt",
                    h3rl_conversion=str(CONVERSION_VERSION))
    Path(dst).parent.mkdir(parents=True, exist_ok=True); save_file(out, dst, metadata={k: str(v) for k, v in new_meta.items()})
    if verbose:
        print(f"wrote {dst}: {len(out)} tensors ({direct} direct, {fused_n} fused qkv modules)")
        for reason, n in sorted(dropped.items(), key=lambda x: -x[1]): print(f"   dropped {n:>4}  {reason}")
    return dict(direct=direct, fused=fused_n, dropped=dict(dropped))



# ----------------------------------------------------------------------------- two-time conditioning at run time
def has_endpoint_lora(path: str) -> bool:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f: return any(k.startswith("lora_unet_endpoint_time_embedder") for k in f.keys())


def hyperflow_gate(path: str) -> float:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f: return float((f.metadata() or {}).get("hyperflow_gate", 0.25))


def install_two_time(transformer, gate: float) -> None:
    """Add `endpoint_time_embedder` (a copy of the base time embedder; the HyperFlow LoRA attaches to it by name, so call
    this before attaching) and blend emb(t) + gate * (emb_r(r) - emb(t)) in a forward hook on the time embedder."""
    import copy
    if getattr(transformer, "_two_time", None) is not None: return
    if transformer.time_embedder is None: raise ValueError("HyperFlow two-time conditioning needs the unpruned time embedder")
    transformer.endpoint_time_embedder = copy.deepcopy(transformer.time_embedder)
    st = transformer._two_time = {"gate": float(gate), "video": None, "audio": None, "t_video": None, "t_audio": None}

    def pre(_mod, _args, kwargs):                                  # remember this forward's target-row model times
        st["t_video"], st["t_audio"] = kwargs.get("model_t_video"), kwargs.get("model_t_audio")

    def post(_mod, inp, out):
        if st["video"] is None: raise RuntimeError("HyperFlow: call hyperflow.bind_schedule(transformer, schedule) before sampling")
        t = inp[0].to(torch.float32); r = t.clone()
        for key, grid in (("t_video", st["video"]), ("t_audio", st["audio"])):
            mt = st[key]
            if mt is None: continue
            mt = float(mt); r = torch.where((t - mt).abs() < 1e-6, torch.full_like(t, 1.0 - next_sigma(grid, 1.0 - mt)), r)
        return out + st["gate"] * (transformer.endpoint_time_embedder(r) - out)

    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.time_embedder.register_forward_hook(post)


def bind_schedule(transformer, schedule) -> None:
    """Give the two-time hook the sampler's sigma grids (endpoint r = 1 - the next grid sigma). No-op without HyperFlow."""
    st = getattr(transformer, "_two_time", None)
    if st is not None: st["video"], st["audio"] = [float(x) for x in schedule.video], [float(x) for x in schedule.audio]


def next_sigma(grid: list[float], sigma: float) -> float:
    """The grid sigma a step from `sigma` lands on: the largest grid value below it (on-grid: the next one)."""
    below = [g for g in grid if g < sigma - 1e-6]
    return max(below) if below else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="src", required=True); ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args(); convert(a.src, a.dst)
