"""Download what training needs, from the original sources:
  python scripts/download_weights.py                      # MiniMax-H3 + the reward models of every registered reward
  python scripts/download_weights.py --rewards videoalign hpspp   # only some reward workers
MiniMax-H3 files come from Comfy-Org/MiniMax-H3 (Comfy layout under $H3RL_MODELS) plus the processor/tokenizer files from
MiniMaxAI/MiniMax-H3. Reward weights go under $H3RL_REWARDS, their code repos are cloned there at pinned commits."""
import argparse, os, shutil, subprocess
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download, snapshot_download
from h3rl import paths
from h3rl.rewards.registry import WORKERS

H3_FILES = ["diffusion_models/minimax_h3_fl2va_bf16.safetensors", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rewards", nargs="*", default=list(WORKERS)); ap.add_argument("--skip_h3", action="store_true")
    ap.add_argument("--hyperflow", action="store_true", help="also fetch the 8-step HyperFlow LoRA (videorebirth/hyperflow) and convert it")
    a = ap.parse_args()
    if not a.skip_h3:
        for f in H3_FILES:
            out = os.path.join(paths.MODELS, f)
            if not os.path.exists(out):
                os.makedirs(os.path.dirname(out), exist_ok=True); print("downloading", f, flush=True)
                shutil.move(hf_hub_download("Comfy-Org/MiniMax-H3", f, local_dir=os.path.join(paths.MODELS, ".dl")), out)
        snapshot_download("MiniMaxAI/MiniMax-H3", allow_patterns=["*.json", "*.txt", "*.jinja", "*tokenizer*", "*.model"])
    from h3rl.core.hyperflow import is_current
    if a.hyperflow and not (os.path.exists(paths.HYPERFLOW_DEFAULT) and is_current(paths.HYPERFLOW_DEFAULT)):   # (re)convert stale files
        from h3rl.core.hyperflow import convert
        src = hf_hub_download("videorebirth/hyperflow", "minimax_h3_hyperflow_8step_v1.0.safetensors", local_dir=os.path.join(paths.MODELS, ".dl"))
        convert(src, paths.HYPERFLOW_DEFAULT)
    for w in a.rewards:
        spec = WORKERS[w]
        for repo, local in spec["hf"].items():
            print("downloading", repo, flush=True)
            snapshot_download(repo, local_dir=os.path.join(paths.REWARDS, local) if local else None)
        for name, (url, commit) in spec["git"].items():
            d = os.path.join(paths.REWARDS, name)
            if not os.path.isdir(d):
                subprocess.run(["git", "clone", "-q", url, d], check=True); subprocess.run(["git", "-C", d, "checkout", "-q", commit], check=True)
    print("done:", paths.MODELS, paths.REWARDS)


if __name__ == "__main__":
    main()
