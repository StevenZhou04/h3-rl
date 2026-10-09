"""Where h3rl finds weights and writes data. Every location is an environment variable under one root:

  H3RL_HOME     (default ~/h3rl)          root
  H3RL_MODELS   $H3RL_HOME/models         MiniMax-H3 weights, Comfy-Org/MiniMax-H3 layout (scripts/download_weights.py)
  H3RL_REWARDS  $H3RL_HOME/rewards        reward-model weights and their code repos
  H3RL_CACHE    $H3RL_HOME/cache          text encodings, worker scratch files
  H3RL_DIT      override for the transformer checkpoint (default: the bf16 FL2VA DiT)
  H3RL_HYPERFLOW optional step-distillation LoRA; empty = train the base model
"""
import os

_e = lambda k, d: os.path.expanduser(os.environ.get(k, d))
HOME = _e("H3RL_HOME", "~/h3rl")
MODELS = _e("H3RL_MODELS", f"{HOME}/models")
REWARDS = _e("H3RL_REWARDS", f"{HOME}/rewards")
CACHE = _e("H3RL_CACHE", f"{HOME}/cache")
SECRETS = _e("H3RL_SECRETS", "~/.h3rl_secrets")          # optional key files (e.g. wandb_key), never in the repo

DIT_BF16 = _e("H3RL_DIT", f"{MODELS}/diffusion_models/minimax_h3_fl2va_bf16.safetensors")
TEXT_ENCODER = _e("H3RL_TEXT_ENCODER", f"{MODELS}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
VIDEO_VAE = f"{MODELS}/vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = f"{MODELS}/vae/minimax_h3_audio_vae_fp32.safetensors"
HYPERFLOW = os.environ.get("H3RL_HYPERFLOW", "")
HYPERFLOW_DEFAULT = f"{MODELS}/loras/hyperflow_musubi.safetensors"   # written by download_weights.py --hyperflow

# names used throughout the training and worker modules
H3_ROOT, H3_RM, H3_CKPTS, H3_SECRETS = CACHE, REWARDS, MODELS, SECRETS
