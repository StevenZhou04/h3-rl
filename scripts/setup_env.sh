#!/bin/bash
# Build the two Python environments h3rl uses (needs uv: https://docs.astral.sh/uv/).
#   main   ($H3RL_HOME/envs/main)   : h3rl + MiniMax-H3 model code (upstream musubi-tuner, pinned) + torch cu130
#   reward ($H3RL_HOME/envs/reward) : VideoAlign/VideoReward, which needs transformers 4.45 + torch cu128
# Also installs a static ffmpeg into ~/.local/bin if none is on PATH.
set -e
HERE=$(cd $(dirname $0)/.. && pwd); H=${H3RL_HOME:-$HOME/h3rl}; mkdir -p $H/envs
MUSUBI="musubi-tuner @ git+https://github.com/kohya-ss/musubi-tuner@4e7c7149249e7715e9168920feb4c420423abba7"
uv python install 3.10 -q
if [ ! -x $H/envs/main/bin/python ]; then
  uv venv -q --python 3.10 $H/envs/main; M=$H/envs/main/bin/python
  uv pip install -q --python $M torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
  uv pip install -q --python $M -r $HERE/envs/main.txt
  uv pip install -q --python $M --no-deps "$MUSUBI"
  uv pip install -q --python $M --no-deps -e $HERE
fi
if [ ! -x $H/envs/reward/bin/python ]; then
  uv venv -q --python 3.10 $H/envs/reward; R=$H/envs/reward/bin/python
  uv pip install -q --python $R torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python $R -r $HERE/envs/reward.txt
fi
if ! command -v ffmpeg > /dev/null; then
  mkdir -p ~/.local/bin; T=$(mktemp -d); curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ -C $T
  cp $T/ffmpeg-*-static/ffmpeg $T/ffmpeg-*-static/ffprobe ~/.local/bin/; rm -rf $T
fi
$H/envs/main/bin/python -c "import torch, h3rl, musubi_tuner.minimax_h3.model; print('main ok:', torch.__version__, torch.cuda.device_count(), 'GPUs')"
$H/envs/reward/bin/python -c "import torch, transformers; print('reward ok:', torch.__version__, transformers.__version__)"
echo "export H3RL_HOME=$H H3RL_REWARD_PYTHON=$H/envs/reward/bin/python   # add to your shell; activate: source $H/envs/main/bin/activate"
