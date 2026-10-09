#!/bin/bash
# One-shot setup of a fresh AWS GPU node (e.g. p6-b200.48xlarge, Amazon Linux 2023) for h3rl.
#   curl -fsSL https://raw.githubusercontent.com/StevenZhou04/h3-rl/main/scripts/setup_aws_node.sh | bash
#   (or: bash scripts/setup_aws_node.sh from a clone)
# 1. RAID0 the empty instance-store NVMe drives into one XFS volume at /mnt/data (never touches drives that hold data;
#    instance-store contents are lost if the instance is stopped, so copy checkpoints off regularly)
# 2. uv, git, a static ffmpeg   3. clone h3rl to /mnt/data/h3rl-repo   4. build both environments   5. download weights
# Options (env): H3RL_HOME (default /mnt/data/h3rl), WITH_HYPERFLOW=1, REWARDS="videoalign hpspp solireward flowmotion".
set -euo pipefail
H=${H3RL_HOME:-/mnt/data/h3rl}; REPO=/mnt/data/h3rl-repo
if ! mountpoint -q /mnt/data; then
  ROOTDEV=$(lsblk -no PKNAME "$(findmnt -no SOURCE /)")
  DEVS=$(lsblk -dn -o NAME,TYPE | awk '$2=="disk"{print $1}' | grep nvme | grep -v "^$ROOTDEV$" || true)
  for d in $DEVS; do [ -z "$(lsblk -no FSTYPE /dev/$d | tr -d ' \n')" ] && [ "$(lsblk -n /dev/$d | wc -l)" -eq 1 ] || { echo "ABORT: /dev/$d is not empty"; exit 1; }; done
  N=$(echo $DEVS | wc -w); [ "$N" -gt 0 ] || { echo "no empty NVMe drives found"; exit 1; }
  command -v mdadm > /dev/null || sudo dnf install -y -q mdadm
  sudo mdadm --create /dev/md0 --level=0 --raid-devices=$N $(for d in $DEVS; do echo /dev/$d; done) --run
  sudo mkfs.xfs -q /dev/md0; sudo mkdir -p /mnt/data; sudo mount -o noatime /dev/md0 /mnt/data
  sudo mdadm --detail --scan | sudo tee -a /etc/mdadm.conf > /dev/null; echo "/dev/md0 /mnt/data xfs noatime,nofail 0 2" | sudo tee -a /etc/fstab > /dev/null
  sudo chown "$(id -un)":"$(id -gn)" /mnt/data
fi
command -v git > /dev/null || sudo dnf install -y -q git
[ -x ~/.local/bin/uv ] || command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/mnt/data/.uv_cache TMPDIR=/mnt/data/tmp HF_HOME=$H/hf_cache HF_HUB_DISABLE_XET=1; mkdir -p $TMPDIR
[ -d $REPO/.git ] || git clone -q https://github.com/StevenZhou04/h3-rl $REPO
cd $REPO && H3RL_HOME=$H bash scripts/setup_env.sh
cat > ~/.h3rlrc <<RC
export H3RL_HOME=$H H3RL_REWARD_PYTHON=$H/envs/reward/bin/python HF_HOME=$H/hf_cache HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/mnt/data/.uv_cache TMPDIR=/mnt/data/tmp PATH=\$HOME/.local/bin:\$PATH
RC
source ~/.h3rlrc; source $H/envs/main/bin/activate
uv pip install -q hf_transfer
python scripts/download_weights.py --rewards ${REWARDS:-videoalign hpspp solireward flowmotion cutcheck} $([ "${WITH_HYPERFLOW:-0}" = 1 ] && echo --hyperflow)
bash tests/run_tests.sh > /dev/null && echo "tests passed"
echo "ready: cd $REPO && source ~/.h3rlrc && source $H/envs/main/bin/activate && python -m h3rl.launch configs/experiments/nft_mix.yaml"
