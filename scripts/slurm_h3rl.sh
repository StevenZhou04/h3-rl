#!/usr/bin/env bash
# Run one h3rl training job as a Slurm batch job that survives preemption (single node).
#   sbatch -p low --gres=gpu:4 -t 3:00:00 --requeue --signal=B:USR1 --open-mode=append -o OUT.slurm.log \
#     scripts/slurm_h3rl.sh OUT configs/experiments/nft_hyperflow.yaml [key=value ...]
# The job trains on every GPU of its allocation (Slurm shows them as 0..N-1) with the reward workers on GPU 0, unless
# the overrides say otherwise. Every start, including a requeue after preemption, resumes from the newest checkpoint in
# OUT, so add run.save_every=1 to lose at most one iteration. Slurm's SIGUSR1 stops the run; checkpoints are written
# atomically, so stopping at any point is safe. Needs the environments and ~/.h3rlrc of scripts/setup_env.sh.
set -uo pipefail
OUT=$1; CFG=$2; shift 2
REPO=${H3RL_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
source ~/.h3rlrc; cd "$REPO"
N=$(nvidia-smi -L | wc -l); G=$(seq -s, 0 $((N - 1)))
CK=$(ls "$OUT"/*-[0-9][0-9][0-9][0-9][0-9].state.json 2>/dev/null | sort | tail -1)
RESUME=(); [ -n "$CK" ] && RESUME=(run.resume="${CK%.state.json}")
echo "[$(date -u '+%F %T')] job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0}: $N GPUs, resume ${CK:-none}"
"$H3RL_HOME/envs/main/bin/python" -m h3rl.launch "$CFG" run.train_gpus="[$G]" run.reward_gpus="[0]" \
  run.port=$((29000 + ${SLURM_JOB_ID:-0} % 1000)) "$@" run.out="$OUT" "${RESUME[@]}" &
PID=$!
trap 'echo "[$(date -u +%T)] SIGUSR1: preempted, stopping"; kill -TERM $PID' USR1
while kill -0 $PID 2> /dev/null; do wait $PID; done
wait $PID; RC=$?
echo "[$(date -u '+%F %T')] exit $RC"; exit $RC
