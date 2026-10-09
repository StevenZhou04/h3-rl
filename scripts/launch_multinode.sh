#!/bin/bash
# Start one training run across several nodes over ssh, from any machine that can ssh to all of them.
#   scripts/launch_multinode.sh hosts.txt configs/experiments/nft_mix.yaml [key=value ...]
# Launching from a machine other than the nodes: set H3RL_REPO to the repo path on the nodes.
# hosts.txt: one node per line, "<ssh target> <private IP>"; the first line is node 0 (rendezvous + metrics + checkpoints).
# Every node needs the repo at the same path, the environments, the weights, and H3RL_* set in ~/.h3rlrc (sourced here).
# Nodes must reach each other on all TCP ports (e.g. a security-group rule allowing traffic from the group itself).
# run.out may be on storage the nodes share: node i > 0 writes to run.out/node<i>.
set -euo pipefail
HOSTS=$1; CFG=$2; shift 2; REPO=${H3RL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}; SSH=${SSH:-ssh}
LINES=(); while IFS= read -r l; do LINES+=("$l"); done < <(grep -vE '^\s*(#|$)' "$HOSTS")   # no mapfile: bash 3 on macOS
N=${#LINES[@]}; MASTER=$(echo "${LINES[0]}" | awk '{print $2}')
RUN_OUT=$(grep -oE "run.out=[^ ]+" <<< "$*" | cut -d= -f2 || true); LOG=${RUN_OUT:-runs}/launch_node
case $LOG in /*) ;; *) LOG=$REPO/$LOG ;; esac
ARGS=$(printf ' %q' "$@")   # quoted for the remote shell: zsh would glob run.train_gpus=[0,1]
for i in $(seq 0 $((N - 1))); do
  T=$(echo "${LINES[$i]}" | awk '{print $1}')
  $SSH "$T" "mkdir -p $(dirname $LOG) && cd $REPO && source ~/.h3rlrc && source \$H3RL_HOME/envs/main/bin/activate && \
    setsid nohup python -m h3rl.launch $CFG run.nnodes=$N run.node_rank=$i run.master_addr=$MASTER$ARGS > $LOG$i.log 2>&1 < /dev/null &" \
    && echo "node $i ($T): started, log $LOG$i.log"
done
