#!/bin/bash
# Held-out check of RL checkpoints against the base model (HyperFlow 8 steps, 5 s clips, one fixed noise per prompt):
# generates the 32 prompts of prompts/heldout_rl.jsonl with the base model and with every checkpoint, then scores each
# checkpoint against base, paired by prompt (h3rl.eval.reward_check). Finished videos are kept, so a rerun only adds what
# is missing (e.g. a new checkpoint against the same base videos).
#
#   bash scripts/heldout_eval.sh runs/heldout 0,1,2,3 runs/nft_hf_mix/nft-00030.safetensors runs/nft_hf_mix/nft-00060.safetensors
#
# GPUs: one generation process per GPU (~60 GB each), then the reward workers spread over the same GPUs.
# Results: <out>/scores_base_<ckpt>.json and the printed tables (also in <out>/report.txt).
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 OUT_DIR GPUS(e.g. 0,1,2,3) CKPT.safetensors [CKPT ...]"; exit 1; }
OUT=$(realpath -m "$1"); IFS=, read -ra GPUS <<< "$2"; shift 2; CKPTS=("$@")
REPO=$(cd "$(dirname "$0")/.." && pwd); SET=$REPO/prompts/heldout_rl.jsonl; N=${#GPUS[@]}
for c in "${CKPTS[@]}"; do [ -f "$c" ] || { echo "missing checkpoint $c"; exit 1; }; done
mkdir -p "$OUT/shards"; cd "$REPO"
for i in $(seq 0 $((N - 1))); do awk -v n=$N -v i=$i 'NR % n == i' "$SET" > "$OUT/shards/$i.jsonl"; done

gen() {   # gen ARM [--adapters CKPT]: every shard on its own GPU, in parallel
  local arm=$1; shift; local pids=()
  for i in $(seq 0 $((N - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m h3rl.eval.generate --eval_set "$OUT/shards/$i.jsonl" --out "$OUT/${arm}_$i" \
      --hyperflow default --seeds 0 --seed_mode prompt --text_cache "$OUT/tc_$i.$arm.pt" "$@" > "$OUT/${arm}_$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { echo "generation failed for $arm (see $OUT/${arm}_*.log)"; exit 1; }; done
  echo "generated $arm: $(cat "$OUT"/${arm}_*/manifest.jsonl | wc -l) videos"
}

gen base
for c in "${CKPTS[@]}"; do gen "$(basename "$c" .safetensors)" --adapters "$(realpath "$c")"; done
for c in "${CKPTS[@]}"; do
  arm=$(basename "$c" .safetensors)
  python -m h3rl.eval.reward_check --dir "$OUT" --arms base "$arm" --gpus "${GPUS[@]}" --eval_set "$SET" 2>&1 \
    | grep -vE "^INFO|Warning|warn" | tee -a "$OUT/report.txt"
done
