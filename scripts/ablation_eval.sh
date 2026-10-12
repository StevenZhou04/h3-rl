#!/bin/bash
# Held-out check of a reward ablation: the final checkpoint (or the one at ITER) of every runs/abl_<name> run against the base model, on the
# 64 held-out prompts, then one table of every judge term per run (h3rl.eval.ablation_table). Arms are named after the
# run (checkpoints are linked as <out>/ckpts/<name>.safetensors), so the base videos are made once and shared.
# Rerunnable: finished videos and scores are kept, so runs that finish later are added by running it again.
#
#   bash scripts/ablation_eval.sh runs/heldout_abl 0,1,2,3,4,5,6,7 [ITER] [RUN_DIR ...]
# Default runs: every runs/abl_* and runs/nft_hf_64g* except the brightness sanity run, each at its last iteration.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 OUT_DIR GPUS [ITER] [RUN_DIR ...]"; exit 1; }
OUT=$(realpath -m "$1"); GPUS=$2; shift 2; ITER=
if [[ ${1:-} =~ ^[0-9]+$ ]]; then ITER=$1; shift; fi
REPO=$(cd "$(dirname "$0")/.." && pwd); RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then shopt -s nullglob; for r in "$REPO"/runs/abl_* "$REPO"/runs/nft_hf_64g*; do [[ $r == *_bright ]] || RUNS+=("$r"); done; fi
mkdir -p "$OUT/ckpts"; CKPTS=()
for r in "${RUNS[@]}"; do
  it=${ITER:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["run"]["iters"])' "$r/config.json" 2>/dev/null || echo 0)}
  c="$r/nft-$(printf %05d "$it").safetensors"
  if [ -f "$c" ]; then n=$(basename "$r"); n=${n#abl_}; ln -sf "$(realpath "$c")" "$OUT/ckpts/$n.safetensors"; cp "$r/config.json" "$OUT/ckpts/$n.config.json"; CKPTS+=("$OUT/ckpts/$n.safetensors")
  else echo "skip $r: no $(basename "$c") yet"; fi
done
[ ${#CKPTS[@]} -gt 0 ] || { echo "no finished checkpoints"; exit 1; }
todo=()
for c in "${CKPTS[@]}"; do [ -f "$OUT/scores_base_$(basename "$c" .safetensors).json" ] || todo+=("$c"); done
[ ${#todo[@]} -eq 0 ] || HELDOUT_SET="${HELDOUT_SET:-$REPO/prompts/t2va_heldout_v2.jsonl}" bash "$REPO/scripts/heldout_eval.sh" "$OUT" "$GPUS" "${todo[@]}"
cd "$REPO" && python -m h3rl.eval.ablation_table --dir "$OUT"
