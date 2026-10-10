#!/bin/bash
# Correctness gates. Seconds on CPU, no checkpoints. GPU=1 adds the RoPE kernel check.
set -euo pipefail
cd "$(dirname "$0")/.."; PY="${PYTHON:-python}"; export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "=== 1/4 policy-gradient correctness (SDE log-prob replay) ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_grpo_ratio.py
echo "=== 2/4 LoRA + GRPO end-to-end convergence ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_rl_correctness.py
echo "=== 0 reward combiner, registry, algorithm plumbing ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_combine.py
echo "=== 0b reward service over HTTP ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_reward_service.py
echo "=== 0c reward-worker replicas claim each request once ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_queue_claim.py
echo "=== 0d stopping workers ends their child processes ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_worker_stop.py
echo "=== 0e checkpoint pruning ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_prune.py
echo "=== 0f HyperFlow sigma grids and two-time endpoints ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_hyperflow_schedule.py
echo "=== 0g speed-ups keep results (partial checkpointing, streamed rollouts) ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_speedups.py
echo "=== 3/4 DiffusionNFT loss + SAGE-GRPO switches ==="; CUDA_VISIBLE_DEVICES="" "$PY" tests/test_nft_sage.py
if [ -n "${GPU:-}" ]; then echo "=== 4/4 fast RoPE equals the reference (GPU) ==="; "$PY" tests/test_rope_gpu.py; else echo "=== 4/4 skipped (GPU=1 to run) ==="; fi
echo "ALL TESTS PASSED"
