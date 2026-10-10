# h3rl

Reinforcement-learning post-training for **MiniMax-H3**, the joint video + audio generator. Choose an algorithm and a
reward in a YAML file; h3rl samples rollouts from the model, scores them with reward models running as separate
worker processes, and trains a LoRA on the result.

- **Algorithms:** DiffusionNFT (forward-process, likelihood-free), Flow-GRPO, and SAGE-GRPO (Flow-GRPO with a precise
  SDE step variance, a per-timestep gradient equalizer and a dual trust region). New ones plug in as one class.
- **Rewards:** VideoAlign (visual quality, motion quality, text alignment), HPSv3++ (frame aesthetics), SoliReward
  (text alignment, physics), UnifiedReward 2.0, an optical-flow motion score that ignores shake and flicker, and a
  hard-cut penalty for one-shot prompts, and group rewards such as a pairwise judge's win rate. Combine them with
  weights, gates and floors.
- **Base or distilled model:** train the base model (30-step sampler) or the 8-step HyperFlow-distilled model
  (`python scripts/download_weights.py --hyperflow`, then `model.hyperflow: default`; see `configs/experiments/*_hyperflow.yaml`).
- **Clip lengths:** alternate buckets such as 5.2 s and 10.1 s; each prompt can be restricted to the buckets it suits.

## Setup

**On a fresh AWS GPU node** (Amazon Linux 2023, e.g. p6-b200), one command sets up the NVMe volume, environments,
weights and runs the tests: `curl -fsSL https://raw.githubusercontent.com/StevenZhou04/h3-rl/main/scripts/setup_aws_node.sh | bash`
(`WITH_HYPERFLOW=1` adds the 8-step LoRA). Elsewhere:

Linux, NVIDIA GPUs with enough memory for the bf16 model (B200/H200 class; one training rank per GPU), and
[uv](https://docs.astral.sh/uv/).

```bash
export H3RL_HOME=/path/with/space          # weights (~140 GB), caches and runs go here
bash scripts/setup_env.sh                  # builds $H3RL_HOME/envs/{main,reward}; prints the exports to add
source $H3RL_HOME/envs/main/bin/activate
python scripts/download_weights.py         # MiniMax-H3 + the reward models (or --rewards videoalign hpspp); --hyperflow adds the 8-step LoRA
bash tests/run_tests.sh                    # correctness tests, seconds on CPU (GPU=1 adds a kernel test)
```

Two environments are used because VideoAlign's reward model needs an older `transformers`; h3rl starts each reward
worker with the right interpreter (`H3RL_REWARD_PYTHON`).

## Train

```bash
python -m h3rl.launch configs/experiments/nft_mix.yaml
python -m h3rl.launch configs/experiments/sage_mix.yaml run.train_gpus=[4,5,6] run.reward_gpus=[7]
python -m h3rl.launch configs/experiments/nft_mix.yaml reward=configs/reward/videoalign.yaml run.iters=50
```

The launcher encodes prompts missing from the text cache, starts only the reward workers the reward needs, runs a
2-iteration smoke test (finite loss and gradients, every reward term present, memory headroom), then trains.
LoRA checkpoints, `metrics.jsonl` and the rollouts are written to `run.out`.

### Multiple nodes

Run the same experiment on every node with `run.nnodes`, `run.node_rank` and `run.master_addr` (node 0's private IP);
`scripts/launch_multinode.sh hosts.txt <experiment.yaml>` does this over ssh. Nodes must reach each other on all TCP
ports (on AWS: a security-group rule allowing all traffic from the group itself, which EFA needs anyway). Set
`run.efa: true` on instances with EFA interfaces to route the gradient all-reduce over EFA. Rewards run on every node
by default; `run.reward_service: http://<host>:8800` sends them to `python -m h3rl.rewards.service` on dedicated
reward nodes instead, so every training GPU trains.

Large runs (8+ nodes):
- Prompts follow one shuffled walk through the pool shared by all ranks, so an iteration repeats no prompt while the
  pool is large enough. Use a pool several times larger than the prompts per iteration (`world / ranks_per_group x
  prompts_per_step`); `prompts/t2va_pool_v2.jsonl` has ~2,000 5 s prompts.
- `algo.ranks_per_group` splits each group over that many ranks (fewer rollouts per rank, shorter iterations). It must
  divide `group_size`, the world size and `grad_accum`; each optimizer step then sees `ranks_per_group` x fewer samples
  per rank, so revisit the learning rate when comparing with `ranks_per_group: 1`.
- Keep rewards on each node (the default); one `run.reward_service` host has too few GPUs for many training nodes.
- `run.keep_videos_every` (default 5) keeps the rollout videos of every 5th iteration only (1: all, 0: none).
- `run.rendezvous_timeout_s` (default 3600) is how long nodes wait for each other at start.

The repo, `run.out` and the text cache may live on storage the nodes share (NFS, FSx). Node 0 writes `run.out`
(metrics, checkpoints) and node i > 0 writes `run.out/node<i>` (its reward queue, logs and rollouts), so nodes never
write the same files. When launching from a machine other than the nodes, set `H3RL_REPO` to the repo path on the nodes.

## Configs

An experiment file names an algorithm config and a reward config and sets data and run options:

```yaml
algo: configs/algo/nft.yaml            # or sage_grpo.yaml, flow_grpo.yaml
algo_overrides: {lr: 5.0e-5}           # optional: change algorithm settings for this experiment only
reward: configs/reward/mix_v1.yaml
model:
  hyperflow: ""                        # "" = base model; "default" or a path = 8-step HyperFlow-distilled model
  infer_steps: null                    # default 30 (base) or 8 (HyperFlow)
data:
  pool: prompts/example_pool.jsonl
  frames: [124, 243]                   # 17n+5 frames at 24 fps (5.2 s, 10.1 s), alternated per iteration
  size: [544, 960]
  frame_sizes: {243: [480, 832]}       # optional per-length canvas
run:
  out: runs/nft_mix
  iters: 30
  save_every: 5
  keep_recent: 3                       # older checkpoints are deleted, except every keep_every-th (default 50); 0 keeps all
  reward_timeout_s: 1800               # per iteration: missing rewards after this give that rollout no reward signal (KL only)
  train_gpus: [0, 1, 2]                # one rank per GPU
  reward_gpus: [3]                     # reward workers are spread over these
  port: 29761
  smoke: true
```

A reward config picks a combiner and weights reward terms. The default combiner, `weighted_z`, z-scores each term over
a running window, averages the weighted terms per branch, and the algorithm then normalises within each group of
rollouts of the same prompt. Gates and floors are optional:

```yaml
combine: weighted_z
video: {va_ta: 1.0, soli_ta: 0.5, hps: 0.3, flow_motion: 0.5}   # negative weight = lower is better
audio: {}                                 # audio-branch terms (NFT trains the audio branch on these)
sync: {}                                  # added to both branches
gates: {cut_free: {min: 0}}               # outside the bound -> the rollout ranks last in its group
floors: {va_vq: 0.5}                      # penalise only drops below the term's start-of-training level
```

Terms come from reward workers (`h3rl/rewards/registry.py`). Per-video workers score each rollout; group workers
score all rollouts of a prompt together, e.g. `think_winrate`, a pairwise VLM judge's win rate with both presentation
orders (`configs/reward/pairwise_think.yaml`). The launcher starts exactly the workers the reward config needs.

| term | worker | measures |
|---|---|---|
| `va_vq`, `va_mq`, `va_ta` | videoalign | visual quality, motion quality, text alignment (VideoReward) |
| `hps`, `hps_min` | hpspp | frame aesthetics, mean and worst frame (HPSv3++) |
| `soli_ta`, `soli_phys` | solireward | text alignment, physics/deformity (SoliReward) |
| `ur_align`, `ur_physics`, `ur_style` | unifiedreward | VLM judge scores (UnifiedReward 2.0) |
| `flow_motion` | flowmotion | coherent optical-flow motion (ignores shake and flicker), CPU |
| `cut_free` | cutcheck | hard cuts in prompts that ask for one continuous shot, CPU |
| `think_winrate` | pairwise_think | group reward: pairwise win rate (UnifiedReward-Think) |

### Distilled (HyperFlow) model

With `model.hyperflow` set, the sampler uses HyperFlow's own 8-step sigma grid (read from the LoRA file), shifted per
modality as the official pipeline does (video shift 12, audio shift 3), with HyperFlow's two-time conditioning: each
step also embeds its endpoint (the next grid sigma) through the adapter's `endpoint_time_embedder`. NFT draws its training
noise levels around that grid and keeps each draw's endpoint at the next grid sigma. Rollouts are about 4x cheaper than
on the base model. Caution: RL or SFT directly on a step-distilled model can erode its few-step sharpness (the plain
flow-matching target is the blurry posterior mean), so compare frame quality across checkpoints. Runs before 2026-10-10
sampled HyperFlow on the unshifted grid and are not comparable.

Reward details: each term is z-scored over a running window (`z_window_iters: 16` iterations, kept per clip length,
the same on every rank) before weighting. A rollout with any black frame (mean luma < 10) or luma strobing on more than
5% of frame transitions counts as broken and ranks last in its group; watch `worst_frac` on prompts that fade to black.

## Held-out check

```bash
bash scripts/heldout_eval.sh runs/heldout 0,1,2,3 runs/nft_hf_mix/nft-00030.safetensors runs/nft_hf_mix/nft-00060.safetensors
```

Generates the 32 prompts of `prompts/heldout_rl.jsonl` (5 s, HyperFlow 8 steps, one fixed noise per prompt) with the
base model and with each checkpoint, then scores every checkpoint against base with the reward workers, paired by prompt
(`h3rl/eval/reward_check.py`): per-term means, difference, win rate and a permutation p-value, for the trained terms
(`va_ta`, `soli_ta`, `hps`, `flow_motion`) and the quality checks (`va_vq`, `va_mq`, `hps_min`, `soli_phys`, flow,
cuts). HPSv3++ is scored without its RL-progress ramp. Tables go to `<out>/report.txt`, per-video scores to
`<out>/scores_base_<ckpt>.json`. The prompts are rows of `example_pool.jsonl`, so later in a run they have been trained
on too; use a fresh output directory per run, since arms are named after the checkpoint file.

## Prompts

A prompt pool is JSONL, one prompt per line:

```json
{"pid": "t2va000", "prompt": "...", "task": "t2va", "image": null, "has_audio": true,
 "reward_prompt": "optional text the reward models read instead (e.g. an English translation)",
 "min_frames": 124, "max_frames": 124}
```

`task` is `t2va` (text to video+audio) or `fl2va` (first frame to video+audio; `image` is the frame path).
`prompts/t2va_pool_v2.jsonl` has 2,076 text-to-video prompts for 5 s clips (`prompts/t2va_SPEC.md`: 8 subject
categories, 14 camera moves, places worldwide, a sound sentence kept out of `reward_prompt` because the video judges
cannot hear it); its 64 held-out prompts, with no shared subjects, are `prompts/t2va_heldout_v2.jsonl`.
`prompts/example_pool.jsonl` has 200 text-to-video prompts for 5 s clips and 92 multi-shot, timecoded prompts for
10 s clips (`prompts/complexshot_SPEC.md` describes their format; 20 more are held out in `complexshot_heldout.jsonl`).

## Extending

**An algorithm.** Subclass `Algorithm` in `h3rl/algos/<name>.py`, register it, and add `configs/algo/<name>.yaml`
(its keys reach the class unchanged as `self.a`). The trainer handles prompts, clip-length buckets, conditioning,
decoding, rewards, metrics and checkpoints; the algorithm only samples and updates:

```python
from h3rl.algos.base import Algorithm, register

@register("my_algo")
class MyAlgo(Algorithm):
    def rollout(self, ctx, prompt, seed):      # one group for one prompt -> [{"video": latents, "audio": latents, ...}]
        ...
    def update(self, samples, iteration):      # samples carry R (combined reward) and group (prompt index) -> metrics
        ...
```

Then import the module in `h3rl/algos/__init__.py`. `nft.py` (ODE rollouts, reward-weighted flow matching) and
`grpo.py` (SDE rollouts with recorded log-probs, clipped-ratio policy gradient) are complete examples.

**A per-video reward.** Subclass `Worker` in `h3rl/rewards/workers/<name>.py`: `load()` once, `score(requests)`
returns `[{"key", "scores": {term: value}}]`. Register the worker and its terms in `h3rl/rewards/registry.py`
(which environment it runs in, which Hugging Face weights and code repos `download_weights.py` should fetch).

**A group reward.** Subclass `GroupWorker` instead and implement `score_groups(requests)`; each request carries
every rollout of one prompt (`members`), and the result gives each member its own terms. Register it with
`kind="group"`. `pairwise_think.py` is a complete example.

**A combiner.** Subclass `Combiner` in `h3rl/rewards/combine.py` and add it to `COMBINERS`.

## Roadmap

- [ ] **Evaluation** (`python -m h3rl.evaluate`): fixed held-out prompts with fixed per-prompt seeds, every checkpoint
      paired against the base model; judges taken from the reward registry, with a check that refuses judges whose
      terms are part of the run's training reward; reward-free checks (optical flow and coherence, cuts vs requested
      cuts, frame-to-frame DINO consistency, black-frame/flicker guards); bootstrap confidence intervals per prompt
      family and clip length; `report.md` + `metrics.json`.
- [ ] **In-training eval and stop rules**: `run.eval_every` in the experiment config, with configurable rules
      (e.g. held-out alignment significantly below base, optical flow below 65% of base) that stop the run.
- [ ] **Human side-by-side**: exported HTML viewer (base vs checkpoint per prompt) with a blind A/B mode that saves
      choices and reports a human win rate with a confidence interval.
- [ ] **Held-out prompt sets**: split a 5 s text-to-video set out of the example pool (alongside
      `complexshot_heldout.jsonl` for 10 s) so evaluation prompts never appear in training.
- [ ] **Reference-to-video (`ref2va`)**: pool rows carry reference images (and optionally audio/video); the trainer
      builds the reference conditioning through the upstream `ref2va` path and loads the `minimax_h3_ref2va_bf16`
      checkpoint for those tasks; a source of real reference images for the complex-shot prompts (generated from the
      `refs` descriptions or user-provided).
- [ ] **Prompt generation** (`h3rl.data.generate`): write prompts from a spec such as `prompts/complexshot_SPEC.md`
      with an LLM API (user-supplied key).
- [ ] **Pool tools** (`h3rl.data.pool`): validate rows, deduplicate, split train/held-out, assign clip-length buckets,
      encode text.
- [ ] **More reward workers**: camera-trajectory geometry (per time segment), audio quality (Audiobox, CLAP) and
      audio-video sync, which need extra environments in `scripts/setup_env.sh`; VideoScore2 and the reasoning judges
      as per-video workers.
- [x] **Multi-node**: launch across nodes, local or HTTP reward service (`run.reward_service`), optional EFA (`run.efa`).
      Tested as two nodes on one machine; a real cross-node run and EFA are still to verify.

## Layout

```
h3rl/launch.py          entry point: config -> encode -> reward workers -> smoke -> train
h3rl/train.py           shared trainer: prompts, buckets, rollouts, decoding, rewards, metrics, checkpoints
h3rl/algos/             base.py (interface + registry), nft.py (DiffusionNFT), grpo.py (Flow-GRPO / SAGE-GRPO)
h3rl/core/              model loading and context building, samplers (ODE/SDE), NFT and GRPO losses, DDP helpers
h3rl/rewards/           combine.py (combiners), registry.py, queue.py, worker_base.py (Worker, GroupWorker), workers/
h3rl/data/encode.py     text-encoder cache for a prompt pool
h3rl/eval/              held-out generation and scoring
configs/                algo/, reward/, experiments/
scripts/                setup_env.sh, download_weights.py, launch_multinode.sh, heldout_eval.sh
envs/                   pinned package versions for the two environments
tests/                  correctness tests
```

## License

Apache-2.0. See `NOTICE` for the upstream model code and the reward models, which have their own licenses.
