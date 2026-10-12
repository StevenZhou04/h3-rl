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

### 64-GPU recipes

All on 8 nodes x 8 GPUs like `nft_hyperflow_64g.yaml` (mix_v1 reward), launched the same way
(`bash scripts/launch_multinode.sh hosts.txt configs/experiments/<recipe>.yaml`), each with its own `run.out` and port.
Best-effort order:

| Recipe | Changes vs nft_hyperflow_64g | Time | Question it answers |
|---|---|---|---|
| `nft_hf_64g_sanity_bright` | reward = mean frame brightness only | ~2 h, 20 it | Does the loop learn a target it certainly can? If videos are not clearly brighter, stop: the trainer is broken. |
| `nft_hf_64g_quality` | reward `quality_v2`: va_ta 1, va_vq 0.5, flow_motion 0.3; floors va_vq, va_mq; hard-cut gate | 1 day | Does a sharper reward give a visible held-out win? |
| `nft_hf_64g_quality_kl` | quality_v2 + kl_beta 1e-2 + timesteps_per_sample 2 | 1 day, 175 it | Does anchoring to HyperFlow and more signal per rollout reduce drift? Check loss_kl vs loss_policy early. |
| `nft_hf_64g_pairwise` | reward `pairwise_v2`: pairwise VLM win rate 1, va_ta 0.5, flow_motion 0.2; va_vq floor; gate | 1 day | Do relative judgments beat absolute scores? Run reward_sanity.py on pairwise_think first. |

Evaluate each with "Evaluating a run" below.

### Reward ablation

Which judges are worth training on, alone or combined? `configs/experiments/ablation/` holds 10 one-day runs, each on its own 64-GPU
cluster (the `nft_hyperflow_64g` setup with only the reward changed; the `think_*` runs keep groups on one rank, 130
iterations), all in parallel. The waves are only a priority order if fewer clusters are free:

| Wave | Runs (reward) |
|---|---|
| 1 | `va_ta`, `va_vq`, `hps`, `think_winrate` (each alone) |
| 2 | `soli_ta`, `flow_motion` (alone); `va_ta_vq` (va_ta + va_vq); `va_all` (va_ta + 0.5 va_vq + 0.5 va_mq) |
| 3 | `va_ta_hps` (va_ta + 0.3 hps); `think_va_ta` (think_winrate + 0.5 va_ta) |

The references are runs above, not duplicated here: `nft_hyperflow_64g` (the current mix_v1 reward) and
`nft_hf_64g_quality` (quality_v2).

Launch each on its cluster: `bash scripts/launch_multinode.sh hosts.txt configs/experiments/ablation/<name>.yaml`
(outputs `runs/abl_<name>`, distinct ports). With the 4 recipes above that is 14 runs: 13 one-day runs and the ~2 h
brightness sanity run (plus the current `nft_hyperflow_64g` run). Run `scripts/reward_sanity.py --workers pairwise_think` before the
`think_*` runs. Then, on one free 8-GPU node:

```bash
bash scripts/ablation_eval.sh runs/heldout_abl 0,1,2,3,4,5,6,7      # every runs/abl_* and runs/nft_hf_64g* (not the brightness run) at its last iteration;
# rerun to add runs that finish later. Runs on other clusters: copy or mount their config.json and final
# nft-*.safetensors into runs/<run name>/ first
```

It generates the 64 held-out prompts once with base and once per run (arms are named after the run), scores them, and
prints one table (`runs/heldout_abl/ablation_table.txt`): a row per run, a column per judge, each cell the held-out
change vs base in base-SD units, `*` for p < 0.05, `[ ]` around the terms the run was trained on. The `nft_hf_64g` row (mix_v1) and the `nft_hf_64g_quality` row are the references. A good reward raises
its own column *and* does not lower the others (especially `va_vq`, `va_mq`); a column that rises only for the run
trained on it is a judge being gamed. Blind-check the 2-3 best rows: `python -m h3rl.eval.blind_pairs --dir
runs/heldout_abl --arms base <name> --n 32` (see "Evaluating a run").

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

## Evaluating a run

Four steps, in order of weight: the blind human check (3) decides; the other steps explain the result.

1. **Training trend:** `python scripts/run_summary.py <run>/metrics.jsonl` prints the mean of every reward axis and of the
   learning-signal metrics (`grad_norm`, `R_video_std`, `loss_kl`) over the first, middle and last ~10% of the iterations.
2. **Held-out rewards vs base** (8 GPUs of 141 GB+, about an hour; use a mid-run and the last checkpoint):
   `HELDOUT_SET=prompts/t2va_heldout_v2.jsonl bash scripts/heldout_eval.sh runs/heldout 0,1,2,3,4,5,6,7 <run>/nft-000MID.safetensors <run>/nft-000LAST.safetensors`
   (see Held-out check).
3. **Blind side-by-side** (`h3rl/eval/blind_pairs.py`) on the videos of step 2:
   `python -m h3rl.eval.blind_pairs --dir runs/heldout --arms base nft-000LAST --n 32` writes `pair_NN.mp4` (base and
   checkpoint side by side, sides random) and `picks.txt` to `runs/heldout/blind_base_nft-000LAST/`. A person watches
   every pair and writes `A` (left better), `B` (right better) or `=` after its number, optionally with a reason
   (`07 B sharper hands`), without opening `.key.json`; then the same command with `--reveal` prints the wins, losses,
   ties, the win rate and a sign test, and the reasons per outcome. The eval videos have no audio.
4. **Reward-hacking flags**, noted while watching (RL vs base): brighter or more saturated; jittery or shaky; more
   static; flicker or texture artifacts; worse faces or hands; hard cuts.

Reading the result: an RL win rate of 0.65 or more with p < 0.05 and consistent reasons means RL helps; 0.4-0.6 means
no visible difference (the rewards do not track what people see); below 0.4, or any hacking flag, means RL hurts. If
`hps` or `flow_motion` rise in step 2 while `va_vq` or `va_mq` fall, quality is being traded for motion or aesthetics.
`R_video_std` near 0, or a tiny flat `grad_norm`, in step 1 means little learning signal: changes are drift.

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
h3rl/eval/              held-out generation, reward scoring (reward_check.py), blind side-by-side (blind_pairs.py)
configs/                algo/, reward/, experiments/
scripts/                setup_env.sh, download_weights.py, launch_multinode.sh, heldout_eval.sh, run_summary.py, ablation_eval.sh
envs/                   pinned package versions for the two environments
tests/                  correctness tests
```

## License

Apache-2.0. See `NOTICE` for the upstream model code and the reward models, which have their own licenses.
