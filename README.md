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

## Configs

An experiment file names an algorithm config and a reward config and sets data and run options:

```yaml
algo: configs/algo/nft.yaml            # or sage_grpo.yaml, flow_grpo.yaml
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

With `model.hyperflow` set, the sampler uses HyperFlow's own 8-step sigma grid (read from the LoRA file), NFT draws its
training noise levels from that grid, and rollouts are about 4x cheaper than on the base model. Two cautions: RL or SFT
directly on a step-distilled model can erode its few-step sharpness (the plain flow-matching target is the blurry
posterior mean), so compare frame quality across checkpoints; and the conversion drops HyperFlow's
`endpoint_time_embedder`, which musubi's H3 lacks, so validate first-frame (`fl2va`) prompts before relying on them.

## Prompts

A prompt pool is JSONL, one prompt per line:

```json
{"pid": "t2va000", "prompt": "...", "task": "t2va", "image": null, "has_audio": true,
 "reward_prompt": "optional text the reward models read instead (e.g. an English translation)",
 "min_frames": 124, "max_frames": 124}
```

`task` is `t2va` (text to video+audio) or `fl2va` (first frame to video+audio; `image` is the frame path).
`prompts/example_pool.jsonl` has 200 text-to-video prompts for 5 s clips and 130 multi-shot, timecoded prompts for
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
- [ ] **Multi-node**: launch across nodes and serve reward models over the network instead of the shared-disk queue.

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
scripts/                setup_env.sh, download_weights.py
envs/                   pinned package versions for the two environments
tests/                  correctness tests
```

## License

Apache-2.0. See `NOTICE` for the upstream model code and the reward models, which have their own licenses.
