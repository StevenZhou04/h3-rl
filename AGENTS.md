# Notes for coding agents

## Evaluating a training run

When asked to evaluate a run (e.g. the 64-GPU NFT run), follow "Evaluating a run" in README.md. Rules:

- Do not stop, restart or modify a running training job, and do not change the checkout it runs from. Work in a
  separate worktree: `git worktree add ../h3-rl-eval nft-64g-v1-eval`, same Python env as training.
- Step 2 needs 8 free GPUs of 141 GB+. Check `nvidia-smi` first; never take GPUs a training run is using.
- Step 3 is a human judgment. Prepare the pairs, then hand over to the person: tell them where the `pair_NN.mp4` files
  and `picks.txt` are. Do not open `.key.json`, do not fill in `picks.txt`, and do not describe or compare the videos
  of a pair yourself before the reveal. Run `--reveal` only after the person has filled in `picks.txt`.
- Results leave this machine only as text: no checkpoints, videos or files are sent anywhere.

Report back in this exact format, so the result can be read out:

```
RUN <run dir>, <iterations> iterations, checkpoints evaluated: <MID>, <LAST>
1 TREND (first / mid / last)
  grad_norm a / b / c ; R_video_std a / b / c ; loss_kl a / b / c
  va_vq, va_mq, va_ta, hps, soli_ta, flow_motion: a / b / c each
2 HELDOUT vs base (diff, B-win rate, p) for each checkpoint
  va_vq, va_mq, va_ta, hps, flow_motion, soli_ta, n_cuts
3 BLIND (<LAST> vs base, n pairs): RL wins x / base wins y / ties z, win rate w, p
  most common reasons for RL wins: ... ; for base wins: ...
4 FLAGS (yes/no): brighter/saturated, jittery, static, flicker/artifacts, faces/hands worse, hard cuts
```

Round to 2 significant digits. Report errors from any step (e.g. reward worker errors, failed generations) verbatim.
