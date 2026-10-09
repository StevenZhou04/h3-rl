"""HPSv3++ (Qwen3-VL-8B) frame-level human preference score. venv: rm/venv_hps.
score = mean over N frames of the mu head; iter_step from meta['rl_progress'] (0.3..1.0 during RL, 0 for eval)."""
from h3rl.paths import H3_RM
import os, sys, numpy as np
from h3rl.rewards.worker_base import Worker, sample_frames
REPO = f"{H3_RM}/HPSv3-PlusPlus"; CKPT = f"{H3_RM}/hpsv3pp/hpsv3++.pth"


class HPSPP(Worker):
    name = "hpspp"; batch = 4; n_frames = 4

    def load(self):
        os.chdir(REPO); sys.path.insert(0, REPO)
        from hpsv3.inference import HPSv3RewardInferencer
        self.m = HPSv3RewardInferencer(config_path=f"{REPO}/hpsv3/config/train_stage2.yaml", checkpoint_path=CKPT, device="cuda")

    def score(self, reqs):
        out = []
        for r in reqs:
            frames = sample_frames(r["mp4"], self.n_frames, f"{self.tmp}/{r['key']}")
            prog = float(r.get("meta", {}).get("rl_progress", 0.0)); it = 0.0 if prog <= 0 else 0.3 + 0.7 * prog
            rw = self.m.reward(prompts=[r["prompt"]] * len(frames), image_paths=frames, iter_step=it)
            mus = [float(x[0].item()) for x in rw]
            out.append({"key": r["key"], "scores": {"hps": float(np.mean(mus)), "hps_min": float(np.min(mus))}})
        return out


if __name__ == "__main__": HPSPP().run()
