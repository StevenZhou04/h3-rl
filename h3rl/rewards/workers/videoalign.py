"""VideoAlign (KwaiVGI/VideoReward) VQ/MQ/TA as a queue worker. venv: rm/venv (transformers 4.45.2)."""
from h3rl.paths import H3_RM
import os, sys, torch
from h3rl.rewards.worker_base import Worker
import datasets  # real package must load before VideoAlign's local ./datasets dir
sys.path.append(f"{H3_RM}/VideoAlign")


class VA(Worker):
    name = "videoalign"; batch = 1   # eager attention: 4 x 10 s clips = ~100 GB

    def load(self):
        import inference
        _TC = inference.TrainingConfig
        inference.TrainingConfig = lambda **kw: _TC(**{**kw, "disable_flash_attn2": True})
        self.m = inference.VideoVLMRewardInference(f"{H3_RM}/VideoReward", device="cuda", dtype=torch.bfloat16)

    def score(self, reqs):
        with torch.no_grad():
            rw = self.m.reward([r["mp4"] for r in reqs], [r["prompt"] for r in reqs], use_norm=True)
        return [{"key": r["key"], "scores": {"va_vq": float(s["VQ"]), "va_mq": float(s["MQ"]), "va_ta": float(s["TA"])}} for r, s in zip(reqs, rw)]


if __name__ == "__main__": VA().run()
