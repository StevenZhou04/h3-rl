"""UnifiedReward-2.0-qwen3vl-8b pointwise video scoring (Alignment / Physics / Style, 1-5), run with
transformers in the main musubi venv (Qwen3-VL supported). As the repo's point_score_APS_video_generation.py: 16
uniformly sampled frames at native resolution and its prompt (verbatim)."""
from h3rl.paths import H3_RM
import os, re, sys, torch
from h3rl.rewards.worker_base import Worker, sample_frames
from PIL import Image
MODEL = f"{H3_RM}/UnifiedReward-2.0-qwen3vl-8b"
PROMPT = ("You are presented with a generated video and its associated text caption. "
          "Your task is to analyze the video across multiple dimensions in relation to the caption. Specifically:\n"
          "Provide overall assessments for the video along the following axes (each rated from 1 to 5):\n"
          "- Alignment Score: How well the video matches the caption in terms of content.\n"
          "- Physics Score: How well the gravity, movements, collisions, and interactions make physical sense.\n"
          "- Style Score: How visually appealing the video looks, regardless of caption accuracy.\n\n"
          "Output your evaluation using the format below:\n\n"
          "Alignment Score (1-5): X\nPhysics Score (1-5): Y\nStyle Score (1-5): Z\n\n"
          "Your task is provided as follows:\nText Caption: [{prompt}]")


class UR(Worker):
    name = "unifiedreward"; batch = 4; n_frames = 16

    def load(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.p = AutoProcessor.from_pretrained(MODEL); self.p.tokenizer.padding_side = "left"

    def score(self, reqs):
        msgs, imgs, bad, good = [], [], [], []
        for r in reqs:
            try: fr = [Image.open(f).convert("RGB") for f in sample_frames(r["mp4"], self.n_frames, f"{self.tmp}/{r['key']}", short_side=None)]
            except Exception as e: bad.append({"key": r["key"], "scores": {}, "error": f"{type(e).__name__}: {e}"}); continue   # this clip only
            good.append(r)
            content = [{"type": "image", "image": im} for im in fr] + [{"type": "text", "text": PROMPT.format(prompt=r["prompt"])}]
            msgs.append([{"role": "user", "content": content}]); imgs.append(fr)
        if not good: return bad
        texts = [self.p.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = self.p(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.m.device)
        with torch.no_grad():
            gen = self.m.generate(**inp, max_new_tokens=64, do_sample=False)
        dec = self.p.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out = []
        for r, txt in zip(good, dec):
            sc = {}
            for name, key in (("Alignment", "ur_align"), ("Physics", "ur_physics"), ("Style", "ur_style")):
                m = re.search(name + r" Score \(1-5\):\s*([0-9.]+)", txt)
                if m: sc[key] = float(m.group(1))
            out.append({"key": r["key"], "scores": sc, "error": None if len(sc) == 3 else f"parse: {txt[:120]!r}"})
        return out + bad


if __name__ == "__main__": UR().run()
