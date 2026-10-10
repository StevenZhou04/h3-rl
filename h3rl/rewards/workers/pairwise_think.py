"""Pairwise VLM judge as a group reward (Pref-GRPO style): every rollout of a group is compared with others by
UnifiedReward-Think (Qwen3-VL-8B, the authors' pairwise prompt, 8 frames per video) in BOTH orders, which cancels the
judge's strong preference for whichever video is shown first. Term: think_winrate in [0, 1] = wins / comparisons.
Cost grows with group size: PAIRS_PER_MEMBER (env, default 3) opponents per rollout, two orders each.
  python -m h3rl.rewards.workers.pairwise_think --queue Q"""
from __future__ import annotations
import itertools, os, random, re
import cv2, numpy as np, torch
from PIL import Image
from h3rl.paths import H3_RM
from h3rl.rewards.worker_base import GroupWorker

MODEL = f"{H3_RM}/UnifiedReward-Think-qwen3vl-8b"
# verbatim from the authors' video pairwise script (CodeGoat24/UnifiedReward, UnifiedReward-Think inference)
PROMPT = '''You are an objective and precise evaluator for video quality comparison. I will provide you with a text caption and a sequence of consecutive frames extracted from two generated videos based on this caption. The first half of the frames belong to Video 1, and the second half of the frames belong to Video 2. You must analyze these two videos carefully and determine which video is better.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Video 1 is better
        - Video 2 is better
        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated videos is: 「{prompt}」. The provided frames represent two candidate videos:
        - First half: Video 1
        - Second half: Video 2

        2. You must evaluate the two videos across these core dimensions:
        - Semantic consistency (how closely the video content aligns with the caption)
        - Temporal coherence (smoothness and logical flow of motion across frames)
        - Authenticity (realism and attention to detail)

        3. You may also add up to two additional evaluation dimensions if they are clearly relevant (e.g., camera stability, lighting consistency, creativity). If no extra dimensions are relevant, keep only the three core dimensions.

        4. For each evaluation dimension:
        - Provide a score between 1–10 for both Video 1 and Video 2.
        - Provide a short rationale for each score (2–5 short sentences).
        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:
            N. Dimension name: 
                Video 1 (x/10) - rationale; 
                Video 2 (y/10) - rationale

        5. After evaluating all dimensions, calculate the total score for each video and show the calculation explicitly, following this exact format:
            Total score:
            Video 1: x+x+x(+...)=total
            Video 2: y+y+y(+...)=total

        6. All reasoning, analysis, scoring, and totals must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Semantic consistency: 
            Video 1 (9/10) - ...; 
            Video 2 (7/10) - ...
        2. Temporal coherence: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        3. Authenticity: 
            Video 1 (7/10) - ...; 
            Video 2 (5/10) - ...
        [Additional dimension if any]: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        [Additional dimension if any]: 
            Video 1 (7/10) - ...; 
            Video 2 (7/10) - ...
        Total score:
        Video 1: 9+8+7+8+7=39
        Video 2: 7+6+5+6+7=31
        </think>
        <answer>Video 1 is better</answer>

        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and the two provided videos (frames divided into two halves).
        '''


def frames(path, n=8):
    """n uniformly spaced frames at native resolution, as the repo's qwen3_infer_cot_video_generation.py; raises if any
    frame cannot be read (the prompt's "first half = Video 1" split must match the images)."""
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    if N <= 0: cap.release(); raise ValueError(f"video has no frames: {path}")
    for i in np.linspace(0, N - 1, n).astype(int):
        for j in range(int(i), max(int(i) - 4, -1), -1):          # the container's frame count can overcount by a frame or two
            cap.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = cap.read()
            if ok: break
        if not ok: cap.release(); raise ValueError(f"failed reading frame {i} of {path}")
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release(); return out


class PairwiseThink(GroupWorker):
    name = "pairwise_think"

    def load(self):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        cfg = AutoConfig.from_pretrained(MODEL); tc = getattr(cfg, "text_config", None)
        if tc is not None and getattr(tc, "rope_scaling", None) is None and getattr(tc, "rope_parameters", None):   # saved by transformers 5.x
            tc.rope_scaling = dict(tc.rope_parameters); tc.rope_theta = tc.rope_scaling.get("rope_theta", 5e6)
        self.m = AutoModelForImageTextToText.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.p = AutoProcessor.from_pretrained(MODEL); self.p.tokenizer.padding_side = "left"
        self.k = int(os.environ.get("PAIRS_PER_MEMBER", "3")); self.bs = int(os.environ.get("JUDGE_BATCH", "4"))

    def _judge(self, jobs, prompt, fr):
        """jobs: [(a, b)] member keys; returns list of winners (key or None)."""
        out = []
        for i in range(0, len(jobs), self.bs):
            chunk = jobs[i:i + self.bs]; msgs, imgs = [], []
            for a, b in chunk:
                ims = fr[a] + fr[b]; imgs.append(ims)
                msgs.append([{"role": "user", "content": [{"type": "image", "image": x} for x in ims] + [{"type": "text", "text": PROMPT.format(prompt=prompt)}]}])
            texts = [self.p.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = self.p(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.m.device)
            with torch.no_grad(): g = self.m.generate(**inp, max_new_tokens=int(os.environ.get("THINK_MAX_TOKENS", "4096")), do_sample=False)
            for (a, b), seq in zip(chunk, g[:, inp["input_ids"].shape[1]:]):
                m = re.search(r"<answer>\s*Video\s*([12])", self.p.tokenizer.decode(seq, skip_special_tokens=True))
                out.append(None if not m else (a if m.group(1) == "1" else b))
        return out

    def score_groups(self, requests):
        res = []
        for r in requests:
            keys = [m["key"] for m in r["members"]]; fr = {m["key"]: frames(m["mp4"]) for m in r["members"]}
            rng = random.Random(r["key"]); pairs = set()
            for a in keys:                                   # PAIRS_PER_MEMBER random opponents per rollout
                for b in rng.sample([x for x in keys if x != a], min(self.k, len(keys) - 1)): pairs.add(tuple(sorted((a, b))))
            jobs = [p for a, b in sorted(pairs) for p in ((a, b), (b, a))]
            wins = {k: 0 for k in keys}; games = {k: 0 for k in keys}
            for (a, b), w in zip(jobs, self._judge(jobs, r["prompt"], fr)):
                if w is None: continue
                games[a] += 1; games[b] += 1; wins[w] += 1
            res.append({"key": r["key"], "members": {k: ({"think_winrate": wins[k] / games[k]} if games[k] else {}) for k in keys}})
        return res


if __name__ == "__main__": PairwiseThink().run()
