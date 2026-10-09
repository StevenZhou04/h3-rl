"""VideoAlign (KwaiVGI/VideoReward) VQ / MQ / TA scores for every row of an eval manifest.
Runs in the separate rm venv (transformers 4.45.2). Output: <out_dir>/scores_videoalign.jsonl"""
from h3rl.paths import H3_RM
import argparse, json, os, sys, torch
import datasets  # the real package must win over VideoAlign's local ./datasets folder, so import it first
sys.path.append(f"{H3_RM}/VideoAlign")   # append, not insert: keeps site-packages ahead of the repo dir
import inference
_TC = inference.TrainingConfig
inference.TrainingConfig = lambda **kw: _TC(**{**kw, "disable_flash_attn2": True})   # no flash_attn in this venv; SDPA is fine for scoring
VideoVLMRewardInference = inference.VideoVLMRewardInference
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, help="eval_gen output dir"); ap.add_argument("--batch", type=int, default=4); a = ap.parse_args()
rows = [json.loads(l) for l in open(f"{a.run}/manifest.jsonl")]
outp = f"{a.run}/scores_videoalign.jsonl"; done = {json.loads(l)["key"] for l in open(outp)} if os.path.exists(outp) else set()
rows = [r for r in rows if r["key"] not in done]; print(f"{len(rows)} videos to score", flush=True)
inf = VideoVLMRewardInference(f"{H3_RM}/VideoReward", device="cuda", dtype=torch.bfloat16)
out = open(outp, "a")
with torch.no_grad():
    for i in range(0, len(rows), a.batch):
        b = rows[i:i + a.batch]
        rw = inf.reward([r["mp4"] for r in b], [r["prompt"] for r in b], use_norm=True)
        for r, s in zip(b, rw):
            out.write(json.dumps(dict(key=r["key"], VQ=round(s["VQ"], 4), MQ=round(s["MQ"], 4), TA=round(s["TA"], 4))) + "\n"); out.flush()
        if i % 40 == 0: print(f"{i+len(b)}/{len(rows)}", flush=True)
print("DONE", flush=True)
