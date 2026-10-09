"""Local judge-free metrics per eval video (our venv): DINOv2 temporal consistency, first-frame
fidelity for fl2va (DINO cosine + LPIPS vs the condition image), plus the guardrails already in the
manifest. Output: <run>/scores_local.jsonl"""
from h3rl.paths import H3_RM
import argparse, json, os, cv2, numpy as np, torch
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
rows = [json.loads(l) for l in open(f"{a.run}/manifest.jsonl")]
outp = f"{a.run}/scores_local.jsonl"; done = {json.loads(l)["key"] for l in open(outp)} if os.path.exists(outp) else set()
rows = [r for r in rows if r["key"] not in done]; print(f"{len(rows)} videos to score", flush=True)
from transformers import AutoImageProcessor, AutoModel
import lpips
dev = "cuda"
dproc = AutoImageProcessor.from_pretrained(f"{H3_RM}/dinov2-base"); dino = AutoModel.from_pretrained(f"{H3_RM}/dinov2-base").to(dev).eval()
lp = lpips.LPIPS(net="alex").to(dev).eval()
def read_frames(path, n=16):
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idx = set(np.linspace(0, N - 1, n).round().astype(int).tolist()); fr = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i in idx: fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release(); return fr
@torch.no_grad()
def dino_feats(frames):
    x = dproc(images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(dev)
    return torch.nn.functional.normalize(dino(**x).last_hidden_state[:, 0], dim=-1)
def to_lp(img, hw):
    im = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(dev)
out = open(outp, "a")
for n, r in enumerate(rows):
    fr = read_frames(r["mp4"])
    if len(fr) < 2: continue
    f = dino_feats(fr)
    adj = (f[1:] * f[:-1]).sum(-1); rec = dict(key=r["key"], dino_adjacent=round(adj.mean().item(), 4), dino_first_last=round((f[0] * f[-1]).sum().item(), 4), dino_min_adjacent=round(adj.min().item(), 4))
    if r.get("task") == "fl2va" and r.get("image") and os.path.exists(r["image"]):
        cond = np.asarray(Image.open(r["image"]).convert("RGB")); hw = fr[0].shape[:2]
        fc = dino_feats([cv2.resize(cond, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)])
        rec["ff_dino"] = round((fc[0] * f[0]).sum().item(), 4)
        with torch.no_grad(): rec["ff_lpips"] = round(lp(to_lp(cond, hw), to_lp(fr[0], hw)).item(), 4)
    out.write(json.dumps(rec) + "\n"); out.flush()
    if n % 40 == 0: print(f"{n+1}/{len(rows)}", flush=True)
print("DONE", flush=True)
