"""KL logging (CPU, seconds): the NFT loss is reported as loss = loss_policy + loss_kl, where loss_kl is the weighted KL
term actually added to the loss; kl_video / kl_audio are the raw |v_theta - v_base|^2; NFT.update logs all of them."""
import sys, random, types, torch
import h3rl.core.nft as T
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

w = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
class Net:
    m = 1.0
    def set_multiplier(self, m): self.m = m
net = Net()
def transformer(video_latents, audio_latents, **kw):       # LoRA on: v depends on w; LoRA off (multiplier 0): base velocity
    v = video_latents.float() * 0.5 + net.m * w[0]; a = audio_latents.float() * 0.5 + net.m * w[1]
    return types.SimpleNamespace(video=v, audio=a)
torch.manual_seed(0)
sample = {"video": torch.randn(2, 3), "audio": torch.randn(4), "vis_cond": (), "aud_cond": ()}
ctx = {"layout": None, "text_hidden_states": None, "text_token_tags": None, "visual_condition_clean": None}
old = [torch.tensor([0.25, -0.1])]

def run(kl_beta, aw=0.5):
    cfg = T.NFTConfig(); cfg.kl_beta = kl_beta; cfg.audio_loss_weight = aw
    w.grad = None
    p = T._nft_loss_at(transformer, net, [w], old, sample, ctx, 0.7, 0.4, cfg, torch.device("cpu"), random.Random(1), 1.0, 0.6, 0.5)
    return p, cfg

p, cfg = run(0.5)
kl_w = cfg.kl_beta * cfg.beta / cfg.adv_clip_max
check("(a) loss = loss_policy + loss_kl", abs(p["loss"] - p["loss_policy"] - p["loss_kl"]) < 1e-6, f"{p['loss']:.6f} vs {p['loss_policy']:.6f} + {p['loss_kl']:.6f}")
check("(a) loss_kl = kl_w (kl_video + audio_loss_weight kl_audio)",
      abs(p["loss_kl"] - kl_w * (p["kl_video"] + cfg.audio_loss_weight * p["kl_audio"])) < 1e-7)
check("(a) raw KL matches |v_theta - v_base|^2 (toy: w^2)",
      abs(p["kl_video"] - 0.3 ** 2) < 1e-5 and abs(p["kl_audio"] - 0.2 ** 2) < 1e-5, f"{p['kl_video']:.5f} {p['kl_audio']:.5f}")
g_kl = w.grad.clone(); p0, _ = run(0.0)
check("(b) kl_beta = 0: loss_kl is 0, no KL keys, the policy part is unchanged",
      p0["loss_kl"] == 0.0 and "kl_video" not in p0 and abs(p0["loss_policy"] - p["loss_policy"]) < 1e-6)
check("(b) the KL term still reaches the gradient", not torch.allclose(g_kl, w.grad))

import h3rl.algos.nft as A
from h3rl.algos.base import TrainContext
class Tr:
    def train(self): pass
TC = TrainContext(models={}, transformer=Tr(), network=net, params=[w], schedule=None, device=torch.device("cpu"), rank=0, world=1,
                  rng=random.Random(0), infer_steps=8, canvas={"frames": 124, "height": 544, "width": 960})
algo = A.NFT({"lr": 0.01, "group_size": 4, "prompts_per_step": 1, "grad_accum": 2, "epochs_per_batch": 1}, TC)
vals = iter([0.1, 0.3, 0.5, 0.7])
def fake_loss(*a, **k):
    x = next(vals); loss = w.sum() * 0 + x; loss.backward()
    return {"loss": x + 0.01, "loss_policy": x, "loss_kl": 0.01, "kl_video": 2 * x, "kl_audio": 3 * x, "loss_video": x, "fm_video": 0.0}
A.nft_loss = fake_loss
samples = [{"group": 0, "gid": "0:0", "member": k, "prompt": {"pid": "p"}, "ctx": None, "R": {"video": float(k), "audio": None}} for k in range(4)]
mt = algo.update(samples, 0)
check("(c) NFT.update logs the breakdown, averaged over samples",
      abs(mt["loss_policy"] - 0.4) < 1e-9 and abs(mt["loss_kl"] - 0.01) < 1e-9 and abs(mt["kl_video"] - 0.8) < 1e-9
      and abs(mt["kl_audio"] - 1.2) < 1e-9 and "loss_audio" not in mt, f"{ {k: round(v, 4) for k, v in mt.items()} }")
if FAILS: sys.exit(1)
print("ALL PASS")
