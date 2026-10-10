"""HyperFlow sampling contract (CPU, seconds): the sigma grids match the official pipeline, and the two-time hook gives
every target row the endpoint of its own modality's step.
 (a) make_schedule shifts the adapter's raw grid per modality (video 12, audio 3), as hyperflow_h3 HyperFlowSetTimestepsStep;
 (b) on-grid steps: video / text rows get r = 1 - next video sigma, audio rows r = 1 - next audio sigma;
 (c) step 0 (video and audio both at t = 0): the rows stay separate, each with its own endpoint;
 (d) inside hyperflow.endpoints(): an off-grid sigma (NFT jitter, above or below its grid point) keeps the step's endpoint;
 (e) condition rows (t = 0.999) keep r = t."""
import sys, torch
from h3rl.core import hyperflow
from h3rl.core.grpo import GRPOConfig, make_schedule
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

RAW = [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]
official = lambda grid, k: [k * x / (1 + (k - 1) * x) for x in grid]   # hyperflow_h3.schedule.shift_sigmas, written out
VIDEO, AUDIO = official(RAW, 12.0), official(RAW, 3.0)                     # [1, .9939, .9843, .9661, .9231, .8349, .6969, .4688, 0]
hf = dict(sigmas=RAW, steps=8, video_shift=12.0, audio_shift=3.0)
cfg = GRPOConfig(); cfg.infer_steps = 8
sch = make_schedule(cfg, torch.device("cpu"), hf)
check("(a) video grid == official shift 12", torch.allclose(sch.video, torch.tensor(VIDEO, dtype=torch.float64), atol=1e-5), str([round(float(x), 4) for x in sch.video]))
check("(a) audio grid == official shift 3", torch.allclose(sch.audio, torch.tensor(AUDIO, dtype=torch.float64), atol=1e-5), str([round(float(x), 4) for x in sch.audio]))

class Fake(torch.nn.Module):          # embeds the unique row times like musubi's build_timestep_rows; identity embedders
    def __init__(self): super().__init__(); self.time_embedder = torch.nn.Identity()
    def forward(self, *, model_t_video, model_t_audio, cond=0.999):
        ts = sorted({float(model_t_video), float(model_t_audio), cond}); out = self.time_embedder(torch.tensor(ts))
        r = {t: float(t + (o - t) / G) for t, o in zip(ts, out)}          # out = t + G (r - t) -> recover r per unique row
        return r[float(model_t_video)], r[float(model_t_audio)], r[cond], ts   # forward sees the hook-adjusted audio time
G = 0.25; m = Fake(); hyperflow.install_two_time(m, G); hyperflow.bind_schedule(m, sch)
def run(sv, sa): return m(model_t_video=1 - sv, model_t_audio=1 - sa)
ok = True
for i in range(1, 8):
    rv, ra, rc, _ = run(VIDEO[i], AUDIO[i]); ok &= abs(rv - (1 - VIDEO[i + 1])) < 1e-5 and abs(ra - (1 - AUDIO[i + 1])) < 1e-5 and abs(rc - 0.999) < 1e-6
check("(b) on-grid steps: each modality's own next sigma; (e) condition rows r = t", ok)
rv, ra, rc, ts = run(1.0, 1.0)
check("(c) step 0: video and audio rows stay separate with their own endpoints", len(ts) == 3 and abs(rv - (1 - VIDEO[1])) < 1e-5 and abs(ra - (1 - AUDIO[1])) < 1e-5,
      f"rows {ts} r_video {rv:.5f} (want {1 - VIDEO[1]:.5f}) r_audio {ra:.5f} (want {1 - AUDIO[1]:.5f})")
ok = True
for i in range(1, 7):
    for frac in (-0.3, 0.3):
        sv = VIDEO[i] + frac * min(VIDEO[i - 1] - VIDEO[i], VIDEO[i] - VIDEO[i + 1]); sa = AUDIO[i] + frac * min(AUDIO[i - 1] - AUDIO[i], AUDIO[i] - AUDIO[i + 1])
        with hyperflow.endpoints(m, VIDEO[i + 1], AUDIO[i + 1]): rv, ra, _, _ = run(sv, sa)
        ok &= abs(rv - (1 - VIDEO[i + 1])) < 1e-5 and abs(ra - (1 - AUDIO[i + 1])) < 1e-5
check("(d) jittered sigmas (above and below the grid point) keep the step's endpoint", ok)
rv, _, _, _ = run(VIDEO[2] + 0.3 * (VIDEO[1] - VIDEO[2]), AUDIO[2])
check("(d') without endpoints() an upward-jittered sigma would land on its own grid point (why NFT passes them)", abs(rv - (1 - VIDEO[2])) < 1e-5)
if FAILS: print("FAILED:", FAILS); sys.exit(1)
print("ALL PASS")
