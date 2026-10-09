"""Checkpoint pruning: keep the newest N plus every K-th, delete all three files of the others, ignore other names."""
import sys, tempfile
from pathlib import Path
from h3rl.core.dist import prune_checkpoints
d = Path(tempfile.mkdtemp())
for s in range(5, 125, 5):
    for ext in (".safetensors", ".state.json", ".train.pt"): (d / f"nft-{s:05d}{ext}").write_text("x")
(d / "nft-00010-fatal.safetensors").write_text("x"); (d / "grpo-00005.safetensors").write_text("x")
prune_checkpoints(d, "nft", keep_recent=3, keep_every=50)
left = sorted(p.name for p in d.iterdir())
want = sorted([f"nft-{s:05d}{e}" for s in (50, 100, 110, 115, 120) for e in (".safetensors", ".state.json", ".train.pt")]
              + ["nft-00010-fatal.safetensors", "grpo-00005.safetensors"])
if left != want: print("FAIL", left); sys.exit(1)
prune_checkpoints(d, "nft", keep_recent=0, keep_every=50); assert sorted(p.name for p in d.iterdir()) == want   # 0 keeps all
print("ALL PASS")
