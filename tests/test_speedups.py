"""Speed-ups that must not change results (CPU, seconds).
 (a) partial checkpointing: gradients equal fully-checkpointed and plain runs; marked blocks are not recomputed;
 (b) streamed NFT rollout: samples come from the old (EMA) policy while the generator runs, the live LoRA is back
     afterwards, also when the consumer stops early."""
import sys, torch, torch.utils.checkpoint
from h3rl.core.checkpointing import limit_checkpointing
FAILS = []
def check(name, ok, info=""): print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); FAILS.extend([] if ok else [name])

calls = {}
class Block(torch.nn.Module):
    def __init__(self, i): super().__init__(); self.i = i; self.l = torch.nn.Linear(8, 8)
    def forward(self, x): calls[self.i] = calls.get(self.i, 0) + 1; return torch.tanh(self.l(x))
class Model(torch.nn.Module):          # the shape of musubi's block loop
    def __init__(self): super().__init__(); self.blocks = torch.nn.ModuleList(Block(i) for i in range(6))
    def forward(self, x):
        for b in self.blocks: x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
        return x.sum()
torch.manual_seed(0); m = Model(); x = torch.randn(4, 8)
def grads(n):
    limit_checkpointing(m, n); m.zero_grad(); calls.clear(); m(x).backward()
    return [p.grad.clone() for p in m.parameters()], dict(calls)
g0, c0 = grads(0); g3, c3 = grads(3); g6, c6 = grads(6)
check("(a) gradients identical with 0 / 3 / 6 plain blocks", all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(g0, g3, g6)))
check("(a) checkpointed blocks recompute, plain ones do not", c0 == {i: 2 for i in range(6)} and c3 == {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1} and c6 == {i: 1 for i in range(6)}, f"{c3}")
limit_checkpointing(m, 0)

import h3rl.algos.nft as A
live = [torch.zeros(3)]; old = [torch.ones(3)]; seen = []
def fake_iter(transformer, network, ctx, cfg, schedule, device, seeds):
    for s in seeds: seen.append(float(live[0].sum())); yield {"seed": s}
A.iter_samples = fake_iter
class Cfg: group_size = 4
class TT: params = live; transformer = network = schedule = device = None
algo = A.NFT.__new__(A.NFT); algo.T = TT(); algo.cfg = Cfg(); algo.old = old; algo._canvas = lambda: None
out = list(algo.rollout({}, {}, 10))
check("(b) streamed rollout samples with the old policy, in seed order", seen == [3.0] * 4 and [o["seed"] for o in out] == [10, 11, 12, 13])
check("(b) live LoRA restored after the group", float(live[0].sum()) == 0.0)
gen = algo.rollout({}, {}, 0); next(gen); mid = float(live[0].sum()); gen.close()
check("(b) early stop (close) also restores the live LoRA", mid == 3.0 and float(live[0].sum()) == 0.0)
if FAILS: print("FAILED:", FAILS); sys.exit(1)
print("ALL PASS")
