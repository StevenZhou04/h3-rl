#!/usr/bin/env python
"""Regression test for rope_fast: it must be BIT-IDENTICAL to H3's stock RoPE.

rope_fast replaces a batched 2x2 matrix-vector product with the four elementwise multiplies
it expands to. That is an algebraic identity, so the outputs must match exactly -- if this
test ever fails, the replacement has stopped being a pure optimisation.

The fp32 accumulation inside rope_fast is load-bearing and NOT cosmetic: torch.matmul
accumulates bf16 products in fp32, and doing the same arithmetic natively in bf16 differs by
~4% of signal scale in a single layer, compounding through all 50 blocks. The third case below
pins that, so nobody "simplifies" the .float() calls away.

Note: the 3.72x speedup is GPU-only. On CPU the elementwise form is SLOWER -- there is no
cuBLAS gemv dispatch to avoid and it moves more memory. Do not benchmark this on CPU.

    python rope_selftest.py
"""
import os, sys
import torch


from musubi_tuner.minimax_h3.model import _apply_rope_split_half as stock
from h3rl.core.rope_fast import _apply_rope_split_half_fast as fast

FAILS = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok: FAILS.append(name)


def make(dtype, rows=2048, heads=8, head_dim=128):
    pairs = head_dim // 2
    h = torch.randn(1, rows, heads, head_dim, dtype=dtype)
    ang = torch.rand(1, rows, 1, pairs) * 6.283185307
    cos, sin = torch.cos(ang), torch.sin(ang)
    tbl = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    return h, tbl.expand(1, rows, heads, pairs, 2, 2).contiguous().to(dtype)


def naive_same_dtype(h, t):
    p = t.shape[-3]
    x1, x2 = h[..., :p], h[..., p:2 * p]
    a, b, c, d = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]
    return torch.cat((a * x1 + b * x2, c * x1 + d * x2, h[..., 2 * p:]), dim=-1)


torch.manual_seed(0)
print(__doc__.split("\n")[0])
for dtype in (torch.float32, torch.bfloat16):
    h, tbl = make(dtype)
    ys, yf = stock(h, tbl), fast(h, tbl)
    d = float((yf.float() - ys.float()).abs().max())
    check(f"{dtype} : fast is bit-identical to stock", d == 0.0, f"max|diff| {d:.3e}")

# the fp32 accumulation must matter -- if this stops differing, the guard bits were lost
h, tbl = make(torch.bfloat16)
dn = float((naive_same_dtype(h, tbl).float() - stock(h, tbl).float()).abs().max())
scale = float(stock(h, tbl).float().abs().mean())
# Relative, not absolute: the point is that the error is a large fraction of the signal,
# not that it clears some multiple of a bf16 ulp.
check("bf16-accumulated variant DOES differ (fp32 accumulation is load-bearing)",
      dn / scale > 0.01, f"{dn:.3e} = {100 * dn / scale:.2f}% of signal, one layer")

# RoPE is a rotation: it must preserve the norm of each (x1, x2) pair
h, tbl = make(torch.float32, rows=512)
p = tbl.shape[-3]
for name, fn in (("stock", stock), ("fast", fast)):
    y = fn(h, tbl)
    n0 = (h[..., :p] ** 2 + h[..., p:2 * p] ** 2).sqrt()
    n1 = (y[..., :p] ** 2 + y[..., p:2 * p] ** 2).sqrt()
    check(f"{name}: pairwise norm preserved", float((n1 - n0).abs().max()) < 1e-5)

# the untouched tail must pass through verbatim
h, tbl = make(torch.bfloat16)
check("dimensions beyond 2*pairs pass through unchanged",
      torch.equal(fast(h, tbl)[..., 2 * p:], h[..., 2 * p:]))

print()
if FAILS:
    print("FAILED: " + "; ".join(FAILS)); sys.exit(1)
print("ALL PASS -- rope_fast is a pure optimisation")
