"""Replace H3's RoPE batched 2x2 matmul with the equivalent elementwise form.

THE bottleneck, measured: `aten::bmm` is 1902 ms of a 2590 ms forward (77%), from exactly 100
calls -- query and key in each of 50 blocks. The stock implementation is

    rotary = torch.stack((h[..., :pairs], h[..., pairs:2*pairs]), dim=-1)   # [..., pairs, 2]
    rotary = torch.matmul(rotation_table, rotary.unsqueeze(-1)).squeeze(-1) # [...,pairs,2,2]@[...,pairs,2,1]

i.e. a batched matrix-VECTOR product with 2x2 matrices and a batch in the tens of thousands.
cuBLAS dispatches that to `gemvx` -- 26,100 launches per forward, no tensor cores, ~4% of the
device's achievable throughput. For comparison, plain bf16 GEMMs on this model's real layer shapes
hit 580-760 TFLOP/s on the same GPU.

Writing out the 2x2 product turns it into four elementwise multiplies and two adds, which are
memory-bound and use the full machine:

    [y1]   [a b] [x1]        y1 = a*x1 + b*x2
    [y2] = [c d] [x2]        y2 = c*x1 + d*x2

This is an algebraic identity, not an approximation, and it makes no assumption that the table is
a rotation -- any 2x2 works, so a non-orthogonal table would still be handled correctly.

Install once, before sampling: install_fast_rope()
"""
from __future__ import annotations

import torch


def _apply_rope_split_half_fast(hidden_states: torch.Tensor, rotation_table: torch.Tensor) -> torch.Tensor:
    pairs = rotation_table.shape[-3]
    x1 = hidden_states[..., :pairs]
    x2 = hidden_states[..., pairs : 2 * pairs]
    a = rotation_table[..., 0, 0]
    b = rotation_table[..., 0, 1]
    c = rotation_table[..., 1, 0]
    d = rotation_table[..., 1, 1]
    # Accumulate in fp32 and cast back. torch.matmul accumulates its products in fp32 even for
    # bf16 inputs; doing the same arithmetic natively in bf16 loses those guard bits, and the
    # error compounds through 50 blocks -- measured 8.6% of signal scale at the model output,
    # which is a behaviour change, not a rounding difference. fp32 elementwise ops are still
    # memory-bound and cost nothing next to the batched-GEMV they replace.
    out_dtype = hidden_states.dtype
    x1f, x2f = x1.float(), x2.float()
    y1 = (a.float() * x1f + b.float() * x2f).to(out_dtype)
    y2 = (c.float() * x1f + d.float() * x2f).to(out_dtype)
    return torch.cat((y1, y2, hidden_states[..., 2 * pairs :]), dim=-1)


def install_fast_rope() -> bool:
    """Swap the module-level helper that Attention.forward resolves by global name."""
    from musubi_tuner.minimax_h3 import model as h3model

    if getattr(h3model, "_rope_fast_installed", False):
        return False
    h3model._apply_rope_split_half_original = h3model._apply_rope_split_half
    h3model._apply_rope_split_half = _apply_rope_split_half_fast
    h3model._rope_fast_installed = True
    return True


def uninstall_fast_rope() -> None:
    from musubi_tuner.minimax_h3 import model as h3model

    if getattr(h3model, "_rope_fast_installed", False):
        h3model._apply_rope_split_half = h3model._apply_rope_split_half_original
        h3model._rope_fast_installed = False
