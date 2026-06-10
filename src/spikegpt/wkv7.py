"""RWKV-7 ("Goose") time-mixing for SpikeGPT, via flash-linear-attention.

RWKV-7 replaces v4's static diagonal decay with a *data-dependent vector decay*
plus a *delta-rule* (DeltaNet-style, error-correcting) state update — the current
state of the art for the RWKV/linear-attention family, and known to beat v4 at
language modelling. Its recurrence is a chunked matmul (tensor cores), so unlike
v4's sequential elementwise kernel it parallelizes over time.

We use ``fla``'s battle-tested chunked RWKV-7 kernel rather than hand-rolling the
delta rule. ``RWKV7TimeMix`` is a drop-in for ``SpikeTimeMix`` (``[B,T,C]`` in/out)
so both can be A/B'd in the same harness.

Caveat (MVP): true RWKV-7 threads a value-residual (``v_first``) across layers;
here each block is self-contained (``layer_idx=0``), which slightly handicaps v7.
Requires the optional ``flash-linear-attention`` dependency.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # fla re-exports the layer at package level; fall back to the submodule
    from fla.layers import RWKV7Attention
except Exception:  # pragma: no cover
    from fla.layers.rwkv7 import RWKV7Attention


class RWKV7TimeMix(nn.Module):
    """Drop-in for ``SpikeTimeMix`` using fla's chunked RWKV-7 recurrence."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        *,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        if n_embd % head_dim != 0:
            raise ValueError(f"n_embd {n_embd} must be divisible by head_dim {head_dim}")
        # layer_idx=0 makes each block self-contained (computes its own value
        # residual instead of consuming the first layer's); see module caveat.
        self.attn = RWKV7Attention(
            mode="chunk",
            hidden_size=n_embd,
            head_dim=head_dim,
            layer_idx=0,
            num_hidden_layers=n_layer,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.attn(inputs)
        return out[0] if isinstance(out, tuple) else out
