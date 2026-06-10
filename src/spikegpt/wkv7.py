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
        shared: dict | None = None,
    ) -> None:
        super().__init__()
        if n_embd % head_dim != 0:
            raise ValueError(f"n_embd {n_embd} must be divisible by head_dim {head_dim}")
        self.layer_id = layer_id
        # ``shared`` is one dict per model: block 0 writes the value residual
        # ``v_first`` into it, later blocks read it. Blocks run sequentially in the
        # forward pass, so this threads RWKV-7's value residual without touching the
        # model's block loop. If shared is None each block is self-contained.
        self._shared = shared
        self.attn = RWKV7Attention(
            mode="chunk",
            hidden_size=n_embd,
            head_dim=head_dim,
            layer_idx=layer_id,
            num_hidden_layers=n_layer,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        v_first = None
        if self._shared is not None and self.layer_id != 0:
            v_first = self._shared.get("v_first")
        out = self.attn(inputs, v_first=v_first)
        result, v_first_out = (out[0], out[-1]) if isinstance(out, tuple) else (out, None)
        if self._shared is not None and self.layer_id == 0 and v_first_out is not None:
            self._shared["v_first"] = v_first_out
        return result
