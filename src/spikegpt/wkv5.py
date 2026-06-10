"""Matrix-state (RWKV-5 / RetNet-style) multi-head time-mixing for SpikeGPT.

The faithful SpikeGPT uses RWKV-v4's *diagonal* WKV: a per-channel scalar
recurrence with no contraction dimension, so it cannot use tensor cores (the
production kernel is a sequential-over-time Triton loop). This module is the
modernized **tuned-track** alternative: reshape the channels into H heads of
``head_dim`` and carry a ``head_dim x head_dim`` state per head, updated by outer
products. That recurrence is a chunked linear attention expressible as batched
matmuls — tensor cores, parallel over time. At the enwik8 repro dims the op is
~3x faster fwd+bwd when compiled.

This is a different model from v4, so it earns its place only if it also clears
the BPC bar; ``MultiHeadRetentionTimeMix`` is a drop-in for ``SpikeTimeMix`` so
both can be A/B'd in the same harness.
"""

from __future__ import annotations

import torch
from torch import nn

HEAD_DIM = 64


def retention_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """Causal multi-head retention with per-head scalar decay, chunked.

    q, k, v: ``[B, H, T, d]``. ``log_gamma``: ``[H]`` with ``gamma = exp(log_gamma)
    in (0, 1)`` the per-head per-step decay. Returns ``[B, H, T, d]``.

    Intra-chunk attention is a fully parallel masked matmul; the cross-chunk
    carry is a short sequential scan (``T/chunk`` steps) over a ``[B,H,d,d]``
    state, each step a batched matmul. Both hit tensor cores.
    """
    B, H, T, d = q.shape
    n_chunks = T // chunk
    assert n_chunks * chunk == T, "context length must be a multiple of chunk"
    qc = q.reshape(B, H, n_chunks, chunk, d)
    kc = k.reshape(B, H, n_chunks, chunk, d)
    vc = v.reshape(B, H, n_chunks, chunk, d)

    gamma = torch.exp(log_gamma).clamp(max=1.0 - 1e-6)  # [H]
    g = gamma.view(H, 1, 1)
    pos = torch.arange(chunk, device=q.device)
    # intra-chunk decay matrix D[h,i,j] = gamma^(i-j) for i>=j else 0
    rel = (pos.view(chunk, 1) - pos.view(1, chunk)).clamp(min=0)
    causal = pos.view(chunk, 1) >= pos.view(1, chunk)
    decay_mat = (g ** rel.unsqueeze(0)) * causal.unsqueeze(0)  # [H,chunk,chunk]

    attn = (qc @ kc.transpose(-1, -2)).float()  # [B,H,nC,chunk,chunk]
    attn = attn * decay_mat.view(1, H, 1, chunk, chunk)
    intra = attn.to(v.dtype) @ vc  # [B,H,nC,chunk,d]

    # cross-chunk: state contribution of each chunk, decayed to its own end
    k_decay = (g ** (chunk - 1 - pos).view(1, chunk, 1)).view(1, H, 1, chunk, 1)
    kv = (kc * k_decay).transpose(-1, -2).to(torch.float32) @ vc.to(torch.float32)  # [B,H,nC,d,d]
    q_decay = (g ** (pos + 1).view(1, chunk, 1)).view(1, H, chunk, 1)
    gamma_chunk = (gamma**chunk).view(1, H, 1, 1)

    state = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    inter = []
    for c in range(n_chunks):
        cross_c = (qc[:, :, c].to(torch.float32) * q_decay) @ state  # [B,H,chunk,d]
        inter.append(cross_c.to(v.dtype))
        state = gamma_chunk * state + kv[:, :, c]
    inter_t = torch.stack(inter, dim=2)  # [B,H,nC,chunk,d]
    return (intra + inter_t).reshape(B, H, T, d)


def _mix_with_previous(inputs: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    previous = torch.zeros_like(inputs)
    previous[:, 1:] = inputs[:, :-1]
    factor = mix[0, 0]
    return inputs * factor + previous * (1.0 - factor)


class MultiHeadRetentionTimeMix(nn.Module):
    """Drop-in for ``SpikeTimeMix`` using a matrix-state retention recurrence.

    Same token-shift mixing and key/value/receptance/output projections as v4, so
    parameter count is comparable. The receptance acts as the per-head query into
    the state (RWKV-5 style) instead of a multiplicative sigmoid gate; a per-head
    LayerNorm stabilizes the readout before the output projection.
    """

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        *,
        head_dim: int = HEAD_DIM,
        chunk: int = 64,
    ) -> None:
        super().__init__()
        if n_embd % head_dim != 0:
            raise ValueError(f"n_embd {n_embd} must be divisible by head_dim {head_dim}")
        self.n_head = n_embd // head_dim
        self.head_dim = head_dim
        self.chunk = chunk

        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        position = torch.arange(n_embd, dtype=torch.float32) / n_embd
        self.time_mix_k = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_v = nn.Parameter(
            (torch.pow(position, ratio_1_to_almost0) + 0.3).view(1, 1, -1)
        )
        self.time_mix_r = nn.Parameter(torch.pow(position, 0.5 * ratio_1_to_almost0).view(1, 1, -1))
        # per-head decay, RetNet-style geometric spread of time constants
        head = torch.arange(self.n_head, dtype=torch.float32)
        gamma = 1.0 - torch.pow(2.0, -5.0 - head)  # in (0,1), longer memory for later heads
        self.log_gamma = nn.Parameter(torch.log(gamma))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.ln_x = nn.LayerNorm(head_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, t, c = inputs.shape
        h, d = self.n_head, self.head_dim
        k = self.key(_mix_with_previous(inputs, self.time_mix_k))
        v = self.value(_mix_with_previous(inputs, self.time_mix_v))
        r = self.receptance(_mix_with_previous(inputs, self.time_mix_r))
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        r = r.view(b, t, h, d).transpose(1, 2)
        out = retention_chunked(r, k, v, self.log_gamma, self.chunk)  # [B,H,T,d]
        out = self.ln_x(out)
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.output(out)
