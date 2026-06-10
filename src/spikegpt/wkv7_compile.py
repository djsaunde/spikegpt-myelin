"""Compile-friendly RWKV-7 ("Goose") time-mixing — pure PyTorch, no custom ops.

``fla``'s RWKV-7 path wraps its Triton chunk kernel in ``torch.compiler.disable``,
so under ``torch.compile`` it is a hard graph break: the recurrence never fuses and
the surrounding projections fragment, leaving the v7 step ~3.6x slower than a fully
compiled v4 (see benchmarks/results/optimizer_wkv_experiments_5090.md).

This module reimplements RWKV-7 entirely in aten ops — the token-shift, the
LoRA-gated projections, the DPLR (diagonal-plus-low-rank) delta-rule recurrence,
the per-head norm, the value residual, and the gate/bonus correction — so
``torch.compile(fullgraph=True)`` fuses the whole block. The DPLR chunk kernel is
the pure-PyTorch reference from flash-linear-attention (Songlin Yang, Yu Zhang,
Zhiyuan Li; MIT), inlined here with einops rewritten as reshape/transpose.

RWKV-7 -> DPLR mapping (matches fla.layers.RWKV7Attention): q=r, k=k, v=v,
alpha=-kk, beta=kk*a, gk=w (log-decay), with kk = l2norm(k*k_k) per head.
``RWKV7TimeMixCompile`` is a drop-in for ``SpikeTimeMix`` (``[B,T,C]`` in/out).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def dplr_chunkwise(q, k, v, alpha, beta, gk, chunk_size: int = 32):
    """Chunked DPLR delta rule: S_t = S_{t-1}(diag(exp(gk)) + alpha beta^T) + v k^T.

    q,k,v,alpha,beta,gk: [B, H, L, D]. Pure aten (matmul/cumsum/exp/masked_fill),
    static loops over chunk_size and L/chunk_size -> torch.compile unrolls + fuses.
    Inlined from flash-linear-attention (dplr/naive.py, MIT), einops -> reshape.
    """
    b, h, length, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k**-0.5)
    assert length % chunk_size == 0
    n = length // chunk_size
    c = chunk_size
    S = k.new_zeros(b, h, d_k, d_v)

    def chunkify(x):
        return x.reshape(b, h, n, c, x.shape[-1]).float()

    q, k, v, alpha, beta, gk = map(chunkify, (q, k, v, alpha, beta, gk))
    gk_cumsum = gk.cumsum(-2)

    A_ab = q.new_zeros(b, h, n, c, c)
    A_qk = q.new_zeros(b, h, n, c, c)
    A_ak = q.new_zeros(b, h, n, c, c)
    A_qb = q.new_zeros(b, h, n, c, c)
    arange = torch.arange(c, device=q.device)
    for i in range(c):
        alpha_i = alpha[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (arange <= i).view(1, c)
        attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float("-inf")).exp()
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * beta * attn_i).sum(-1).clone()
        mask = (arange < i).view(1, c)
        attn_i = gk_i - gk[:, :, :, i, None] - gk_cumsum
        attn_i = attn_i.masked_fill(~mask.unsqueeze(-1), float("-inf")).exp()
        A_ab[:, :, :, i, :] = (alpha_i * beta * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (alpha_i * k * attn_i).sum(-1).clone()

    for i in range(1, c):
        prefix = (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + prefix
    A_ab = A_ab + torch.eye(c, dtype=torch.float, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum - gk).exp() * alpha)

    o = torch.zeros_like(v)
    for i in range(n):
        q_i, k_i, v_i, u_i, w_i, beta_i = (
            q[:, :, i],
            k[:, :, i],
            v[:, :, i],
            u[:, :, i],
            w[:, :, i],
            beta[:, :, i],
        )
        v2_i = u_i + w_i @ S
        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = (
            S * gk_cumsum[:, :, i, -1, :, None].exp()
            + (k_i * decay).transpose(-1, -2) @ v_i
            + (beta_i * decay).transpose(-1, -2) @ v2_i
        )
    return o.reshape(b, h, length, d_v).to(v.dtype)


class _LoRA(nn.Module):
    """Low-rank gate: up(act(down(x))) (+bias), matching fla's LoRA."""

    def __init__(self, dim: int, out: int, rank: int, activation: str | None, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, out, bias=bias)
        self.activation = activation

    def forward(self, x):
        h = self.down(x)
        if self.activation == "tanh":
            h = torch.tanh(h)
        elif self.activation == "sigmoid":
            h = torch.sigmoid(h)
        return self.up(h)


class RWKV7TimeMixCompile(nn.Module):
    """Drop-in for ``SpikeTimeMix``: compile-friendly pure-torch RWKV-7."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        *,
        head_dim: int = 64,
        chunk_size: int = 32,
        shared: dict | None = None,
    ) -> None:
        super().__init__()
        if n_embd % head_dim != 0:
            raise ValueError(f"n_embd {n_embd} must be divisible by head_dim {head_dim}")
        self.n_head = n_embd // head_dim
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.layer_id = layer_id
        self._shared = shared

        ratio = 1.0 - layer_id / max(1, n_layer)
        ramp = (torch.arange(n_embd, dtype=torch.float32) / n_embd).view(1, 1, -1)
        self.x_r = nn.Parameter(1.0 - torch.pow(ramp, 0.2 * ratio))
        self.x_w = nn.Parameter(1.0 - torch.pow(ramp, 0.9 * ratio))
        self.x_k = nn.Parameter(1.0 - torch.pow(ramp, 0.7 * ratio))
        self.x_v = nn.Parameter(1.0 - torch.pow(ramp, 0.7 * ratio))
        self.x_a = nn.Parameter(1.0 - torch.pow(ramp, 0.9 * ratio))
        self.x_g = nn.Parameter(1.0 - torch.pow(ramp, 0.2 * ratio))

        rank_w = max(32, n_embd // 8)
        rank_a = max(32, n_embd // 16)
        rank_g = max(32, n_embd // 8)
        self.r_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.w_lora = _LoRA(n_embd, n_embd, rank_w, "tanh")
        self.a_lora = _LoRA(n_embd, n_embd, rank_a, None)
        self.g_lora = _LoRA(n_embd, n_embd, rank_g, "sigmoid", bias=False)
        self.v_lora = _LoRA(n_embd, n_embd, rank_a, None) if layer_id != 0 else None

        self.k_k = nn.Parameter(torch.ones(n_embd) * 0.85)
        self.k_a = nn.Parameter(torch.ones(n_embd))
        self.r_k = nn.Parameter(torch.zeros(self.n_head, head_dim))
        self.ln_x = nn.LayerNorm(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h, d = self.n_head, self.head_dim
        xx = F.pad(x, (0, 0, 1, -1)) - x  # token shift: x_{t-1} - x_t

        def mix(p):
            return x + xx * p

        r = self.r_proj(mix(self.x_r))
        w = -0.6065306597126334 * torch.sigmoid(self.w_lora(mix(self.x_w)))
        k = self.k_proj(mix(self.x_k))
        v = self.v_proj(mix(self.x_v))
        a = torch.sigmoid(self.a_lora(mix(self.x_a)))
        g = self.g_lora(mix(self.x_g))

        if self._shared is not None and self.layer_id != 0 and self.v_lora is not None:
            v = torch.lerp(v, self._shared["v_first"], torch.sigmoid(self.v_lora(mix(self.x_v))))
        elif self._shared is not None and self.layer_id == 0:
            self._shared["v_first"] = v

        kk = F.normalize((k * self.k_k).view(b, t, h, d), dim=-1, p=2.0).reshape(b, t, c)
        k = k * (1 + (a - 1) * self.k_a)

        def heads(z):
            return z.view(b, t, h, d).transpose(1, 2)  # [B,H,T,D]

        r_h, k_h, v_h, a_h, w_h, kk_h = map(heads, (r, k, v, a, w, kk))
        o = dplr_chunkwise(r_h, k_h, v_h, -kk_h, kk_h * a_h, w_h, chunk_size=self.chunk_size)
        o = o.transpose(1, 2)  # [B,T,H,D]
        o = self.ln_x(o)

        # gate/bonus correction: (o + (r·k·r_k) v) * g, per head
        rh, kh, vh, gh = (z.view(b, t, h, d) for z in (r, k, v, g))
        bonus = (rh * kh * self.r_k).sum(-1, keepdim=True) * vh
        o = (o + bonus) * gh
        return self.o_proj(o.reshape(b, t, c))
