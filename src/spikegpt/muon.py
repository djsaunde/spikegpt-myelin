"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz) for SpikeGPT.

Muon — the optimizer behind Keller Jordan's nanoGPT speedrun — updates 2D hidden
weight matrices by orthogonalizing the momentum buffer with a quintic
Newton-Schulz iteration before applying it. On transformer hidden matrices it
reaches a given loss in materially fewer steps than AdamW, which is a wall-clock
win when the per-step orthogonalization cost (a handful of bf16 matmuls per
weight) is small next to the model's forward/backward.

This is the single-device variant: no distributed all-gather, no FP8. Use it for
the hidden ``nn.Linear`` weights only (key/value/receptance/output time-mixing
and channel-mixing projections). Embeddings, the LM head, LayerNorm gains/biases,
and the WKV ``time_decay``/``time_first``/``time_mix_*`` vectors are 1D or
embedding-like and must stay on AdamW — :func:`split_muon_params` does that split.

References: Jordan et al., "Muon: An optimizer for the hidden layers of neural
networks" (2024); the modded-nanogpt speedrun.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


@torch.compile(fullgraph=True)
def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize ``grad`` via a quintic Newton-Schulz iteration in bf16.

    Returns a matrix with (approximately) the same column/row space as ``grad``
    but with singular values pushed toward 1 — i.e. ``U V^T`` of ``grad = U S V^T``
    without an explicit SVD. The quintic coefficients are tuned so the iteration
    converges from a spectrally-normalized start in ~5 steps.
    """
    assert grad.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    # Spectral-norm lower bound: divide by the Frobenius norm so ||x||_2 <= 1.
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        update = b * gram + c * gram @ gram
        x = a * x + update @ x
    if transposed:
        x = x.mT
    return x


class Muon(torch.optim.Optimizer):
    """Single-device Muon for 2D weight matrices.

    Args:
        params: 2D parameters only (use :func:`split_muon_params`).
        lr: learning rate (Muon tolerates a larger lr than AdamW; ~0.02-0.05).
        momentum: heavy-ball momentum on the raw gradient.
        nesterov: look-ahead momentum (recommended).
        ns_steps: Newton-Schulz iterations per step (5 is plenty).
        weight_decay: decoupled (AdamW-style) weight decay.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        momentum_warmup: int = 0,
        momentum_start: float = 0.85,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            momentum_warmup=momentum_warmup,
            momentum_start=momentum_start,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        for group in self.param_groups:
            # Momentum warmup (modded-nanogpt): ramp momentum_start -> momentum over
            # the first momentum_warmup steps; small early momentum lets the
            # orthogonalized direction settle before heavy averaging.
            warm = group["momentum_warmup"]
            if warm > 0 and self._step_count <= warm:
                frac = self._step_count / warm
                momentum = group["momentum_start"] + frac * (
                    group["momentum"] - group["momentum_start"]
                )
            else:
                momentum = group["momentum"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.ndim != 2:
                    raise ValueError(
                        f"Muon only updates 2D matrices; got {tuple(p.shape)}. "
                        "Route 1D/embedding params to AdamW via split_muon_params()."
                    )
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = _zeropower_via_newtonschulz5(update, group["ns_steps"])
                # RMS-matching scale (Moonshot "Muon is Scalable"): an orthogonal
                # matrix has unit singular values, so RMS(update) = 1/sqrt(max(m,n)).
                # Multiplying by 0.2*sqrt(max(m,n)) fixes the update RMS at ~0.2 for
                # every matrix shape, matching AdamW's typical update RMS — so Muon
                # shares AdamW's learning rate and schedule (set muon lr ~ adam lr).
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)
        return loss


class CombinedOptimizer:
    """Drive several optimizers as one (e.g. Muon for hidden matrices + AdamW).

    Exposes the union of the children's ``param_groups`` so an external LR
    scheduler can iterate them. Each group carries an ``lr_scale`` (default 1.0):
    a scheduler that does ``group["lr"] = base * group["lr_scale"]`` then drives
    Muon and AdamW from a single cosine shape at different magnitudes.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        for opt in optimizers:
            for group in opt.param_groups:
                group.setdefault("lr_scale", 1.0)

    @property
    def param_groups(self):  # type: ignore[no-untyped-def]
        return [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for opt, sub in zip(self.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(sub)


def split_muon_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition ``model``'s parameters into (muon_2d, adamw_rest).

    Muon takes the 2D hidden projection weights (the time-mixing and channel-mixing
    ``nn.Linear`` weights). Everything else — token embeddings, the LM head, the
    ``LayerNorm`` gains/biases, and the WKV ``time_*`` vectors — is 1D or
    embedding-shaped and goes to AdamW. The head and embedding are deliberately
    kept on AdamW even though the head weight is 2D: large vocab-projection
    matrices train better with Adam (this matches the nanogpt-speedrun split).
    """
    muon: list[nn.Parameter] = []
    rest: list[nn.Parameter] = []
    # Module-qualified names whose weights must stay on AdamW even if 2D.
    head_like = {"head", "emb", "embedding", "token_embedding"}
    head_modules = {
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Embedding,)) or name.split(".")[-1] in head_like
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        owner = name.rsplit(".", 1)[0]
        is_head = owner in head_modules or name.rsplit(".", 1)[-1] != "weight"
        if param.ndim == 2 and not is_head:
            muon.append(param)
        else:
            rest.append(param)
    return muon, rest
