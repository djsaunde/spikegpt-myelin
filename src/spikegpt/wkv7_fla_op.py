"""fla's fast triton RWKV-7 kernel, wrapped as torch.library custom ops.

fla guards its chunk kernel with ``torch.compiler.disable``, so under
``torch.compile`` it is a hard graph break (surrounding projections fragment and
don't fuse). Wrapping it in ``torch.library.custom_op`` makes it an *opaque* node:
dynamo emits an extern call (no break), the pure-torch projections still fuse, and
the op runs fla's fast triton kernel eagerly.

Both the forward AND the backward are opaque custom ops with ``register_fake``
shape rules — otherwise AOTAutograd traces the backward and tries to run the
(un-faketraceable) triton kernel on FakeTensors. The backward recomputes the
forward under fla's own autograd (one extra forward; cheap vs a fragmented step).

Layout matches fla.layers.RWKV7Attention: r,w,k,v,a,b are ``[B, T, H, D]`` with
a=-kk, b=kk*a, w=log-decay. Returns o ``[B, T, H, D]``.
"""

from __future__ import annotations

import torch
from fla.ops.rwkv7 import chunk_rwkv7

_CHUNK = 64


def _fla_forward(r, w, k, v, a, b):
    o, _ = chunk_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        scale=1.0,
        initial_state=None,
        output_final_state=False,
        chunk_size=_CHUNK,
    )
    return o


@torch.library.custom_op("spikegpt::rwkv7_chunk", mutates_args=())
def rwkv7_chunk(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _fla_forward(r, w, k, v, a, b)


@rwkv7_chunk.register_fake
def _(r, w, k, v, a, b):
    return torch.empty_like(v)


@torch.library.custom_op("spikegpt::rwkv7_chunk_bwd", mutates_args=())
def rwkv7_chunk_bwd(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        ins = [x.detach().requires_grad_(True) for x in (r, w, k, v, a, b)]
        o = _fla_forward(*ins)
        grads = torch.autograd.grad(o, ins, grad_o, allow_unused=True)
    return tuple(
        g if g is not None else torch.zeros_like(x) for g, x in zip(grads, ins, strict=True)
    )


@rwkv7_chunk_bwd.register_fake
def _(r, w, k, v, a, b, grad_o):
    return (
        torch.empty_like(r),
        torch.empty_like(w),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(a),
        torch.empty_like(b),
    )


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_o):
    return rwkv7_chunk_bwd(*ctx.saved_tensors, grad_o.contiguous())


rwkv7_chunk.register_autograd(_backward, setup_context=_setup_context)
