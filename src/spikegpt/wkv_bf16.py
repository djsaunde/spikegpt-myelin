"""bf16-I/O variant of the fused Triton WKV kernel (RWKV-v4 recurrence).

The production kernel (``wkv_triton.py``) upcasts k/v (and grad_y/k/v on the
backward) to float32 in the wrapper via ``.float().contiguous()`` before the
kernel reads them at 4 bytes/element — ~14% of WKV step time (Modal
``wkv_io_cast_cost``), plus the doubled input bandwidth inside the kernel. Here
the kernel loads the native bf16/fp16 tensors and casts to float32 *in-register*,
so the recurrence still runs in float32 (the y/z/zexp backward-replay scratch
stays float32) but the I/O traffic is halved.

Because ``key.float()`` is a lossless bf16->fp32 widening, loading bf16 and
casting in-register yields the *identical* float32 values; with fp32 scratch the
output and gradients are bit-identical to the fp32-I/O path — this is a pure
bandwidth win, not a precision trade (validated on Modal, ``wkv_bf16io_validate``).
Same RWKV-v4 math as ``wkv_triton.py``, so it is reproduction-safe.
"""

from __future__ import annotations

import torch
from myelin._optional import has_triton
from torch import Tensor

if has_triton():
    import triton
    import triton.language as tl

    _BLOCK = 64

    @triton.jit
    def _wkv_forward_kernel_bf16(k_ptr, v_ptr, w_ptr, u_ptr, y_ptr, T, C, BLOCK: tl.constexpr):
        pid_b = tl.program_id(0)
        cs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cs < C
        w = tl.load(w_ptr + cs, mask=mask, other=0.0)
        u = tl.load(u_ptr + cs, mask=mask, other=0.0)
        p = tl.zeros([BLOCK], tl.float32)
        q = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        base = pid_b * T * C
        for t in range(T):
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + off, mask=mask, other=0.0).to(tl.float32)
            no = tl.maximum(o, u + k)
            a = tl.exp(o - no)
            b = tl.exp(u + k - no)
            tl.store(y_ptr + off, (a * p + b * v) / (a * q + b), mask=mask)
            no = tl.maximum(w + o, k)
            a = tl.exp(w + o - no)
            b = tl.exp(k - no)
            p = a * p + b * v
            q = a * q + b
            o = no

    @triton.jit
    def _wkv_backward_kernel_bf16(
        w_ptr,
        u_ptr,
        k_ptr,
        v_ptr,
        gy_ptr,
        y_ptr,
        z_ptr,
        zexp_ptr,
        gw_ptr,
        gu_ptr,
        gk_ptr,
        gv_ptr,
        T,
        C,
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        cs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cs < C
        w = tl.load(w_ptr + cs, mask=mask, other=0.0)
        u = tl.load(u_ptr + cs, mask=mask, other=0.0)
        base = pid_b * T * C

        # Pass 1: forward replay; cache y/z/zexp (fp32 scratch) and accumulate gw, gu.
        p = tl.zeros([BLOCK], tl.float32)
        q = tl.zeros([BLOCK], tl.float32)
        dpdw = tl.zeros([BLOCK], tl.float32)
        dqdw = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        gw = tl.zeros([BLOCK], tl.float32)
        gu = tl.zeros([BLOCK], tl.float32)
        for t in range(T):
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + off, mask=mask, other=0.0).to(tl.float32)
            gy = tl.load(gy_ptr + off, mask=mask, other=0.0).to(tl.float32)
            no = tl.maximum(o, k + u)
            a = tl.exp(o - no)
            b = tl.exp(k + u - no)
            iden = 1.0 / (a * q + b)
            y = (a * p + b * v) * iden
            tl.store(y_ptr + off, y, mask=mask)
            tl.store(z_ptr + off, iden, mask=mask)
            tl.store(zexp_ptr + off, k + u - no, mask=mask)
            gw += gy * (dpdw - dqdw * y) * iden * a
            gu += gy * (v - y) * b * iden
            no = tl.maximum(w + o, k)
            a = tl.exp(w + o - no)
            b = tl.exp(k - no)
            dpdw = a * (p + dpdw)
            dqdw = a * (q + dqdw)
            p = a * p + b * v
            q = a * q + b
            o = no
        bc = pid_b * C + cs
        tl.store(gw_ptr + bc, gw * w, mask=mask)  # chain rule: w = -exp(time_decay)
        tl.store(gu_ptr + bc, gu, mask=mask)

        # Pass 2: reverse accumulation for gk, gv.
        gp = tl.zeros([BLOCK], tl.float32)
        gq = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        for ti in range(T):
            t = T - 1 - ti
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + off, mask=mask, other=0.0).to(tl.float32)
            gy = tl.load(gy_ptr + off, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + off, mask=mask, other=0.0)
            z = tl.load(z_ptr + off, mask=mask, other=0.0)
            zx = tl.load(zexp_ptr + off, mask=mask, other=0.0)
            a = gy * z * tl.exp(zx)
            b = tl.exp(k + o)
            tl.store(gk_ptr + off, a * (v - y) + b * (gp * v + gq), mask=mask)
            tl.store(gv_ptr + off, a + b * gp, mask=mask)
            no = tl.maximum(w + o, zx - k - u)
            a = tl.exp(w + o - no)
            b = gy * z * tl.exp(zx - k - u - no)
            gp = a * gp + b
            gq = a * gq - b * y
            o = no

    def _grid(b: int, c: int) -> tuple[int, int]:
        return (b, int(triton.cdiv(c, _BLOCK)))

    @torch.library.custom_op("myelin::wkv_forward_bf16io", mutates_args=())
    def _wkv_forward_bf16io_op(
        key: Tensor, value: Tensor, time_decay: Tensor, time_first: Tensor
    ) -> Tensor:
        b, t, c = key.shape
        w = (-torch.exp(time_decay.float())).contiguous()
        u = time_first.float().contiguous()
        k = key.contiguous()  # native dtype — cast to fp32 in-register
        v = value.contiguous()
        y = torch.empty_like(k)  # native dtype output (store auto-casts fp32->bf16)
        _wkv_forward_kernel_bf16[_grid(b, c)](k, v, w, u, y, t, c, BLOCK=_BLOCK)  # type: ignore[arg-type]
        return y

    @_wkv_forward_bf16io_op.register_fake
    def _(key, value, time_decay, time_first):
        return torch.empty_like(key)

    @torch.library.custom_op("myelin::wkv_backward_bf16io", mutates_args=())
    def _wkv_backward_bf16io_op(
        grad_y: Tensor, w: Tensor, u: Tensor, k: Tensor, v: Tensor
    ) -> list[Tensor]:
        b, t, c = k.shape
        gy = grad_y.contiguous()  # native dtype — cast in-register
        f32 = {"dtype": torch.float32, "device": k.device}
        ys = torch.empty((b, t, c), **f32)  # fp32 replay scratch — keeps grads bit-exact
        zs = torch.empty((b, t, c), **f32)
        zexps = torch.empty((b, t, c), **f32)
        gw = torch.empty((b, c), **f32)
        gu = torch.empty((b, c), **f32)
        gk = torch.empty_like(k)  # native dtype grads (store auto-casts)
        gv = torch.empty_like(k)
        _wkv_backward_kernel_bf16[_grid(b, c)](
            w,
            u,
            k,
            v,
            gy,
            ys,
            zs,
            zexps,
            gw,
            gu,
            gk,
            gv,
            t,
            c,
            BLOCK=_BLOCK,  # type: ignore[arg-type]
        )
        return [gk, gv, gw.sum(0), gu.sum(0)]  # gw already includes the *w chain rule

    @_wkv_backward_bf16io_op.register_fake
    def _(grad_y, w, u, k, v):
        return [torch.empty_like(k), torch.empty_like(k), torch.empty_like(w), torch.empty_like(u)]

    def _setup_context(ctx, inputs, output):
        key, value, time_decay, time_first = inputs
        ctx.save_for_backward(key, value, time_decay, time_first)

    def _backward(ctx, grad_y):
        key, value, time_decay, time_first = ctx.saved_tensors
        w = (-torch.exp(time_decay.float())).contiguous()
        u = time_first.float().contiguous()
        k = key.contiguous()  # native dtype
        v = value.contiguous()
        gk, gv, gtd, gtf = torch.ops.myelin.wkv_backward_bf16io(grad_y.contiguous(), w, u, k, v)
        return (
            gk.to(key.dtype),
            gv.to(value.dtype),
            gtd.to(time_decay.dtype),
            gtf.to(time_first.dtype),
        )

    _wkv_forward_bf16io_op.register_autograd(_backward, setup_context=_setup_context)


def weighted_key_value_triton_bf16io(
    key: Tensor,
    value: Tensor,
    time_decay: Tensor,
    time_first: Tensor,
) -> Tensor:
    """RWKV WKV via the bf16-I/O fused Triton custom op (CUDA only).

    Bit-identical to :func:`spikegpt.wkv_triton.weighted_key_value_triton` for
    bf16/fp16 inputs (fp32 in-register recurrence + fp32 replay scratch), but
    skips the wrapper's ``.float().contiguous()`` materialization and reads k/v at
    2 bytes/element instead of 4.
    """
    return torch.ops.myelin.wkv_forward_bf16io(key, value, time_decay, time_first)
