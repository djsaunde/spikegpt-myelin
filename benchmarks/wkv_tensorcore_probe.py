"""Throughput probe: v4 diagonal WKV (sequential Triton) vs a chunked multi-head
linear-attention WKV (tensor-core matmuls), at the enwik8 repro dims.

The v4 recurrence is per-channel scalar -> no contraction -> no tensor cores. A
matrix-state (RWKV-5/GLA-style) variant reshapes C into H heads of d and carries
a d x d state per head, expressible as chunked matmuls that hit tensor cores and
parallelize over time. This measures whether that op is actually faster fwd+bwd
(decay omitted: it's elementwise and doesn't change the matmul throughput).
"""

from __future__ import annotations

import argparse
import time

import torch

from spikegpt.wkv_triton import weighted_key_value_triton


def bench(fn, *tensors, iters=50, warmup=10):
    for _ in range(warmup):
        out = fn(*tensors)
        out.sum().backward()
        for t in tensors:
            if t.grad is not None:
                t.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*tensors)
        out.sum().backward()
        for t in tensors:
            if t.grad is not None:
                t.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def chunked_linear_attention(q, k, v, chunk: int = 64):
    """Causal linear attention in chunks -> bmm tensor-core matmuls. [B,H,T,d]."""
    B, H, T, d = q.shape
    nC = T // chunk
    qc = q.reshape(B, H, nC, chunk, d)
    kc = k.reshape(B, H, nC, chunk, d)
    vc = v.reshape(B, H, nC, chunk, d)
    # intra-chunk, causal within the chunk
    attn = qc @ kc.transpose(-1, -2)
    mask = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool))
    attn = attn.masked_fill(~mask, 0.0)
    intra = attn @ vc
    # inter-chunk via running KV state (exclusive cumulative sum over chunks)
    kv = kc.transpose(-1, -2) @ vc  # [B,H,nC,d,d]
    kv_excl = torch.cumsum(kv, dim=2) - kv
    inter = qc @ kv_excl
    return (intra + inter).reshape(B, H, T, d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--channels", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    B, T, C, H = args.batch, args.ctx, args.channels, args.heads
    d = C // H
    dt = torch.bfloat16
    print(f"dims: B{B} T{T} C{C} (H{H} x d{d}) chunk{args.chunk}  dtype bf16", flush=True)

    # --- v4 diagonal WKV (production CUDA path) ---
    key = torch.randn(B, T, C, device=dev, dtype=dt, requires_grad=True)
    val = torch.randn(B, T, C, device=dev, dtype=dt, requires_grad=True)
    decay = torch.randn(C, device=dev, dtype=torch.float32, requires_grad=True)
    first = torch.randn(C, device=dev, dtype=torch.float32, requires_grad=True)
    v4_ms = bench(
        lambda k, v: weighted_key_value_triton(k, v, decay, first),
        key,
        val,
        iters=args.iters,
    )
    print(f"  v4 diagonal WKV (sequential triton)   {v4_ms:8.3f} ms fwd+bwd", flush=True)

    # --- chunked multi-head linear attention (tensor cores), eager + compiled ---
    q = torch.randn(B, H, T, d, device=dev, dtype=dt, requires_grad=True)
    k2 = torch.randn(B, H, T, d, device=dev, dtype=dt, requires_grad=True)
    v2 = torch.randn(B, H, T, d, device=dev, dtype=dt, requires_grad=True)
    la_eager = bench(
        lambda a, b, c: chunked_linear_attention(a, b, c, args.chunk), q, k2, v2, iters=args.iters
    )
    print(f"  chunked MHLA  (eager, tensor cores)    {la_eager:8.3f} ms fwd+bwd", flush=True)
    la_c = torch.compile(chunked_linear_attention)
    la_comp = bench(lambda a, b, c: la_c(a, b, c, args.chunk), q, k2, v2, iters=args.iters)
    print(f"  chunked MHLA  (compiled)               {la_comp:8.3f} ms fwd+bwd", flush=True)

    print(
        f"\nspeedup vs v4:  eager {v4_ms / la_eager:.2f}x   compiled {v4_ms / la_comp:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
