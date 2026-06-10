"""Compiled end-to-end step time: v4 vs v5 time-mix, full SpikeGPT model.

Pairs with wkv_ab.py (which gives BPC vs step). Together: wall-clock to a BPC
target = (steps from wkv_ab) x (seconds/step here). Regional torch.compile per
block, bf16, fused AdamW — matching the production training path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from wkv_ab import build  # noqa: E402

from spikegpt.language import SpikeGPTConfig  # noqa: E402


def measure(model, opt, B, T, V, device, warmup=8, iters=25):
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.randint(0, V, (B, T), device=device)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = model(x, y)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--embedding", type=int, default=512)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=256)
    args = p.parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    cfg = SpikeGPTConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        n_layer=args.layers,
        n_embd=args.embedding,
    )
    print(
        f"config: {args.layers}L/{args.embedding}d ctx{args.context_length} "
        f"batch{args.batch} (compiled)",
        flush=True,
    )
    torch._dynamo.config.cache_size_limit = 64
    out = {}
    # v4: fully compiles. v7 (fla): graph-breaks at the disabled triton kernel.
    # v7c (pure-torch): fully fuses. fullgraph=False = "best each can compile".
    for variant in ("v4", "v7", "v7c"):
        torch.compiler.reset()
        model = build(cfg, variant, device, seed=0)
        for i, block in enumerate(model.blocks):
            model.blocks[i] = torch.compile(block, fullgraph=False)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, fused=True)
        ms = measure(model, opt, args.batch, args.context_length, args.vocab_size, device)
        out[variant] = ms
        print(f"  {variant:4s} {ms:8.3f} ms/step", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    base = out.get("v4")
    if base:
        print("\nstep speedup vs v4:", flush=True)
        for v in out:
            print(f"  {v:4s} {base / out[v]:.2f}x", flush=True)


if __name__ == "__main__":
    main()
