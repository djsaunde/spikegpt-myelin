"""A/B: v4 diagonal WKV vs v5 matrix-state retention time-mix, on enwik8.

Same tuned AdamW recipe, same seeded batch stream. Reports val bits/byte vs step
(eager — loss-vs-step is compile-invariant) plus a separate compiled end-to-end
step-time for each variant, so we can judge wall-clock = steps-to-BPC x s/step.

  python benchmarks/wkv_ab.py --layers 6 --embedding 384 --steps 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from muon_ab import (  # noqa: E402
    cosine_factor,
    fixed_val_batches,
    load_bytes,
    sample_batch,
    val_bpc,
)

from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel  # noqa: E402
from spikegpt.wkv5 import HEAD_DIM, MultiHeadRetentionTimeMix  # noqa: E402


def build(cfg, variant: str, device, seed: int):
    torch.manual_seed(seed)
    model = SpikeLanguageModel(cfg)
    if variant in ("v5", "v7"):
        if variant == "v7":
            from spikegpt.wkv7 import RWKV7TimeMix

            shared: dict = {}  # one per model: threads RWKV-7's value residual

            def make(n_embd, n_layer, layer_id):
                return RWKV7TimeMix(n_embd, n_layer, layer_id, head_dim=HEAD_DIM, shared=shared)
        else:

            def make(n_embd, n_layer, layer_id):
                return MultiHeadRetentionTimeMix(n_embd, n_layer, layer_id, head_dim=HEAD_DIM)

        for layer_id, block in enumerate(model.blocks):
            if getattr(block, "att", None) is not None:
                block.att = make(cfg.n_embd, cfg.n_layer, layer_id)
    return model.to(device)


def run_arm(variant, cfg, train_tokens, val_batches, args, device, amp_ctx):
    model = build(cfg, variant, device, seed=args.init_seed)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    base = [args.lr for _ in opt.param_groups]
    gen = torch.Generator().manual_seed(args.data_seed)
    log, ema, step_ms = [], None, []
    for step in range(args.steps):
        factor = cosine_factor(step, args.steps, args.warmup, args.lr_final_ratio)
        for g, b in zip(opt.param_groups, base, strict=True):
            g["lr"] = b * factor
        x, y = sample_batch(train_tokens, args.batch, cfg.context_length, device, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpc = val_bpc(model, val_batches, amp_ctx)
            log.append((step + 1, bpc))
            print(
                f"  [{variant}] step {step + 1:5d}  train_ema {ema:.3f}  val_bpc {bpc:.4f}",
                flush=True,
            )
    return log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--enwik8", default="/home/danjs/code/myelin/data/enwik8")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--embedding", type=int, default=384)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr-final-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--train-bytes", type=int, default=30_000_000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--variants", default="v4,v5")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg = SpikeGPTConfig(
        vocab_size=256,
        context_length=args.context_length,
        n_layer=args.layers,
        n_embd=args.embedding,
    )
    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else (lambda: torch.autocast(device_type="cpu", enabled=False))
    )
    train_tokens = load_bytes(args.enwik8, args.train_bytes, 0)
    val_tokens = load_bytes(args.enwik8, 2_000_000, 90_000_000)
    val_batches = fixed_val_batches(val_tokens, args.batch, cfg.context_length, 8, device)
    print(
        f"config: {args.layers}L/{args.embedding}d ctx{cfg.context_length} batch{args.batch} "
        f"steps{args.steps} head_dim{HEAD_DIM}",
        flush=True,
    )
    results = {}
    for variant in args.variants.split(","):
        n_params = sum(p.numel() for p in build(cfg, variant, "cpu", 0).parameters())
        print(f"\n=== variant: {variant}  ({n_params / 1e6:.1f}M params) ===", flush=True)
        results[variant] = run_arm(variant, cfg, train_tokens, val_batches, args, device, amp_ctx)

    print("\n=== summary: val_bpc @ step ===", flush=True)
    axis = [s for s, _ in next(iter(results.values()))]
    print("step    " + "  ".join(f"{n:>10s}" for n in results), flush=True)
    for i, s in enumerate(axis):
        print(f"{s:5d}   " + "  ".join(f"{results[n][i][1]:10.4f}" for n in results), flush=True)


if __name__ == "__main__":
    main()
