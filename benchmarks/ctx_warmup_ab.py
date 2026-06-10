"""A/B: constant context vs context-length warmup, same v4 model, on enwik8.

v4's WKV recurrence is sequential over time, so its wall-clock is latency-bound
in the context length T. Holding tokens-per-step constant (B*T fixed) while
*shrinking T and growing B* early in training cuts the sequential depth -> faster
steps, same tokens seen, *identical model* (no architecture/quality risk). RWKV
length-generalizes, so we still eval at the full target context.

This measures wall-clock to a fixed BPC: val bits/byte (at the fixed eval ctx) vs
*cumulative training seconds*, for a constant-ctx baseline and a ctx-warmup
schedule. A win = the warmup curve reaches the same BPC at less wall-clock.

  python benchmarks/ctx_warmup_ab.py --steps 2500 --eval-ctx 512 --base-batch 32
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


def parse_schedule(spec: str, eval_ctx: int):
    """'0.25:0.4,0.5:0.3,1.0:0.3' -> [(ctx_abs, step_frac), ...] (last ctx==eval_ctx)."""
    out = []
    for part in spec.split(","):
        cfrac, _, sfrac = part.partition(":")
        out.append((max(1, round(float(cfrac) * eval_ctx)), float(sfrac)))
    out[-1] = (eval_ctx, out[-1][1])
    return out


def ctx_at_step(step, total, eval_ctx, base_batch, warmup, schedule):
    """Return (ctx, batch) for this step. tokens/step = eval_ctx*base_batch fixed."""
    tokens = eval_ctx * base_batch
    if not warmup:
        return eval_ctx, base_batch
    frac = step / total
    acc = 0.0
    for ctx, sfrac in schedule:
        acc += sfrac
        if frac < acc:
            return ctx, max(1, tokens // ctx)
    return eval_ctx, base_batch


def run_arm(name, warmup, cfg, train_tokens, val_batches, args, device, amp_ctx, schedule):
    torch.manual_seed(args.init_seed)
    model = SpikeLanguageModel(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    gen = torch.Generator().manual_seed(args.data_seed)
    log, ema, wall = [], None, 0.0
    for step in range(args.steps):
        factor = cosine_factor(step, args.steps, args.warmup, args.lr_final_ratio)
        for g in opt.param_groups:
            g["lr"] = args.lr * factor
        ctx, batch = ctx_at_step(step, args.steps, args.eval_ctx, args.base_batch, warmup, schedule)
        x, y = sample_batch(train_tokens, batch, ctx, device, gen)
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
        wall += time.perf_counter() - t0
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpc = val_bpc(model, val_batches, amp_ctx)
            log.append((step + 1, wall, bpc))
            print(
                f"  [{name}] step {step + 1:5d}  wall {wall:6.1f}s  "
                f"ctx{ctx:4d}xB{batch:<4d}  val_bpc {bpc:.4f}",
                flush=True,
            )
    return log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--enwik8", default="/home/danjs/code/myelin/data/enwik8")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--embedding", type=int, default=384)
    p.add_argument("--eval-ctx", type=int, default=512)
    p.add_argument("--base-batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr-final-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--train-bytes", type=int, default=40_000_000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument(
        "--schedule",
        default="0.25:0.4,0.5:0.3,1.0:0.3",
        help="ctxfrac:stepfrac phases for the warmup arm (last ctxfrac must be 1.0)",
    )
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg = SpikeGPTConfig(
        vocab_size=256,
        context_length=args.eval_ctx,
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
    val_batches = fixed_val_batches(val_tokens, args.base_batch, args.eval_ctx, 8, device)
    print(
        f"config: {args.layers}L/{args.embedding}d eval_ctx{args.eval_ctx} "
        f"tokens/step={args.eval_ctx * args.base_batch} steps{args.steps}",
        flush=True,
    )
    results = {}
    for name, warmup in (("constant", False), ("warmup", True)):
        print(f"\n=== arm: {name} ===", flush=True)
        schedule = parse_schedule(args.schedule, args.eval_ctx)
        results[name] = run_arm(
            name, warmup, cfg, train_tokens, val_batches, args, device, amp_ctx, schedule
        )

    print("\n=== final: BPC and total wall-clock ===", flush=True)
    for name, log in results.items():
        s, wall, bpc = log[-1]
        print(f"  {name:9s} {bpc:.4f} BPC in {wall:.1f}s ({s} steps)", flush=True)
    # iso-BPC wall-clock: at the constant arm's final BPC, when did warmup reach it?
    target = results["constant"][-1][2]
    for name, log in results.items():
        hit = next((w for (_, w, b) in log if b <= target), None)
        print(
            f"  {name:9s} reached {target:.4f} BPC at {hit if hit else float('nan'):.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
