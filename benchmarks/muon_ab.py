"""A/B: AdamW (current recipe) vs Muon-hybrid on enwik8, steps-to-target BPC.

Both arms share the same model init and the same batch stream (seeded), so the
only difference is the optimizer. We log validation bits/byte on a fixed held-out
slice vs step. Loss-vs-step is torch.compile-invariant, so this runs eager for
fast iteration; Muon's per-step compute overhead is reported separately.

Wall-clock at fixed quality = (steps to reach a target BPC) x (seconds/step).
Muon wins if it reaches the target in materially fewer steps.

  python benchmarks/muon_ab.py --enwik8 /home/danjs/code/myelin/data/enwik8 \
      --layers 6 --embedding 384 --context-length 512 --batch 32 --steps 1500
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel
from spikegpt.muon import CombinedOptimizer, Muon, split_muon_params


def load_bytes(path: str, n: int, offset: int = 0) -> torch.Tensor:
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(n)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


def sample_batch(tokens: torch.Tensor, B: int, T: int, device, gen: torch.Generator):
    idx = torch.randint(0, tokens.numel() - T - 1, (B,), generator=gen)
    x = torch.stack([tokens[i : i + T] for i in idx]).to(device)
    y = torch.stack([tokens[i + 1 : i + 1 + T] for i in idx]).to(device)
    return x, y


def fixed_val_batches(tokens: torch.Tensor, B: int, T: int, n_batches: int, device):
    """Deterministic contiguous windows for a stable val metric."""
    stride = T
    starts = list(range(0, n_batches * B * stride, stride))[: n_batches * B]
    batches = []
    for k in range(n_batches):
        sel = starts[k * B : (k + 1) * B]
        x = torch.stack([tokens[i : i + T] for i in sel]).to(device)
        y = torch.stack([tokens[i + 1 : i + 1 + T] for i in sel]).to(device)
        batches.append((x, y))
    return batches


@torch.no_grad()
def val_bpc(model, batches, amp_ctx) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        with amp_ctx():
            loss, _ = model(x, y)
        total += loss.item()
    model.train()
    return (total / len(batches)) / math.log(2.0)


def cosine_factor(step: int, total: int, warmup: int, final_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return final_ratio + 0.5 * (1 - final_ratio) * (1 + math.cos(math.pi * t))


def make_model(cfg, device, seed=0):
    torch.manual_seed(seed)
    return SpikeLanguageModel(cfg).to(device)


def run_arm(name, cfg, optimizer_fn, train_tokens, val_batches, args, device, amp_ctx):
    model = make_model(cfg, device, seed=args.init_seed)
    model.train()
    optimizer, group_base = optimizer_fn(model)
    gen = torch.Generator().manual_seed(args.data_seed)
    log = []
    ema = None
    step_ms = []
    for step in range(args.steps):
        factor = cosine_factor(step, args.steps, args.warmup, args.lr_final_ratio)
        for g, base in zip(optimizer.param_groups, group_base, strict=True):
            g["lr"] = base * factor
        x, y = sample_batch(train_tokens, args.batch, cfg.context_length, device, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            loss, _ = model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpc = val_bpc(model, val_batches, amp_ctx)
            log.append((step + 1, bpc))
            recent = step_ms[-args.eval_every :]
            ms = sum(recent) / len(recent)
            print(
                f"  [{name}] step {step + 1:5d}  train_ema {ema:.3f}  "
                f"val_bpc {bpc:.4f}  {ms:.1f} ms/step",
                flush=True,
            )
    return log, sum(step_ms[len(step_ms) // 2 :]) / max(1, len(step_ms) - len(step_ms) // 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--enwik8", default="/home/danjs/code/myelin/data/enwik8")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--embedding", type=int, default=384)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--lr-final-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--train-bytes", type=int, default=20_000_000)
    p.add_argument("--adam-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-adam-lr", type=float, default=2e-3)
    p.add_argument("--arms", default="adamw,muon")
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
    train_tokens = load_bytes(args.enwik8, args.train_bytes, offset=0)
    val_tokens = load_bytes(args.enwik8, 2_000_000, offset=90_000_000)
    val_batches = fixed_val_batches(val_tokens, args.batch, cfg.context_length, 8, device)
    print(
        f"config: {args.layers}L/{args.embedding}d ctx{cfg.context_length} batch{args.batch} "
        f"steps{args.steps} | train {train_tokens.numel() / 1e6:.0f}M "
        f"val {val_tokens.numel() / 1e6:.0f}M",
        flush=True,
    )

    def adamw_fn(model):
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.adam_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
        return opt, [args.adam_lr for _ in opt.param_groups]

    def muon_fn(model):
        muon_p, rest_p = split_muon_params(model)
        m = Muon(muon_p, lr=args.muon_lr, weight_decay=args.weight_decay)
        a = torch.optim.AdamW(
            rest_p,
            lr=args.muon_adam_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
        opt = CombinedOptimizer([m, a])
        base = [args.muon_lr] * len(m.param_groups) + [args.muon_adam_lr] * len(a.param_groups)
        return opt, base

    arms = {"adamw": adamw_fn, "muon": muon_fn}
    results = {}
    for name in args.arms.split(","):
        print(f"\n=== arm: {name} ===", flush=True)
        log, ms = run_arm(name, cfg, arms[name], train_tokens, val_batches, args, device, amp_ctx)
        results[name] = (log, ms)

    print("\n=== summary: val_bpc @ step ===", flush=True)
    steps_axis = [s for s, _ in next(iter(results.values()))[0]]
    header = "step    " + "  ".join(f"{n:>10s}" for n in results)
    print(header, flush=True)
    for i, s in enumerate(steps_axis):
        row = f"{s:5d}   " + "  ".join(f"{results[n][0][i][1]:10.4f}" for n in results)
        print(row, flush=True)
    print("\nmedian ms/step:", {n: round(results[n][1], 1) for n in results}, flush=True)


if __name__ == "__main__":
    main()
