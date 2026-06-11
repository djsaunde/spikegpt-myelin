"""Speedrun engine: minimize wall-clock to a target val BPC at the 46M (12L/512d)
enwik8 model, one config at a time.

Each invocation trains one config (recipe + levers), logs val bits/byte vs
(step, *steady-state* wall-clock), and reports the wall-clock to reach a target
BPC. Steady-state step time excludes the compile/warmup ramp, so the number
reflects the full run (where compile cost is amortized) rather than the proxy's
short horizon. Append-only leaderboard at benchmarks/results/speedrun_46m.md.

This is the iteration tool: change a lever, run, compare wall-clock-to-target.

  python benchmarks/speedrun.py --label baseline --steps 3000 --target 1.70
  python benchmarks/speedrun.py --label +ctxwarmup --ctx-schedule 256:0.4,512:0.3,1024:0.3
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel

LEADERBOARD = Path(__file__).resolve().parent / "results" / "speedrun_46m.md"
ENWIK8 = "/home/danjs/code/myelin/data/enwik8"


def load_bytes(path, n, offset=0):
    with open(path, "rb") as f:
        f.seek(offset)
        return torch.frombuffer(bytearray(f.read(n)), dtype=torch.uint8).long()


def sample_batch(tokens, batch, ctx, device, gen):
    idx = torch.randint(0, tokens.numel() - ctx - 1, (batch,), generator=gen)
    x = torch.stack([tokens[i : i + ctx] for i in idx]).to(device)
    y = torch.stack([tokens[i + 1 : i + 1 + ctx] for i in idx]).to(device)
    return x, y


def fixed_val_batches(tokens, batch, ctx, n_batches, device):
    out = []
    for kk in range(n_batches):
        sel = [kk * batch * ctx + j * ctx for j in range(batch)]
        x = torch.stack([tokens[i : i + ctx] for i in sel]).to(device)
        y = torch.stack([tokens[i + 1 : i + 1 + ctx] for i in sel]).to(device)
        out.append((x, y))
    return out


@torch.no_grad()
def val_bpc(model, batches, amp):
    model.eval()
    tot = 0.0
    for x, y in batches:
        with amp():
            loss, _ = model(x, y)
        tot += loss.item()
    model.train()
    return tot / len(batches) / math.log(2.0)


def lr_at(step, total, lr, lr_final_ratio, warmup, schedule="cosine", decay_frac=0.4):
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "wsd":
        # warmup-stable-decay: constant lr, then cosine-decay the last decay_frac.
        decay_start = total - int(decay_frac * total)
        if step < decay_start:
            return lr
        t = (step - decay_start) / max(1, total - decay_start)
        return lr * (lr_final_ratio + 0.5 * (1 - lr_final_ratio) * (1 + math.cos(math.pi * t)))
    t = (step - warmup) / max(1, total - warmup)
    return lr * (lr_final_ratio + 0.5 * (1 - lr_final_ratio) * (1 + math.cos(math.pi * t)))


def parse_schedule(spec, full_ctx):
    phases = []
    for part in spec.split(","):
        cf, _, sf = part.partition(":")
        phases.append((int(round(float(cf) * full_ctx)) if float(cf) <= 1 else int(cf), float(sf)))
    phases[-1] = (full_ctx, phases[-1][1])
    return phases


def ctx_at(step, total, schedule, full_ctx, base_batch):
    if schedule is None:
        return full_ctx, base_batch
    tokens = full_ctx * base_batch
    frac, acc = step / total, 0.0
    for ctx, sf in schedule:
        acc += sf
        if frac < acc:
            return ctx, max(1, tokens // ctx)
    return full_ctx, base_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--embedding", type=int, default=512)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--target", type=float, default=1.70, help="target val BPC")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-final-ratio", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--schedule", choices=("cosine", "wsd"), default="cosine")
    p.add_argument("--decay-frac", type=float, default=0.4)
    p.add_argument("--logit-softcap", type=float, default=0.0, help="cap*tanh(logits/cap); 0=off")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ctx-schedule", default=None)
    p.add_argument("--compile", choices=("off", "regional"), default="regional")
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--train-bytes", type=int, default=90_000_000)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--init-seed", type=int, default=0)
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64

    cfg = SpikeGPTConfig(
        vocab_size=256, context_length=args.ctx, n_layer=args.layers, n_embd=args.embedding
    )
    torch.manual_seed(args.init_seed)
    model = SpikeLanguageModel(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    schedule = parse_schedule(args.ctx_schedule, args.ctx) if args.ctx_schedule else None
    if args.compile == "regional":
        for i, block in enumerate(model.blocks):
            model.blocks[i] = torch.compile(block, fullgraph=True, dynamic=bool(schedule))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )
    amp = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731

    train_tokens = load_bytes(ENWIK8, args.train_bytes, 0)
    val_tokens = load_bytes(ENWIK8, 2_000_000, 90_000_000)  # 90-92M: val region (test is 95M+)
    val_batches = fixed_val_batches(val_tokens, 16, args.ctx, 8, dev)
    gen = torch.Generator().manual_seed(args.data_seed)
    print(
        f"[{args.label}] {args.layers}L/{args.embedding}d {n_params / 1e6:.1f}M "
        f"ctx{args.ctx} batch{args.batch} target{args.target} "
        f"schedule={args.ctx_schedule} compile={args.compile}",
        flush=True,
    )

    model.train()
    step_ms, curve, wall_steady = [], [], 0.0
    for step in range(args.steps):
        lr = lr_at(
            step,
            args.steps,
            args.lr,
            args.lr_final_ratio,
            args.warmup,
            args.schedule,
            args.decay_frac,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        ctx, batch = ctx_at(step, args.steps, schedule, args.ctx, args.batch)
        x, y = sample_batch(train_tokens, batch, ctx, dev, gen)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp():
            if args.logit_softcap > 0:
                logits = model(x)  # [B,T,V], no internal CE
                cap = args.logit_softcap
                logits = cap * torch.tanh(logits.float() / cap)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
            else:
                loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_ms.append(dt * 1e3)
        if step >= 30:  # exclude compile/warmup ramp from steady-state wall-clock
            wall_steady += dt
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpc = val_bpc(model, val_batches, amp)
            curve.append((step + 1, wall_steady, bpc))
            print(
                f"  step {step + 1:5d}  wall_steady {wall_steady:6.1f}s  val_bpc {bpc:.4f}",
                flush=True,
            )

    steady = sorted(step_ms[30:])[len(step_ms[30:]) // 2] if len(step_ms) > 30 else step_ms[-1]
    hit = next(((s, w) for s, w, bpc_ in curve if bpc_ <= args.target), None)
    final_bpc = curve[-1][2]
    ttt = f"{hit[1]:.1f}s @ step {hit[0]}" if hit else f"not reached (best {final_bpc:.4f})"
    print(
        f"[{args.label}] RESULT: steady {steady:.1f} ms/step | "
        f"wall-to-{args.target}BPC: {ttt} | final {final_bpc:.4f} @ {args.steps} steps",
        flush=True,
    )
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# Speedrun leaderboard — 46M (12L/512d) enwik8, wall-clock to target val BPC\n\n"
            "Steady-state ms/step excludes the compile/warmup ramp. wall-to-target is "
            "steady-state seconds to reach the target BPC (proxy for the full run).\n\n"
            "| label | target | steady ms/step | wall-to-target | final BPC @ steps |\n"
            "|---|---:|---:|---|---|\n"
        )
    with LEADERBOARD.open("a") as f:
        f.write(
            f"| {args.label} | {args.target} | {steady:.1f} | {ttt} | "
            f"{final_bpc:.4f} @ {args.steps} |\n"
        )


if __name__ == "__main__":
    main()
