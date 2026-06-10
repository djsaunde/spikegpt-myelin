"""A/B: does the prototype bf16-I/O LIF hurt convergence vs the fp32 LIF?

Trains a small SpikeGPT twice — identical recipe, only the LIF implementation
differs — and prints the val-BPC trajectory for each arm. If the bf16-I/O arm
tracks the fp32 arm, the ~1% gradient difference is harmless and the kernel can
be promoted into lif.py for the ~7% step-time win.

Run (per arm):
  python examples/lif_convergence_ab.py --text-file data/enwik8 --lif bf16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--lif", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--embedding", type=int, default=512)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-every", type=int, default=1000)
    args = p.parse_args()

    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    toks = torch.frombuffer(bytearray(args.text_file.read_bytes()), dtype=torch.uint8).to(
        torch.long
    )
    split = int(toks.numel() * 0.95)
    train, val = toks[:split], toks[split:]
    t_len, batch = args.context_length, args.batch

    def sample(src: torch.Tensor, bs: int):
        ix = torch.randint(0, src.numel() - t_len - 1, (bs,))
        return (
            torch.stack([src[i : i + t_len] for i in ix]).to(dev),
            torch.stack([src[i + 1 : i + t_len + 1] for i in ix]).to(dev),
        )

    if args.lif == "bf16":
        # Swap SpikingSequenceLIF's forward for the bf16-I/O prototype kernel.
        from myelin.neurons import LIFParams, LIFState
        from myelin.triton.lif_bf16 import surrogate_lif_bf16io

        import spikegpt.language as lang

        def bf16_forward(self, inputs):
            currents = inputs.movedim(1, 0).contiguous()
            init = LIFState(membrane=inputs.new_zeros(inputs.shape[0], inputs.shape[2]))
            params = LIFParams(tau_mem=self.tau, threshold=self.threshold, reset=self.reset)
            spikes = surrogate_lif_bf16io(
                currents, init, params, surrogate_slope=self.surrogate_slope
            )
            return spikes.movedim(0, 1)

        lang.SpikingSequenceLIF.forward = bf16_forward

    cfg = SpikeGPTConfig(
        vocab_size=256,
        context_length=t_len,
        n_layer=args.layers,
        n_embd=args.embedding,
        dropout=0.03,
    )
    model = SpikeLanguageModel(cfg).to(dev).train()
    for i, blk in enumerate(model.blocks):
        model.blocks[i] = torch.compile(blk)  # type: ignore[call-overload]
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=4e-9, weight_decay=0.1
    )

    @torch.no_grad()
    def val_bpc() -> float:
        model.eval()
        tot = 0.0
        for _ in range(8):
            x, y = sample(val, 32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            tot += torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256).float(), y.reshape(-1)
            ).item()
        model.train()
        return tot / 8 / math.log(2.0)

    print(
        f"[lif={args.lif}] {args.layers}L/{args.embedding}d ctx{t_len} B{batch} lr{args.lr}",
        flush=True,
    )
    ema = None
    for s in range(1, args.steps + 1):
        warm, total = 200, args.steps
        if s < warm:
            lr = args.lr * s / warm
        else:
            prog = min(1.0, (s - warm) / (total - warm))
            lr = 1e-4 + 0.5 * (args.lr - 1e-4) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = sample(train, batch)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ln = float(loss) / math.log(2.0)
        ema = ln if ema is None else 0.97 * ema + 0.03 * ln
        if s == 1 or s % args.eval_every == 0:
            print(
                f"  [{args.lif}] step {s:5d} train_bpc={ema:.3f} val_bpc={val_bpc():.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
