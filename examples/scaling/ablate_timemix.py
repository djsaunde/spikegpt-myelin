"""Ablate each block's time-mix (RWKV token-mixing) one at a time and measure val-loss impact.

Tests whether the d1024 mid-network dead pocket (L9 time-mix) is a redundant sublayer the
model pruned itself: hook block[L].att to output zeros (so residual += 0 for that time-mix),
and compare val loss to baseline on a FIXED set of val batches. A near-zero Delta => the
layer's token-mixing is doing little; a large Delta => it matters.

    uv run --extra cuda --extra tracking python examples/scaling/ablate_timemix.py \
        --checkpoint runs/c1e19_1024d.best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from spikegpt import MemmapTokenCorpus, load_spike_language_checkpoint, sample_token_batch


@torch.no_grad()
def val_loss(model, batches) -> float:
    tot = 0.0
    for x, y in batches:
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
    return tot / len(batches)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", default="data/fineweb-edu_sample-100BT_25000000000.bin")
    ap.add_argument("--val-holdout", type=int, default=50_000_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="append per-layer results to this JSON")
    args = ap.parse_args()

    ck = load_spike_language_checkpoint(args.checkpoint, map_location=args.device)
    model = ck.model.to(args.device).eval()
    corpus = MemmapTokenCorpus.open(args.corpus)
    val = corpus[len(corpus) - args.val_holdout:]

    # Fix the val batches once so baseline and every ablation see identical data.
    torch.manual_seed(0)
    batches = [sample_token_batch(val, batch_size=args.batch, context_length=args.ctx,
                                  device=args.device) for _ in range(args.n_batches)]

    base = val_loss(model, batches)
    print(f"{args.checkpoint}  ({model.config.n_layer} layers)  baseline val loss = {base:.4f}\n")
    print(f"{'layer':>6}{'val loss':>11}{'Δ vs baseline':>15}")
    deltas = []
    for L in range(model.config.n_layer):
        h = model.blocks[L].att.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        l = val_loss(model, batches)
        h.remove()
        deltas.append(l - base)
        flag = "  <-- dead-pocket layer" if L == 9 else ""
        print(f"{L:>6}{l:>11.4f}{l - base:>+15.4f}{flag}")

    if args.out:
        p = Path(args.out)
        rows = json.loads(p.read_text()) if p.exists() else []
        rows = [r for r in rows if r["checkpoint"] != args.checkpoint]
        rows.append({"checkpoint": args.checkpoint, "n_layer": model.config.n_layer,
                     "n_embd": model.config.n_embd, "baseline": base, "delta": deltas})
        p.write_text(json.dumps(rows, indent=2))
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
