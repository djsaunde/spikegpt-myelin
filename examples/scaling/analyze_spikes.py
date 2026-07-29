"""Per-layer spike-rate, dead-neuron, and energy analysis of trained SpikeGPT checkpoints.

Runs a forward pass over a fixed val slice with hooks on each block's LIF modules
(lif1 = time-mix spikes, lif2 = channel-mix spikes), accumulating per-channel firing.
From that we report, per block and overall:
  * spike rate  — mean fraction of (neuron, token) pairs that fire
  * dead-neuron rate — fraction of channels that NEVER fire across the whole val slice

Energy (SpikeGPT-paper method, Horowitz 45nm): a spike-driven matmul does accumulate-only
ops (AC) gated by its input spike rate, vs a dense ANN's multiply-accumulate (MAC). We price
the block projections as AC at the measured per-sublayer rate and the LM head as dense MAC,
and report the ANN/SNN energy ratio.

    uv run --extra cuda --extra tracking python examples/scaling/analyze_spikes.py \
        --checkpoints runs/c3e17_512d.best.pt runs/c1e19_1280d.best.pt --out runs/spike_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spikegpt import MemmapTokenCorpus, load_spike_language_checkpoint, sample_token_batch

# Horowitz 45nm, 32-bit (the figures the SpikeGPT paper used), picojoules
E_MAC = 4.6
E_AC = 0.9


@torch.no_grad()
def analyze(ckpt_path: str, corpus_path: str, *, val_holdout: int, batch: int,
            ctx: int, n_batches: int, device: str = "cuda") -> dict:
    ck = load_spike_language_checkpoint(ckpt_path, map_location=device)
    model = ck.model.to(device).eval()
    cfg = model.config
    C, vocab = cfg.n_embd, cfg.vocab_size

    corpus = MemmapTokenCorpus.open(corpus_path)
    # Use the held-out tail (the region training reserved for validation).
    total = len(corpus)
    val = corpus[total - val_holdout:]

    # Accumulate per-channel firing on each block's two LIF outputs.
    stats: dict[str, dict] = {}

    def mk_hook(key: str):
        def hook(_m, _inp, out):
            s = out.detach()
            s = s.reshape(-1, s.shape[-1])          # (tokens, channels)
            fired = (s > 0)
            per_ch = fired.sum(dim=0).float()       # (channels,)
            ever = fired.any(dim=0)                 # (channels,)
            if key not in stats:
                stats[key] = {"sum": per_ch.clone(), "ever": ever.clone(), "tok": s.shape[0]}
            else:
                stats[key]["sum"] += per_ch
                stats[key]["ever"] |= ever
                stats[key]["tok"] += s.shape[0]
        return hook

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.lif1.register_forward_hook(mk_hook(f"blocks.{i}.time")))
        handles.append(blk.lif2.register_forward_hook(mk_hook(f"blocks.{i}.channel")))

    for _ in range(n_batches):
        x, _ = sample_token_batch(val, batch_size=batch, context_length=ctx, device=device)
        model(x)
    for h in handles:
        h.remove()

    # Per-block rates + dead rates.
    per_layer = []
    for i in range(cfg.n_layer):
        t, c = stats[f"blocks.{i}.time"], stats[f"blocks.{i}.channel"]
        per_layer.append({
            "layer": i,
            "time_rate": float((t["sum"] / t["tok"]).mean()),
            "channel_rate": float((c["sum"] / c["tok"]).mean()),
            "time_dead": float((~t["ever"]).float().mean()),
            "channel_dead": float((~c["ever"]).float().mean()),
        })
    # Overall (mean over both sublayers, all layers).
    all_rate = sum(p["time_rate"] + p["channel_rate"] for p in per_layer) / (2 * cfg.n_layer)
    all_dead = sum(p["time_dead"] + p["channel_dead"] for p in per_layer) / (2 * cfg.n_layer)

    # Energy: block time-mix = 4C^2 params (gated by lif1 rate), channel-mix = 9C^2 (lif2 rate);
    # LM head = vocab*C dense. Per-token ops; tokens cancel in the ratio.
    e_snn_blocks = sum(p["time_rate"] * 4 * C * C + p["channel_rate"] * 9 * C * C
                       for p in per_layer) * E_AC
    e_ann_blocks = cfg.n_layer * (4 + 9) * C * C * E_MAC
    head = vocab * C * E_MAC  # dense in both (head input is the continuous ln_out stream)
    return {
        "checkpoint": ckpt_path, "n_embd": C, "n_layer": cfg.n_layer,
        "n_params": sum(p.numel() for p in model.parameters()),
        "overall_spike_rate": all_rate, "overall_dead_rate": all_dead,
        # full-model savings (dense vocab head included) and blocks-only (the pure
        # spiking-compute advantage, à la the SpikeGPT paper).
        "energy_ratio_full": (e_ann_blocks + head) / (e_snn_blocks + head),
        "energy_ratio_blocks_only": e_ann_blocks / e_snn_blocks,
        "per_layer": per_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--corpus", default="data/fineweb-edu_sample-100BT_25000000000.bin")
    ap.add_argument("--val-holdout", type=int, default=50_000_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--out", default="runs/spike_analysis.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results = []
    for c in args.checkpoints:
        print(f"analyzing {c} ...", flush=True)
        r = analyze(c, args.corpus, val_holdout=args.val_holdout, batch=args.batch,
                    ctx=args.ctx, n_batches=args.n_batches, device=args.device)
        print(f"  N={r['n_params']/1e6:.0f}M  spike_rate={r['overall_spike_rate']:.3f}  "
              f"dead={r['overall_dead_rate']:.3f}  energy full={r['energy_ratio_full']:.1f}x  "
              f"blocks-only={r['energy_ratio_blocks_only']:.1f}x", flush=True)
        results.append(r)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
