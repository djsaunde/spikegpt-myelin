"""SpiNNaker2 feasibility + hardware-graph mapping for a trained SpikeGPT.

Two deliverables toward a neuromorphic port (no hardware/SDK access needed):

1. **Hardware-graph decomposition** — walk the model and classify every weighted
   op as ``spike-synaptic`` (a Linear whose INPUT is spikes -> the chip's
   event-driven synapse array / sparse compute), ``continuous-ARM`` (matmul / WKV /
   LayerNorm fed by continuous values -> hand-written FP32 C on the Cortex-M4F PEs,
   the EGRU pattern), or ``spiking`` (a weightless LIF population). The split is
   architecture-dependent: canonical ``Linear->LIF`` puts the LIF after each
   sub-block, so the projections read the *continuous* residual and are all
   continuous-ARM; ``spike_input`` (``LIF->Linear``) moves the input projections to
   spike-synaptic (see spikegpt_216m_energy.md).

2. **Per-PE memory feasibility** — SpiNNaker2: 152 PEs/chip, 128 KB SRAM/PE. Weight
   storage is the binding constraint (EGRU, arXiv:2312.09084, was memory-bound at
   batch 1). Reports PE/chip counts for the weights under fp32 / int8 / pruned-CSR.

Reference HW figures: arXiv:2401.04491 (architecture), arXiv:2312.09084 (EGRU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spikegpt.language import SPIKEGPT_PRESETS

# SpiNNaker2 (arXiv:2401.04491): per-PE SRAM and PEs per chip.
PE_SRAM_KB = 128
PES_PER_CHIP = 152


def weight_breakdown(n_layer: int, n_embd: int, vocab: int, ffn_mult: int = 4) -> dict[str, int]:
    """Weight counts by component (matches the trained model's named_parameters)."""
    hidden = ffn_mult * n_embd
    return {
        # token-mix projections: key, value, receptance (each n_embd^2) ...
        "att.kvr_proj": n_layer * 3 * n_embd * n_embd,
        "att.output": n_layer * n_embd * n_embd,  # ... and the output projection
        # channel-mix: key n_embd->hidden, value hidden->n_embd, receptance n_embd^2
        "ffn_proj": n_layer * (2 * n_embd * hidden + n_embd * n_embd),
        "embedding": vocab * n_embd,
        "head": n_embd * vocab,
    }


def classify(component: str, spike_input: bool) -> str:
    """Map a weight component to its SpiNNaker2 target.

    A projection is ``spike-synaptic`` only if its INPUT is a spike train. With the
    canonical Linear->LIF placement nothing is (the residual is continuous); with
    spike_input the sub-block *input* projections (att.kvr, ffn key/receptance) are.
    The att.output and ffn.value read WKV/relu^2 outputs (continuous) either way.
    """
    if component in ("embedding", "head"):
        return "continuous-ARM"
    if not spike_input:
        return "continuous-ARM"
    # spike_input: input projections consume spikes; output/value still continuous.
    if component == "att.kvr_proj":
        return "spike-synaptic"
    if component == "ffn_proj":
        return "spike-synaptic (input parts)"
    return "continuous-ARM"


def feasibility(total_params: int, bytes_per_w: float, usable_kb: float, label: str) -> str:
    total_bytes = total_params * bytes_per_w
    per_pe = usable_kb * 1024
    pes = total_bytes / per_pe
    chips = pes / PES_PER_CHIP
    return f"| {label} | {total_bytes / 1e6:,.1f} | {pes:,.0f} | {chips:,.1f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="read config from a checkpoint")
    parser.add_argument("--preset", choices=tuple(SPIKEGPT_PRESETS), default="custom")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--embedding", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--spike-input", action="store_true", help="LIF->Linear variant")
    parser.add_argument(
        "--usable-kb",
        type=float,
        default=96.0,
        help="per-PE KB usable for weights (128 KB total minus code+state)",
    )
    parser.add_argument("--prune", type=float, default=0.9, help="sparsity for the CSR row")
    args = parser.parse_args()

    if args.checkpoint is not None:
        from spikegpt import load_spike_language_checkpoint

        cfg = load_spike_language_checkpoint(args.checkpoint, map_location="cpu").model.config
        n_layer, n_embd, vocab, spike_input = (
            cfg.n_layer,
            cfg.n_embd,
            cfg.vocab_size,
            cfg.spike_input,
        )
    elif args.preset != "custom":
        spec = SPIKEGPT_PRESETS[args.preset]
        n_layer, n_embd, vocab, spike_input = (
            spec.n_layer,
            spec.n_embd,
            args.vocab_size,
            args.spike_input,
        )
    else:
        n_layer, n_embd, vocab, spike_input = (
            args.layers,
            args.embedding,
            args.vocab_size,
            args.spike_input,
        )

    weights = weight_breakdown(n_layer, n_embd, vocab)
    total = sum(weights.values())

    print(f"# SpikeGPT -> SpiNNaker2 feasibility — {n_layer}L/{n_embd}d, vocab {vocab}")
    print(f"placement: {'LIF->Linear (spike_input)' if spike_input else 'Linear->LIF (canonical)'}")
    print()
    print("## Hardware-graph decomposition (weighted ops)")
    print("| component | params | % | SpiNNaker2 target |")
    print("|---|---:|---:|---|")
    targets: dict[str, int] = {}
    for comp, n in sorted(weights.items(), key=lambda kv: -kv[1]):
        tgt = classify(comp, spike_input)
        targets[tgt] = targets.get(tgt, 0) + n
        print(f"| {comp} | {n:,} | {100 * n / total:.1f}% | {tgt} |")
    print(f"| **total** | **{total:,}** | 100% | |")
    print()
    print(
        "weightless ops: 2*n_layer LIF populations (spiking fabric, the only true "
        "event-driven part), plus WKV / LayerNorm / token-shift / residual adds "
        "(continuous-ARM / routing)."
    )
    print()
    print("## Weight share by target")
    for tgt, n in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"- {tgt}: {n / 1e6:.1f}M ({100 * n / total:.1f}%)")
    print()
    print(
        f"## Per-PE memory feasibility ({PE_SRAM_KB} KB/PE, {PES_PER_CHIP} PEs/chip, "
        f"{args.usable_kb:.0f} KB/PE usable for weights)"
    )
    print("| weight format | MB | PEs | chips |")
    print("|---|---:|---:|---:|")
    print(feasibility(total, 4.0, args.usable_kb, "fp32 dense (EGRU ran FP32 on ARM)"))
    print(feasibility(total, 1.0, args.usable_kb, "int8 dense (MAC accelerator format)"))
    keep = 1.0 - args.prune
    # CSR ~ value (int8) + ~2-byte column index per nonzero.
    print(
        feasibility(
            int(total * keep),
            3.0,
            args.usable_kb,
            f"int8 CSR, {args.prune * 100:.0f}% pruned (~3 B/nonzero)",
        )
    )
    print()
    print(
        "EGRU (arXiv:2312.09084), a *smaller* event-based LM, was memory-bound at "
        "batch 1 with 95%-pruned sparse-CSR weights. Dense storage here implies a "
        "multi-chip, batch-1 deployment; pruning needs retraining (accuracy hit TBD)."
    )


if __name__ == "__main__":
    main()
