"""Per-token energy estimate for SpikeGPT: dense-ANN vs spike-driven AC scenarios.

Token-level perplexity aside, the SNN pitch is *energy*: a binary spike turns a
spike-fed ``Linear``'s multiply-accumulate (MAC) into a multiply-free accumulate
(AC), and silent neurons cost ~nothing on event-driven hardware. This script
makes that concrete for a SpikeGPT config — it counts the per-token matmul MACs
by component, then prices them under three scenarios:

* **dense-ANN** — every matmul is a MAC (what a same-size transformer pays).
* **canonical-SNN (optimistic)** — the projections fed by spikes do AC, scaled by
  the measured firing rate ``r`` (the headline SNN claim; *requires*
  LIF-before-Linear so a projection's input is a spike train).
* **as-built** — our impl places the LIF *after* each sub-block and adds the
  spikes to a *continuous* residual, so every projection reads continuous values
  => all MAC => **no AC win**. The WKV time-mixing (exp/division) is continuous in
  every scenario, as is the LM head.

Energy-per-op defaults to the 45 nm CMOS figures the SpikeGPT paper itself uses
(Horowitz, ISSCC 2014): a MAC ≈ 4.5 pJ, an accumulate ≈ 0.9 pJ. Override with
``--e-mac`` / ``--e-ac``. ``--firing-rate`` is the fraction of block neurons that
spike (our 216M measures ~0.34). The model is recurrence-bound, so the LIF/WKV
elementwise ops are negligible next to the projections and are reported but not
priced as matmuls.
"""

from __future__ import annotations

import argparse

from spikegpt.language import SPIKEGPT_PRESETS


def matmul_macs_per_token(n_layer: int, n_embd: int, vocab_size: int) -> dict[str, int]:
    """Per-token matmul MACs by component (the dominant cost; biases ignored).

    Token-mix has 4 projections (key, value, receptance, output), each n_embd^2.
    Channel-mix has key (n_embd->4*n_embd), value (4*n_embd->n_embd) and receptance
    (n_embd->n_embd) = 9*n_embd^2. The head is n_embd*vocab.
    """
    token_mix = n_layer * 4 * n_embd * n_embd
    channel_mix = n_layer * 9 * n_embd * n_embd
    head = n_embd * vocab_size
    return {"token_mix": token_mix, "channel_mix": channel_mix, "head": head}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(SPIKEGPT_PRESETS), default="gpt2-216m")
    parser.add_argument("--layers", type=int, help="override preset n_layer")
    parser.add_argument("--embedding", type=int, help="override preset n_embd")
    parser.add_argument("--vocab-size", type=int, default=50277)
    parser.add_argument(
        "--firing-rate",
        type=float,
        default=0.34,
        help="fraction of block neurons that spike (our 216M measures ~0.34)",
    )
    parser.add_argument(
        "--e-mac", type=float, default=4.5, help="pJ per MAC (SpikeGPT/Horowitz 45nm: 4.5)"
    )
    parser.add_argument("--e-ac", type=float, default=0.9, help="pJ per accumulate (45nm: 0.9)")
    args = parser.parse_args()

    spec = SPIKEGPT_PRESETS[args.preset]
    n_layer = args.layers or spec.n_layer
    n_embd = args.embedding or spec.n_embd
    macs = matmul_macs_per_token(n_layer, n_embd, args.vocab_size)
    total = sum(macs.values())
    # Projections fed by the (spiking) residual — the AC-capable ones IF the LIF
    # precedes the Linear. The head reads the final continuous LayerNorm, never a
    # spike, so it is MAC in every scenario.
    spike_capable = macs["token_mix"] + macs["channel_mix"]
    r, e_mac, e_ac = args.firing_rate, args.e_mac, args.e_ac

    dense = total * e_mac
    # canonical SNN: spike-capable projections do AC at the firing rate; head MAC.
    canonical = spike_capable * r * e_ac + macs["head"] * e_mac
    # as-built: spikes feed a continuous residual -> every projection is MAC.
    as_built = total * e_mac

    def line(label: str, pj: float) -> str:
        return f"| {label} | {pj / 1e6:,.2f} | {dense / pj:.2f}× |"

    print(
        f"# SpikeGPT per-token energy estimate — {args.preset} "
        f"({n_layer}L/{n_embd}d, vocab {args.vocab_size})"
    )
    print()
    print(
        f"matmul MACs/token: {total:,}  "
        f"(token-mix {macs['token_mix']:,} + channel-mix {macs['channel_mix']:,} "
        f"+ head {macs['head']:,})"
    )
    print(f"firing rate r={r:.2f}   E_MAC={e_mac} pJ   E_AC={e_ac} pJ")
    print()
    per_proj = e_mac / (r * e_ac)
    print("| scenario | energy/token (µJ, 45nm) | vs dense |")
    print("|---|---:|---:|")
    print(line("dense ANN (all MAC)", dense))
    print(line(f"SpikeGPT Table-1 *claim* (projections as AC×r={r:.2f})", canonical))
    print(line("as-built = canonical SpikeGPT (Linear→LIF; all MAC)", as_built))
    print()
    print(
        f"per-projection idealized ratio E_MAC/(r·E_AC) = {per_proj:.1f}× "
        f"(paper's Table 1 uses r=0.15 → 33.3×)."
    )
    print(
        "\nNote: the 'SpikeGPT Table-1 claim' row is the paper's energy accounting — it "
        "prices the projection matmuls as firing-rate-scaled accumulate (AC) ops. But "
        "that requires a spike train as each Linear's INPUT (LIF→Linear). The CANONICAL "
        "SpikeGPT code (and this faithful reproduction) place the LIF AFTER each "
        "sub-block and add spikes to a CONTINUOUS residual (Linear→LIF), so every "
        "projection reads continuous floats and stays MAC — the 'as-built' row, which is "
        "what the released architecture actually computes. The WKV (exp/division) and the "
        "LM head are continuous MAC in every scenario; the large BPE-vocab head further "
        "dilutes even the idealized savings (vs the paper's tiny 256-byte-vocab head)."
    )


if __name__ == "__main__":
    main()
