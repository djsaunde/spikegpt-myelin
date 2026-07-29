"""Plot the spike-rate / dead-neuron / energy analysis from analyze_spikes.py.

    uv run --with matplotlib python examples/scaling/plot_spikes.py

Reads runs/spike_analysis.json and writes three figures to runs/:
  spike_rate_per_layer.png  — per-layer spike rate vs relative depth (2 panels)
  dead_neuron_per_layer.png — per-layer dead-neuron rate (2 panels)
  energy_vs_scale.png       — energy savings vs model size (full-model + blocks-only)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The two target sets (checkpoint stem -> label), matched to the analysis run.
SET_A = [  # compute-optimal per FLOP tier
    ("c3e17_512d", "c3e17 · 92M"), ("c1e18_640d", "c1e18 · 139M"),
    ("c3e18_1024d", "c3e18 · 376M"), ("c1e19_1280d", "c1e19 · 640M"),
]
SET_B = [  # c1e19 width sweep (fixed 1e19 compute)
    ("c1e19_640d", "139M"), ("c1e19_768d", "215M"),
    ("c1e19_1024d", "376M"), ("c1e19_1280d", "640M"),
]


def load():
    return {r["checkpoint"]: r for r in json.load(open("runs/spike_analysis.json"))}


def _mean_layer(r, field_a, field_b):
    return np.array([(p[field_a] + p[field_b]) / 2 for p in r["per_layer"]])


def per_layer_fig(data, field_a, field_b, ylabel, title, out, ylim=None, pct=False):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, sset, sub in ((axA, SET_A, "A · compute-optimal per tier"),
                          (axB, SET_B, "B · c1e19 width sweep (fixed 1e19)")):
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sset)))
        for (stem, lab), col in zip(sset, cmap):
            r = data[f"runs/{stem}.best.pt"]
            y = _mean_layer(r, field_a, field_b) * (100 if pct else 1)
            x = np.linspace(0, 1, len(y))
            ax.plot(x, y, "o-", color=col, ms=4, lw=1.6, label=lab)
        ax.set_title(sub, fontsize=10, loc="left")
        ax.set_xlabel("relative depth  (input → output)")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=8.5, title="model")
        if ylim:
            ax.set_ylim(*ylim)
    axA.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"runs/{out}", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote runs/{out}")


def energy_fig(data, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    N = np.array([data[f"runs/{s}.best.pt"]["n_params"] / 1e6 for s, _ in SET_A])
    full = np.array([data[f"runs/{s}.best.pt"]["energy_ratio_full"] for s, _ in SET_A])
    blk = np.array([data[f"runs/{s}.best.pt"]["energy_ratio_blocks_only"] for s, _ in SET_A])
    ax.plot(N, full, "o-", color="#2f5fa6", lw=2, ms=8, label="full model (incl. dense vocab head)")
    ax.plot(N, blk, "s--", color="#b83a3a", lw=2, ms=8, label="blocks only (spiking compute)")
    for x, y in zip(N, full):
        ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8.5, color="#2f5fa6")
    ax.set_xscale("log")
    ax.set_xlabel("model size  N  (total params incl. embedding, M)")
    ax.set_ylabel("energy savings vs dense ANN  (×)")
    ax.set_title("Energy savings vs scale  (Horowitz 45nm; E_MAC 4.6pJ, E_AC 0.9pJ)",
                 fontsize=10.5, loc="left")
    ax.set_ylim(0, max(blk.max(), full.max()) * 1.15)
    ax.grid(alpha=.25, which="both")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.annotate("full-model savings climbs toward the blocks-only ceiling\n"
                "as the dense vocab head shrinks with scale",
                (0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=8, color="#666", style="italic")
    fig.tight_layout()
    fig.savefig(f"runs/{out}", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote runs/{out}")


def main():
    data = load()
    per_layer_fig(data, "time_rate", "channel_rate", "spike rate  (fraction firing)",
                  "Per-layer spike rate — sparse early, denser late", "spike_rate_per_layer.png")
    per_layer_fig(data, "time_dead", "channel_dead", "dead-neuron rate  (%, never fires)",
                  "Per-layer dead neurons — ≈0, except a mid-network pocket at 376M",
                  "dead_neuron_per_layer.png", ylim=(0, 7.5), pct=True)
    energy_fig(data, "energy_vs_scale.png")


if __name__ == "__main__":
    main()
