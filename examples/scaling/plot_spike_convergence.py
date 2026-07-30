"""Spike rate vs tokens-per-parameter — de-confounds the "sparser at scale" panel.

Panel C of the grid plot shows spike rate falling with model size, but spike rate also
rises with training. Since wider models in a FLOP tier see fewer tokens, plotting spike
rate against tokens/param (D/N) tests whether the size trend is really a training-
convergence trend: if the width curves collapse onto one rising, saturating curve, sparsity
is governed by how much each model was trained, not its size.

    uv run --with matplotlib --extra tracking python examples/scaling/plot_spike_convergence.py
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spikegpt.scaling import count_spikegpt_params

WIDTH_LAYERS = {384: 10, 512: 12, 640: 14, 768: 18, 1024: 20, 1280: 24, 1536: 29}


def main() -> None:
    import wandb

    api = wandb.Api()
    pts = []  # (tokens_per_param, spike_rate, width)
    for r in api.runs("pitheta/spikegpt-scaling"):
        m = re.match(r"c\de\d+_(\d+)d$", r.name or "")
        if not m:
            continue
        c, s = r.config, r.summary
        sp = c.get("spiking")
        sp = c.get("spike_embedding", True) if sp is None else sp
        if not (sp and (c.get("attention") or "rwkv") == "rwkv"):
            continue
        d = int(m.group(1))
        sr, st = s.get("train/mean_block_spike_rate"), c.get("total_steps")
        if sr is None or sr < 0.05 or st is None or d not in WIDTH_LAYERS:
            continue
        n = count_spikegpt_params(vocab_size=50277, n_layer=WIDTH_LAYERS[d], n_embd=d).total
        pts.append((st * c.get("batch", 16) * 1024 / n, sr, d))

    widths = sorted({p[2] for p in pts})
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for w, col in zip(widths, cmap):
        wp = sorted((x, y) for x, y, d in pts if d == w)
        params = count_spikegpt_params(vocab_size=50277, n_layer=WIDTH_LAYERS[w], n_embd=w).total
        ax.plot([p[0] for p in wp], [p[1] for p in wp], "o-", color=col, ms=6, lw=1.5,
                label=f"d{w} · {params/1e6:.0f}M")
    ax.axhline(0.36, ls=":", color="#888", lw=1)
    ax.annotate("saturation ~0.36", (ax.get_xlim()[0], 0.36), fontsize=8.5, color="#888",
                va="bottom", xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.15, ls=":", color="#b83a3a", lw=1)
    ax.annotate("SpikeGPT-46M paper (0.15)", (ax.get_xlim()[0], 0.15), fontsize=8.5,
                color="#b83a3a", va="bottom", xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens per parameter  (D / N)")
    ax.set_ylabel("mean block spike rate  (fraction firing)")
    ax.set_title("Spike rate is set by training, not size — width curves collapse on D/N",
                 fontsize=10.5, loc="left")
    ax.set_ylim(0.10, 0.42)
    ax.grid(alpha=.25, which="both")
    ax.legend(frameon=False, fontsize=8.5, title="width", ncol=2)
    fig.tight_layout()
    fig.savefig("runs/spike_vs_tokens_per_param.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote runs/spike_vs_tokens_per_param.png")


if __name__ == "__main__":
    main()
