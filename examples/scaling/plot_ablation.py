"""Plot per-layer time-mix ablation Δloss (from ablate_timemix.py --out runs/ablation.json).

    uv run --with matplotlib python examples/scaling/plot_ablation.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"c3e18_1024d": "376M · L20 (c3e18)", "c1e19_1024d": "376M · L20 (c1e19)",
          "c1e19_1280d": "640M · L24 (c1e19)"}
COL = {"c3e18_1024d": "#54A24B", "c1e19_1024d": "#2f5fa6", "c1e19_1280d": "#E45756"}


def main() -> None:
    rows = {r["checkpoint"].split("/")[-1].replace(".best.pt", ""): r
            for r in json.load(open("runs/ablation.json"))}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for stem, lab in LABELS.items():
        if stem not in rows:
            continue
        r = rows[stem]
        d = np.array(r["delta"]) * 1000  # milli-nats for readability
        x = np.linspace(0, 1, len(d))
        ax.plot(x, d, "o-", color=COL[stem], ms=5, lw=1.7, label=lab)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("relative depth  (input → output)")
    ax.set_ylabel("Δ val loss from ablating that layer's time-mix  (milli-nats)")
    ax.set_title("Time-mix layer importance — the middle band is nearly free to remove",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=9, title="model")
    ax.annotate("d1024/L20: a middle layer is fully redundant (Δ≈0)\n"
                "d1280/L24: milder mid-dip, none free; late layers matter most",
                (0.5, 0.96), xycoords="axes fraction", ha="center", va="top",
                fontsize=8, color="#666", style="italic")
    fig.tight_layout()
    fig.savefig("runs/timemix_ablation.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote runs/timemix_ablation.png")


if __name__ == "__main__":
    main()
