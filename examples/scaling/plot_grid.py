"""Grid-wide scaling plots — regenerate anytime as more tiers finish.

Three panels that grow with the grid:
  A. isoFLOP parabolas (loss vs N) per FLOP tier, with each tier's compute-optimal N.
  B. the compute-optimal frontier N_opt(C) and D_opt(C) (the headline exponents).
  C. spike rate vs model size across ALL runs (SpikeGPT-specific — does internal
     sparsity keep improving with scale?).

    uv run --with matplotlib --extra tracking python examples/scaling/plot_grid.py

Saves runs/grid_plots.png. Pull + dedup logic is shared with fit_isoflop.py.
"""

import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter

from fit_isoflop import pull, _vertex  # same directory

# one colour per FLOP tier (extends as tiers are added)
TIER_COL = {"3e+17": "#4C78A8", "1e+18": "#F58518", "3e+18": "#54A24B", "1e+19": "#E45756"}


def main() -> None:
    rows = pull("pitheta", "spikegpt-scaling")
    tiers = defaultdict(list)
    for r in rows:
        tiers[f"{r['C']:.0e}"].append(r)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.0))
    frontier = []  # (C, N_opt, D_opt)
    opt_pts = []  # (N_opt, loss at the optimum) — traces the compute-optimal trajectory

    # ---- A: isoFLOP parabolas per tier ----
    for cf in sorted(tiers, key=float):
        ts = sorted(tiers[cf], key=lambda r: r["N"])
        col = TIER_COL.get(cf, "#888")
        N = np.array([t["N"] for t in ts], float)
        L = np.array([t["val_loss"] for t in ts])
        axA.plot(N / 1e6, L, "o", ms=9, color=col, mec="white", mew=1.2, zorder=4,
                 label=f"C={cf} ({len(ts)}w)")
        if len(ts) >= 3:
            a2, a1, a0 = np.polyfit(np.log(N), L, 2)
            xs = np.logspace(math.log10(N.min() * 0.8), math.log10(N.max() * 1.2), 150)
            axA.plot(xs / 1e6, a2 * np.log(xs) ** 2 + a1 * np.log(xs) + a0, "-", color=col, lw=1.8)
            logNo, interior = _vertex(np.log(N), L)
            No = math.exp(logNo)
            Lo = a2 * logNo ** 2 + a1 * logNo + a0
            axA.plot(No / 1e6, Lo, "*", ms=15, color=col, mec="white", mew=1.0, zorder=5)
            if interior:
                frontier.append((float(cf), No, float(cf) / (6 * No)))
                opt_pts.append((No, Lo))

    # The compute-optimal trajectory: the path the sweet spot traces as compute grows.
    # This is the scaling law projected onto (N, loss). Dashed beyond the measured
    # optima = a prediction the remaining tiers will test. Kept short: extrapolating a
    # straight line here forever would eventually cross the irreducible loss floor.
    if len(opt_pts) >= 2:
        op = sorted(opt_pts)
        sl, ic = np.polyfit(np.log([p[0] for p in op]), [p[1] for p in op], 1)
        meas = np.logspace(math.log10(op[0][0]), math.log10(op[-1][0]), 30)
        # Extrapolate only as far as the next tier's predicted optimum — a straight
        # line in (log N, loss) has no irreducible-loss floor, so a long extension
        # would be dishonest (and would squash the measured parabolas off-scale).
        pred = np.logspace(math.log10(op[-1][0]), math.log10(2.0e8), 30)
        axA.plot(meas / 1e6, sl * np.log(meas) + ic, "-", color="#666", lw=1.8, zorder=3,
                 label=f"compute-optimal path ({len(op)} tiers)")
        axA.plot(pred / 1e6, sl * np.log(pred) + ic, ":", color="#666", lw=1.8, zorder=3,
                 label="→ preliminary extrapolation")
        axA.axvline(138, ls="-", color="#c0392b", lw=1.1, alpha=.5)
        axA.annotate("d768 = local VRAM ceiling →\npredicted optima lie beyond it",
                     (138, 3.97), fontsize=7.8, color="#c0392b", ha="right", va="top",
                     xytext=(-5, 0), textcoords="offset points")
    axA.set_ylim(3.15, 4.0)
    axA.set_xscale("log")
    axA.set_xlabel("model size  N  (non-embedding params, M)")
    axA.set_ylabel("final val loss  (nats/token)")
    axA.set_title("A · isoFLOP parabolas — sweet spot per compute budget", fontsize=11, loc="left")
    axA.legend(frameon=False, fontsize=9, title="FLOP tier  (★ = optimum)")
    axA.grid(alpha=.25, which="both")
    axA.xaxis.set_minor_locator(NullLocator())
    axA.xaxis.set_major_locator(FixedLocator([19, 41, 75, 138, 273, 512]))
    axA.xaxis.set_major_formatter(FixedFormatter(["19", "41", "75", "138", "273", "512"]))

    # ---- B: compute-optimal frontier ----
    if len(frontier) >= 2:
        C = np.array([f[0] for f in frontier])
        No = np.array([f[1] for f in frontier])
        Do = np.array([f[2] for f in frontier])
        aN = np.polyfit(np.log(C), np.log(No), 1)[0]
        aD = np.polyfit(np.log(C), np.log(Do), 1)[0]
        cx = np.logspace(math.log10(C.min() / 2), math.log10(C.max() * 20), 50)
        axB.plot(cx, np.exp(np.polyval(np.polyfit(np.log(C), np.log(No), 1), np.log(cx))) / 1e6,
                 "-", color="#333", lw=2, label=f"N_opt ∝ C^{aN:.2f}")
        axB.plot(C, No / 1e6, "o", ms=11, color="#333", mec="white", mew=1.4, zorder=4)
        axB.plot(cx, np.exp(np.polyval(np.polyfit(np.log(C), np.log(Do), 1), np.log(cx))) / 1e9,
                 "--", color="#B0447A", lw=2, label=f"D_opt ∝ C^{aD:.2f}  (÷10³, B tok)")
        axB.plot(C, Do / 1e9, "s", ms=10, color="#B0447A", mec="white", mew=1.4, zorder=4)
        axB.set_yscale("log")
        note = "2 tiers — exact fit, firms up with more" if len(frontier) == 2 else f"{len(frontier)} tiers"
        axB.annotate(note, (0.5, 0.03), xycoords="axes fraction", ha="center",
                     fontsize=8.5, color="#888", style="italic")
    axB.set_xscale("log")
    axB.set_xlabel("compute  C  (FLOPs)")
    axB.set_ylabel("N_opt (M params)  /  D_opt (B tokens)")
    axB.set_title("B · compute-optimal frontier — the scaling exponents", fontsize=11, loc="left")
    axB.legend(frameon=False, fontsize=10, loc="upper left")
    axB.grid(alpha=.25, which="both")

    # ---- C: spike rate vs scale, all tiers ----
    for cf in sorted(tiers, key=float):
        ts = sorted(tiers[cf], key=lambda r: r["N"])
        col = TIER_COL.get(cf, "#888")
        Ns = [t["N"] / 1e6 for t in ts]
        bl = [t["block_spike"] for t in ts]
        if any(b is None for b in bl):
            continue
        axC.plot(Ns, bl, "o-", color=col, lw=1.8, ms=9, mec="white", mew=1.1, label=f"C={cf}")
    # embedding spike (≈const ~0.5) as a faint reference from all runs
    allN = [r["N"] / 1e6 for r in rows if r.get("emb_spike") is not None]
    allE = [r["emb_spike"] for r in rows if r.get("emb_spike") is not None]
    if allE:
        axC.plot(allN, allE, "s", color="#999", ms=6, alpha=.6, label="input embedding (~const)")
    axC.set_xscale("log")
    axC.set_xlabel("model size  N  (non-embedding params, M)")
    axC.set_ylabel("spike rate  (fraction firing)")
    axC.set_title("C · internal sparsity vs scale (novel)", fontsize=11, loc="left")
    axC.legend(frameon=False, fontsize=9, loc="center left")
    axC.grid(alpha=.25, which="both")
    axC.set_ylim(0.15, 0.55)
    axC.xaxis.set_minor_locator(NullLocator())
    axC.xaxis.set_major_locator(FixedLocator([19, 41, 75, 138, 273, 512]))
    axC.xaxis.set_major_formatter(FixedFormatter(["19", "41", "75", "138", "273", "512"]))

    fig.suptitle(f"SpikeGPT scaling grid — {len(rows)} runs, {len(tiers)} FLOP tiers",
                 fontsize=13.5, y=1.01)
    fig.tight_layout()
    fig.savefig("runs/grid_plots.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"{len(rows)} runs, {len(tiers)} tiers, {len(frontier)} on the frontier "
          f"-> runs/grid_plots.png")


if __name__ == "__main__":
    main()
