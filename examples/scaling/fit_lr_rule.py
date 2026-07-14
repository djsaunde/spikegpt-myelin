"""Fit the LR scaling rule lr_opt(d) = a * d^b from the Phase-1 pilot.

Two-stage fit, following the standard LR-sweep protocol:

1. Per width, fit a parabola in (log lr, val_loss) and take its vertex as that
   width's optimal LR. Three LRs bracketing the optimum make the parabola exact;
   a vertex on/outside the bracket is flagged (extend that width by one ~2.5x LR
   step and re-run -- the bracket missed the minimum).
2. Across widths, fit log(lr_opt) = log a + b*log d. The slope b is the headline:
   b ~= -1 means LR should scale as 1/width (muP / spectral-condition consistent),
   which is itself a reportable result, not just a knob.

Finally, print the suggested LR at each main-grid width (a * d^b) ready to paste
into examples/scaling/main_grid.json.

    uv run python examples/scaling/fit_lr_rule.py runs/pilot_results.json
    uv run python examples/scaling/fit_lr_rule.py runs/pilot_results.json --grid-widths 384 512 640 768 1024 1280
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Grid widths whose LR we ultimately want to set (examples/scaling/main_grid.json).
DEFAULT_GRID_WIDTHS = (384, 512, 640, 768, 1024, 1280)
# Relative margin: a vertex within this factor of a bracket end is "at the edge".
EDGE_MARGIN = 1.05


def fit_width_vertex(lrs: np.ndarray, losses: np.ndarray) -> tuple[float, bool, str]:
    """Return (lr_opt, interior, note) from a log-lr parabola fit for one width."""
    order = np.argsort(lrs)
    lrs, losses = lrs[order], losses[order]
    x = np.log(lrs)
    a2, a1, _ = np.polyfit(x, losses, 2)  # loss ~= a2*x^2 + a1*x + a0
    if a2 <= 0:
        # Not convex in log-lr: the sampled points are monotonic, so the optimum
        # lies beyond one end of the bracket.
        low = losses[0] < losses[-1]
        end = lrs[0] if low else lrs[-1]
        direction = "below" if low else "above"
        return float(end), False, f"NON-CONVEX: loss monotonic, optimum {direction} bracket; extend"
    lr_opt = float(np.exp(-a1 / (2 * a2)))
    lo, hi = float(lrs[0]), float(lrs[-1])
    if lr_opt < lo * EDGE_MARGIN:
        return lr_opt, False, f"vertex {lr_opt:.3g} at/below low end {lo:.3g}; extend down (~/2.5)"
    if lr_opt > hi / EDGE_MARGIN:
        return lr_opt, False, f"vertex {lr_opt:.3g} at/above high end {hi:.3g}; extend up (~x2.5)"
    return lr_opt, True, "interior"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("results", type=Path, help="pilot_results.json from pilot_results.py")
    parser.add_argument("--grid-widths", type=int, nargs="+", default=list(DEFAULT_GRID_WIDTHS))
    args = parser.parse_args()

    rows = json.loads(args.results.read_text())
    by_width: dict[int, list[tuple[float, float]]] = {}
    batches = {r.get("batch") for r in rows}
    for r in rows:
        by_width.setdefault(int(r["embedding"]), []).append((float(r["lr"]), float(r["val_loss"])))

    if len(batches) > 1:
        print(f"WARNING: mixed batch sizes {batches} -- lr_opt(d) is confounded by batch; "
              "re-run so every width shares one batch before trusting the fit.\n")

    print(f"{'d':>6} {'#lr':>4} {'lr_opt':>10} {'interior':>9}  note")
    fit_d, fit_lr = [], []
    for d in sorted(by_width):
        pts = sorted(by_width[d])
        lrs = np.array([p[0] for p in pts])
        losses = np.array([p[1] for p in pts])
        if len(pts) < 3:
            print(f"{d:>6} {len(pts):>4} {'-':>10} {'-':>9}  need >=3 LRs to fit a parabola")
            continue
        lr_opt, interior, note = fit_width_vertex(lrs, losses)
        print(f"{d:>6} {len(pts):>4} {lr_opt:>10.4g} {str(interior):>9}  {note}")
        if interior:
            fit_d.append(d)
            fit_lr.append(lr_opt)

    if len(fit_d) < 2:
        raise SystemExit(
            f"\nonly {len(fit_d)} interior width(s) -- need >=2 to fit lr_opt(d)=a*d^b. "
            "Extend the flagged brackets and re-run the pilot."
        )

    logd = np.log(np.array(fit_d, dtype=float))
    loglr = np.log(np.array(fit_lr))
    b, log_a = np.polyfit(logd, loglr, 1)
    a = float(np.exp(log_a))
    resid = loglr - (log_a + b * logd)
    rms = float(np.sqrt(np.mean(resid**2)))
    print(f"\nlr_opt(d) = {a:.4g} * d^({b:+.3f})   (fit on widths {fit_d}, log-resid RMS {rms:.3f})")
    muP = "~1/d (muP-consistent)" if abs(b + 1.0) < 0.15 else "NOT ~1/d"
    print(f"slope b = {b:+.3f}  ->  {muP}")

    print("\nsuggested main_grid.json LRs:")
    for d in args.grid_widths:
        lr = a * d**b
        extrap = "" if min(fit_d) <= d <= max(fit_d) else "  (extrapolated beyond pilot widths)"
        print(f'  d={d:<5} "lr": {lr:.4g}{extrap}')


if __name__ == "__main__":
    main()
