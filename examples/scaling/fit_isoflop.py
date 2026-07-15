"""IsoFLOP analysis (Hoffmann approach 2) of the scaling grid — the robust primary.

Pulls finished grid runs from W&B, groups them by FLOP tier, and for each tier
with >=3 widths fits a parabola of val_loss vs log(N) to locate the
compute-optimal model size N_opt(C) and the achievable loss L_min(C). Across
tiers (>=2) it then fits the Chinchilla frontier N_opt(C) ~ C^a, D_opt(C) ~ C^b.

More robust than the full L(N,D) fit (fit_scaling_law.py) because each tier's
minimum is found independently, sidestepping the E-degeneracy. Works on partial
data: reports whatever tiers are complete enough.

    uv run --extra tracking python examples/scaling/fit_isoflop.py

N = non-vocab params (spikegpt.scaling.non_vocab). Dedup: for each run name keeps
the finished run with the most logged steps; a crashed/partial run is dropped and
a leftover duplicate is warned about.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from spikegpt.scaling import count_spikegpt_params, training_flops


def pull(entity: str, project: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    best: dict[str, dict] = {}
    dups: set[str] = set()
    for r in api.runs(f"{entity}/{project}"):
        name = r.name or ""
        if not name or name.startswith("pilot_"):
            continue
        if r.state != "finished":
            continue
        c, s = r.config, r.summary
        d, L, tok, vl = c.get("embedding"), c.get("layers"), c.get("steps"), s.get("val/loss")
        if d is None or L is None or vl is None:
            continue
        counts = count_spikegpt_params(vocab_size=50277, n_layer=L, n_embd=d)
        # D = tokens the run actually trained on = total_steps * batch * ctx. Use
        # total_steps (not steps) so a resumed run counts its full horizon, not just
        # the resumed tail. NOT config train_tokens — that's the whole corpus size.
        steps_total = c.get("total_steps") or tok
        D = steps_total * c.get("batch", 16) * c.get("context_length", 1024)
        row = {
            "name": name, "embedding": d, "layers": L, "N": counts.non_vocab,
            "D": float(D), "C": training_flops(counts, int(D)), "val_loss": float(vl),
            "steps": s.get("_step", 0), "id": r.id,
        }
        if name in best:
            dups.add(name)
            if row["steps"] <= best[name]["steps"]:
                continue
        best[name] = row
    for n in sorted(dups):
        print(f"  WARN: duplicate finished runs named {n}; kept id={best[n]['id']} "
              f"(most steps). Investigate/delete the others.")
    return list(best.values())


def _vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Log-parabola vertex of y vs x; (x_opt, interior?)."""
    a2, a1, _ = np.polyfit(x, y, 2)
    if a2 <= 0:
        return float(x[np.argmin(y)]), False
    xo = -a1 / (2 * a2)
    return float(xo), bool(x.min() < xo < x.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--entity", default="pitheta")
    ap.add_argument("--project", default="spikegpt-scaling")
    args = ap.parse_args()

    rows = pull(args.entity, args.project)
    tiers: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tiers[f"{r['C']:.0e}"].append(r)

    print(f"\n{len(rows)} finished grid runs across {len(tiers)} FLOP tiers\n")
    frontier = []  # (C, N_opt, D_opt, L_min)
    for cflops in sorted(tiers, key=float):
        ts = sorted(tiers[cflops], key=lambda r: r["N"])
        widths = [t["embedding"] for t in ts]
        print(f"── C = {cflops} FLOPs — widths {widths} ──")
        for t in ts:
            print(f"    d{t['embedding']:<5} N={t['N']/1e6:6.1f}M  D={t['D']/1e9:6.2f}B  loss={t['val_loss']:.4f}")
        if len(ts) < 3:
            print("    (need >=3 widths for a parabola — incomplete tier)\n")
            continue
        N = np.array([t["N"] for t in ts], float)
        Lv = np.array([t["val_loss"] for t in ts])
        logNopt, interior = _vertex(np.log(N), Lv)
        Nopt = np.exp(logNopt)
        C = float(cflops)
        Dopt = C / (6 * Nopt)  # from C = 6 N D (approx; N_opt here is non-vocab)
        Lmin = Lv.min()
        tag = "interior" if interior else "AT EDGE (optimum outside sampled widths — extend)"
        print(f"    => N_opt ~ {Nopt/1e6:.0f}M, D_opt ~ {Dopt/1e9:.2f}B, L_min <= {Lmin:.4f}  [{tag}]\n")
        if interior:
            frontier.append((C, Nopt, Dopt, Lmin))

    if len(frontier) < 2:
        print(f"Only {len(frontier)} tier(s) with an interior optimum — need >=2 to fit "
              "the compute-optimal frontier N_opt(C). Run more tiers.")
        return
    C = np.array([f[0] for f in frontier])
    No = np.array([f[1] for f in frontier])
    Do = np.array([f[2] for f in frontier])
    a, _ = np.polyfit(np.log(C), np.log(No), 1)
    b, _ = np.polyfit(np.log(C), np.log(Do), 1)
    print(f"COMPUTE-OPTIMAL FRONTIER ({len(frontier)} tiers):")
    print(f"  N_opt(C) ~ C^{a:.3f}   D_opt(C) ~ C^{b:.3f}   (Chinchilla: ~0.5 / ~0.5)")


if __name__ == "__main__":
    main()
