"""IsoFLOP analysis (Hoffmann approach 2) of the scaling grid — the robust primary.

Pulls finished grid runs from W&B, groups them by FLOP tier, and for each tier
with >=3 widths fits a parabola of val_loss vs log(N) to locate the
compute-optimal model size N_opt(C) and the achievable loss L_min(C). Across
tiers (>=2) it then fits the Chinchilla frontier N_opt(C) ~ C^a, D_opt(C) ~ C^b.

More robust than the full L(N,D) fit (fit_scaling_law.py) because each tier's
minimum is found independently, sidestepping the E-degeneracy. Works on partial
data: reports whatever tiers are complete enough.

    uv run --extra tracking python examples/scaling/fit_isoflop.py

N = TOTAL params including embedding (the Chinchilla convention). Kaplan's
non-embedding counting is the documented cause of his inflated N_opt ~ C^0.73:
at small scale with a large vocab it biases the exponent upward (Pearce & Song
2024, arXiv:2406.12907). Our regime is exactly that — a 50,277 vocab makes the
embedding tables 36-67% of these models — so total params is the defensible and
comparable choice; the non-embedding exponent is printed only as a diagnostic.

This does not disturb the FLOP accounting: the isoFLOP method locates the
loss-minimising N at each *measured* compute, so the N reported here never has to
match the N inside C = 6*N_flop*D.

Dedup: for each run name keeps the finished run with the most logged steps; a
crashed/partial run is dropped and a leftover duplicate is warned about.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from spikegpt.scaling import count_spikegpt_params, training_flops


def _arm(config: dict) -> str:
    """The {spiking, mixer} ablation arm a run belongs to, e.g. 'spiking+rwkv'."""
    spiking = config.get("spiking")
    if spiking is None:  # pre-2026-07 runs: infer from the spike-embedding proxy
        spiking = config.get("spike_embedding", True)
    attention = config.get("attention") or "rwkv"
    return f"{'spiking' if spiking else 'continuous'}+{attention}"


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
            # N = TOTAL params incl. embedding — the Chinchilla convention. Kaplan's
            # non-embedding counting is the documented cause of his inflated C^0.73
            # (Pearce & Song 2024, arXiv:2406.12907): at small scale with a large
            # vocab it biases the exponent upward. Our regime is exactly that, so we
            # report `N` as total and keep non_vocab only as a diagnostic.
            "name": name, "embedding": d, "layers": L,
            "N": counts.total, "N_nonvocab": counts.non_vocab,
            "D": float(D), "C": training_flops(counts, int(D)), "val_loss": float(vl),
            "steps": s.get("_step", 0), "id": r.id,
            # SpikeGPT-specific: final spike rates (fraction of neurons firing).
            "emb_spike": s.get("train/embedding_spike_rate"),
            "block_spike": s.get("train/mean_block_spike_rate"),
            # Ablation arm. `spiking` is logged since 2026-07; older runs lack it, so
            # fall back to spike_embedding=False as the non-spiking proxy. Runs of
            # different arms are NEVER fit together -- a continuous twin at the same
            # (N, C) is a different loss surface, not another point on the same one.
            "arm": _arm(c),
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
    """Log-parabola vertex of y vs x; (x_opt, interior?).

    "interior" requires BOTH the fitted vertex to lie inside the sampled range AND
    the sampled minimum to not be at an endpoint. The second guard matters: a
    monotonically-decreasing tier (loss still falling at the widest width) has its
    true optimum at/beyond the edge, but a quadratic can curve up just past the last
    point and place a spurious "interior" vertex on it -- pinning N_opt to the
    largest sampled width and biasing the frontier. Such a tier needs a wider run,
    not a fit.
    """
    a2, a1, _ = np.polyfit(x, y, 2)
    if a2 <= 0:
        return float(x[np.argmin(y)]), False
    xo = -a1 / (2 * a2)
    sampled_min_interior = 0 < int(np.argmin(y)) < len(y) - 1
    return float(xo), bool(x.min() < xo < x.max()) and sampled_min_interior


def fit_arm(rows: list[dict], key: str = "N") -> list[tuple[float, float, float, float]]:
    """Per-tier parabola -> interior compute-optimal frontier for ONE arm's runs.

    Returns [(C, N_opt, D_opt, L_min), ...] over tiers with an interior optimum.
    Prints the tiers. ``key`` selects the N convention ("N" total / "N_nonvocab").
    """
    tiers: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tiers[f"{r['C']:.0e}"].append(r)
    frontier = []
    for cflops in sorted(tiers, key=float):
        ts = sorted(tiers[cflops], key=lambda r: r[key])
        if key == "N":  # only narrate for the headline convention
            print(f"  C = {cflops} FLOPs — widths {[t['embedding'] for t in ts]}")
            for t in ts:
                print(f"    d{t['embedding']:<5} N={t['N'] / 1e6:6.1f}M  "
                      f"D={t['D'] / 1e9:6.2f}B  loss={t['val_loss']:.4f}")
        if len(ts) < 3:
            continue
        N = np.array([t[key] for t in ts], float)
        Lv = np.array([t["val_loss"] for t in ts])
        logNopt, interior = _vertex(np.log(N), Lv)
        Nopt, C = np.exp(logNopt), float(cflops)
        if key == "N":
            tag = "interior" if interior else "AT EDGE (optimum outside widths — extend)"
            print(f"    => N_opt ~ {Nopt / 1e6:.0f}M, D_opt ~ {C / (6 * Nopt) / 1e9:.2f}B, "
                  f"L_min <= {Lv.min():.4f}  [{tag}]")
        if interior:
            frontier.append((C, Nopt, C / (6 * Nopt), float(Lv.min())))
    return frontier


def _exponent(frontier: list[tuple], idx: int) -> float:
    C = np.array([f[0] for f in frontier])
    return float(np.polyfit(np.log(C), np.log([f[idx] for f in frontier]), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--entity", default="pitheta")
    ap.add_argument("--project", default="spikegpt-scaling")
    args = ap.parse_args()

    rows = pull(args.entity, args.project)
    arms: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)
    print(f"\n{len(rows)} finished grid runs across {len(arms)} arm(s): "
          f"{', '.join(f'{a} ({len(rs)})' for a, rs in sorted(arms.items()))}")

    # Baseline (SpikeGPT) first, then any ablation arms. Arms are never mixed:
    # a continuous/vanilla run at the same (N, C) is a different loss surface.
    for arm in sorted(arms, key=lambda a: (a != "spiking+rwkv", a)):
        print(f"\n===== arm: {arm} =====")
        frontier = fit_arm(arms[arm])
        if len(frontier) < 2:
            print(f"  ({len(frontier)} interior tier(s) — need >=2 for the N_opt(C) frontier)")
            continue
        aN, aD = _exponent(frontier, 1), _exponent(frontier, 2)
        print(f"  COMPUTE-OPTIMAL FRONTIER ({len(frontier)} tiers): N_opt ~ C^{aN:.3f}, "
              f"D_opt ~ C^{aD:.3f}  [total params, Chinchilla ~0.5/0.5]")
        kaplan = fit_arm(arms[arm], key="N_nonvocab")
        if len(kaplan) >= 2:
            print(f"  (diagnostic — Kaplan/non-embedding: C^{_exponent(kaplan, 1):.3f}, "
                  "biased high at this scale)")


if __name__ == "__main__":
    main()
