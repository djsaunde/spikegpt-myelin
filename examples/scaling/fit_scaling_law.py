"""Fit the Chinchilla loss surface L(N, D) = E + A/N^alpha + B/D^beta.

Phase-3 of the scaling study. Takes the finished grid runs as (N, D, loss) points
-- N = non-vocab params (spikegpt.scaling.non_vocab), D = training tokens, loss =
final val/loss -- and fits the parametric form with the Hoffmann et al. protocol:

* fit in log space (E, A, B held positive via log-parameters),
* Huber loss on log-residuals log(L_pred) - log(L_actual) (robust to outliers),
* multi-start L-BFGS over a grid of initializations, keep the best objective,
* bootstrap over runs for CIs on the exponents.

From the fit it derives the compute-optimal frontier under C = 6*N*D:
N_opt(C) ∝ C^a, D_opt(C) ∝ C^b with a = beta/(alpha+beta), b = alpha/(alpha+beta),
and inverts N_opt(C) to answer "at what compute is width/size X optimal".

    uv run --with scipy python examples/scaling/fit_scaling_law.py runs/grid_results.json
    uv run --with scipy python examples/scaling/fit_scaling_law.py --self-test   # synthetic identifiability check

Input JSON: a list of {"non_vocab": N, "tokens": D, "val_loss": L, "name": ...}.
(Produce it from W&B with pilot_results.py's approach, or --self-test to validate
the fitter + the grid design before the runs finish.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    raise SystemExit("needs scipy: run with `uv run --with scipy python ...`")

HUBER_DELTA = 1e-3
# Multi-start grid (Hoffmann Table A2 style): a handful of physically-reasonable
# inits is enough -- E~1-2 (logE~0-0.7), A,B large (logAB~5-8), exponents ~0.25-0.4.
INIT_LOGE = (0.0, 0.7)
INIT_LOGAB = (5.0, 8.0)
INIT_EXP = (0.25, 0.4)


def _predict_log(params: np.ndarray, logN: np.ndarray, logD: np.ndarray) -> np.ndarray:
    """log L_pred via logsumexp of [logE, logA - a*logN, logB - b*logD]."""
    logE, logA, logB, a, b = params
    terms = np.stack([
        np.full_like(logN, logE),
        logA - a * logN,
        logB - b * logD,
    ])
    m = terms.max(axis=0)
    return m + np.log(np.exp(terms - m).sum(axis=0))


def _huber(r: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def fit(N: np.ndarray, D: np.ndarray, L: np.ndarray) -> dict:
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def obj(p):
        return _huber(_predict_log(p, logN, logD) - logL).sum()

    best = None
    for le in INIT_LOGE:
        for lab in INIT_LOGAB:
            for a in INIT_EXP:
                for b in INIT_EXP:
                    p0 = np.array([le, lab, lab, a, b])
                    res = minimize(obj, p0, method="L-BFGS-B",
                                   bounds=[(-5, 5), (-5, 20), (-5, 20), (0.05, 1.5), (0.05, 1.5)])
                    if best is None or res.fun < best.fun:
                        best = res
    logE, logA, logB, alpha, beta = best.x
    a_exp = beta / (alpha + beta)   # N_opt ∝ C^a_exp
    b_exp = alpha / (alpha + beta)  # D_opt ∝ C^b_exp
    return {
        "E": float(np.exp(logE)), "A": float(np.exp(logA)), "B": float(np.exp(logB)),
        "alpha": float(alpha), "beta": float(beta),
        "a_exp": float(a_exp), "b_exp": float(b_exp), "obj": float(best.fun),
    }


def bootstrap(N, D, L, n=300, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    keys = ["alpha", "beta", "a_exp", "b_exp"]
    samples = {k: [] for k in keys}
    idx = np.arange(len(N))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            f = fit(N[s], D[s], L[s])
        except Exception:
            continue
        for k in keys:
            samples[k].append(f[k])
    return {k: (float(np.percentile(v, 5)), float(np.percentile(v, 95))) for k, v in samples.items()}


def _grid_points() -> tuple[np.ndarray, np.ndarray]:
    """(N_nonvocab, D) at the actual main_grid.json runs -- for the self-test."""
    from spikegpt.scaling import count_spikegpt_params
    grid = json.loads((Path(__file__).parent / "main_grid.json").read_text())
    N, D = [], []
    for r in grid["runs"]:
        c = count_spikegpt_params(vocab_size=50277, n_layer=r["layers"], n_embd=r["embedding"])
        N.append(c.non_vocab)
        D.append(r["tokens"])
    return np.array(N, float), np.array(D, float)


def self_test() -> None:
    """Can the grid's (N,D) coverage recover known exponents under realistic noise?"""
    N, D = _grid_points()
    # Calibrated to our real loss scale (~3.2-3.9 nats): high irreducible floor E,
    # so the reducible A/N^a + B/D^b signal is only ~0.7-1.0 nat -- the honest,
    # harder identifiability regime (vs a toy with a 3-nat reducible range).
    true = dict(E=2.8, A=150.0, B=120.0, alpha=0.34, beta=0.28)
    rng = np.random.default_rng(0)
    Lclean = true["E"] + true["A"] / N**true["alpha"] + true["B"] / D**true["beta"]
    print(f"synthetic loss range: {Lclean.min():.3f}..{Lclean.max():.3f} nats "
          f"(grid has {len(N)} runs)")
    for noise in (0.0, 0.01, 0.02):
        L = Lclean * np.exp(rng.normal(0, noise, len(N)))
        f = fit(N, D, L)
        print(f"\n-- {noise*100:.0f}% log-noise --")
        print(f"  alpha {f['alpha']:.3f} (true {true['alpha']}),  "
              f"beta {f['beta']:.3f} (true {true['beta']})")
        print(f"  N_opt∝C^{f['a_exp']:.3f}, D_opt∝C^{f['b_exp']:.3f} "
              f"(true {true['beta']/(true['alpha']+true['beta']):.3f}/"
              f"{true['alpha']/(true['alpha']+true['beta']):.3f})")
        if noise > 0:
            ci = bootstrap(N, D, L, n=60)
            print(f"  90% CI: alpha [{ci['alpha'][0]:.3f},{ci['alpha'][1]:.3f}]  "
                  f"beta [{ci['beta'][0]:.3f},{ci['beta'][1]:.3f}]  "
                  f"a_exp [{ci['a_exp'][0]:.3f},{ci['a_exp'][1]:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("results", type=Path, nargs="?", help="grid results JSON")
    parser.add_argument("--self-test", action="store_true", help="synthetic identifiability check")
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.results:
        raise SystemExit("give a results JSON, or --self-test")

    rows = json.loads(args.results.read_text())
    N = np.array([r["non_vocab"] for r in rows], float)
    D = np.array([r["tokens"] for r in rows], float)
    L = np.array([r["val_loss"] for r in rows], float)
    f = fit(N, D, L)
    ci = bootstrap(N, D, L, n=args.bootstrap)
    print(f"L(N,D) = {f['E']:.3f} + {f['A']:.1f}/N^{f['alpha']:.3f} + {f['B']:.1f}/D^{f['beta']:.3f}")
    print(f"  alpha = {f['alpha']:.3f}  90% CI [{ci['alpha'][0]:.3f}, {ci['alpha'][1]:.3f}]")
    print(f"  beta  = {f['beta']:.3f}  90% CI [{ci['beta'][0]:.3f}, {ci['beta'][1]:.3f}]")
    print(f"compute-optimal: N_opt ∝ C^{f['a_exp']:.3f}, D_opt ∝ C^{f['b_exp']:.3f}")
    print(f"  90% CI a_exp [{ci['a_exp'][0]:.3f}, {ci['a_exp'][1]:.3f}]")


if __name__ == "__main__":
    main()
