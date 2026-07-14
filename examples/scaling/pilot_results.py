"""Pull finished LR-pilot runs from Weights & Biases into a flat table.

Phase-1 of the scaling study runs a few short jobs per width (see
``pilot_local.sh``) to locate the LR optimum at each width. This script reads
their final ``val/loss`` back out of W&B and emits ``(embedding, lr, batch,
val_loss)`` rows for ``fit_lr_rule.py`` to fit ``lr_opt(d) = a * d^b`` on.

Runs are matched by name prefix (default ``pilot_``) and must be ``finished``.
The entity defaults to ``pitheta`` (where every spikegpt-scaling project lives)
and can be overridden; the same W&B credentials that write the runs must be able
to read them (``wandb login`` / ``WANDB_API_KEY``).

    uv run --extra tracking python examples/scaling/pilot_results.py \
        --entity pitheta --project spikegpt-scaling --prefix pilot_ \
        --out runs/pilot_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Metric to fit on. val/loss is the raw cross-entropy (nats/token); it is
# unambiguous across byte/bpe vocab (unlike the bits-per-* label) and is what the
# d384 arm reported (1.1e-3->3.7894, 2.7e-3->3.7683, 6.7e-3->3.8720).
LOSS_KEY = "val/loss"


def pull(entity: str, project: str, prefix: str) -> list[dict]:
    import wandb  # imported lazily so --help works without the tracking extra

    api = wandb.Api()
    rows: list[dict] = []
    for run in api.runs(f"{entity}/{project}"):
        if not run.name or not run.name.startswith(prefix):
            continue
        cfg, summ = run.config, run.summary
        d, lr = cfg.get("embedding"), cfg.get("lr")
        loss = summ.get(LOSS_KEY)
        if d is None or lr is None or loss is None:
            print(
                f"  skip {run.name} (state={run.state}, d={d}, lr={lr}, {LOSS_KEY}={loss})",
                file=sys.stderr,
            )
            continue
        rows.append(
            {
                "name": run.name,
                "state": run.state,
                "embedding": int(d),
                "layers": cfg.get("layers"),
                "lr": float(lr),
                "batch": cfg.get("batch"),
                "steps": cfg.get("steps"),
                "val_loss": float(loss),
                "val_perplexity": summ.get("val/perplexity"),
            }
        )
    rows.sort(key=lambda r: (r["embedding"], r["lr"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--entity", default="pitheta")
    parser.add_argument("--project", default="spikegpt-scaling")
    parser.add_argument("--prefix", default="pilot_", help="only runs whose name starts with this")
    parser.add_argument("--out", type=Path, default=Path("runs/pilot_results.json"))
    args = parser.parse_args()

    rows = pull(args.entity, args.project, args.prefix)
    if not rows:
        raise SystemExit(
            f"no finished '{args.prefix}*' runs with {LOSS_KEY} in {args.entity}/{args.project} "
            "(wrong entity/credentials, or the runs are still training?)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"{'run':<28} {'d':>5} {'lr':>10} {'batch':>6} {'val_loss':>10}")
    for r in rows:
        print(f"{r['name']:<28} {r['embedding']:>5} {r['lr']:>10.4g} "
              f"{str(r['batch']):>6} {r['val_loss']:>10.4f}")
    widths = sorted({r["embedding"] for r in rows})
    print(f"\n{len(rows)} runs across widths {widths} -> {args.out}")
    print("next: uv run python examples/scaling/fit_lr_rule.py", args.out)


if __name__ == "__main__":
    main()
