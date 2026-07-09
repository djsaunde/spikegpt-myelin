"""Plan and launch a Chinchilla-style scaling-law sweep from a JSON grid.

Computes exact parameter counts / token budgets / FLOPs / cost estimates per
run (via ``spikegpt.scaling``) and prints the plan or emits/executes launch
commands for the ``modal/`` or ``lambda/`` harnesses.

Grid schema (see ``examples/scaling/main_grid.json``): ``{"defaults": {...},
"runs": [{...}, ...]}`` where each run requires ``name``, ``layers``,
``embedding``, ``tokens`` (the D of the run), ``lr``, and may override any
key in ``RUN_DEFAULTS`` below. ``corpus_tokens`` must be >= ``tokens`` for
single-epoch training; steps are ``ceil(tokens / (batch * context_length))``.

Commands::

    uv run python examples/scaling_sweep.py plan examples/scaling/main_grid.json
    uv run python examples/scaling_sweep.py emit examples/scaling/main_grid.json --backend modal
    uv run python examples/scaling_sweep.py launch examples/scaling/main_grid.json \
        --backend modal --filter c1e18 --yes

Modal is the intended grid backend (corpus cached on the ``myelin-corpora``
volume — prepare it once with ``prepare_corpus`` in ``modal/train_spikegpt.py``).
The lambda backend emits one-shot ``lambda/run.py run`` commands; each Lambda
VM is fresh, so the corpus .bin must be built on-VM first (see ONBOARDING.md).
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spikegpt.scaling import SpikeGPTParamCounts, count_spikegpt_params, training_flops

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_DEFAULTS: dict = {
    "context_length": 1024,
    "vocab_size": 50277,
    "batch": 64,
    "lr_final": None,  # -> lr / 10
    "warmup_steps": None,  # -> clamp(steps // 20, 100, 2000)
    "dropout": 0.0,
    "weight_decay": 0.1,
    "dataset": "fineweb-edu",
    "hf_config": "sample-100BT",
    "corpus_tokens": 25_000_000_000,
    "val_holdout_tokens": 50_000_000,
    "activation_checkpointing": None,  # -> embedding >= 1024
    "wandb_project": "spikegpt-scaling",
    "tier_flops": None,
    "log_every": 100,
    "eval_every": 1000,
    "eval_batches": 16,
}
REQUIRED_FIELDS = ("name", "layers", "embedding", "tokens", "lr")


@dataclass(frozen=True)
class PlannedRun:
    spec: dict
    counts: SpikeGPTParamCounts

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def steps(self) -> int:
        return math.ceil(self.spec["tokens"] / (self.spec["batch"] * self.spec["context_length"]))

    @property
    def lr_final(self) -> float:
        return self.spec["lr_final"] or self.spec["lr"] / 10.0

    @property
    def warmup_steps(self) -> int:
        if self.spec["warmup_steps"] is not None:
            return self.spec["warmup_steps"]
        return max(100, min(2000, self.steps // 20))

    @property
    def activation_checkpointing(self) -> bool:
        if self.spec["activation_checkpointing"] is not None:
            return self.spec["activation_checkpointing"]
        return self.spec["embedding"] >= 1024

    @property
    def flops(self) -> int:
        return training_flops(self.counts, self.spec["tokens"])

    def hours(self, eff_tflops: float) -> float:
        return self.flops / (eff_tflops * 1e12) / 3600.0


def load_grid(path: Path) -> list[PlannedRun]:
    grid = json.loads(path.read_text())
    defaults = {**RUN_DEFAULTS, **grid.get("defaults", {})}
    runs = []
    for entry in grid["runs"]:
        spec = {**defaults, **entry}
        missing = [field for field in REQUIRED_FIELDS if spec.get(field) is None]
        if missing:
            raise SystemExit(f"run {entry.get('name', entry)!r} missing fields: {missing}")
        counts = count_spikegpt_params(
            vocab_size=spec["vocab_size"], n_layer=spec["layers"], n_embd=spec["embedding"]
        )
        runs.append(PlannedRun(spec=spec, counts=counts))
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate run names in grid")
    return runs


def filter_runs(runs: list[PlannedRun], pattern: str | None) -> list[PlannedRun]:
    if not pattern:
        return runs
    kept = [run for run in runs if pattern in run.name]
    if not kept:
        raise SystemExit(f"--filter {pattern!r} matched no runs")
    return kept


def cmd_plan(runs: list[PlannedRun], args: argparse.Namespace) -> None:
    print(
        "| run | d/L | N total | N non-vocab | D tokens | D/N | steps | C FLOPs | est h | est $ |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    total_flops = 0
    total_hours = 0.0
    for run in runs:
        spec, counts = run.spec, run.counts
        hours = run.hours(args.eff_tflops)
        total_flops += run.flops
        total_hours += hours
        single_epoch = spec["tokens"] <= spec["corpus_tokens"] - spec["val_holdout_tokens"]
        over = "" if single_epoch else " ⚠️>corpus"
        print(
            f"| {run.name} | {spec['embedding']}/{spec['layers']} "
            f"| {counts.total / 1e6:.1f}M | {counts.non_vocab / 1e6:.1f}M "
            f"| {spec['tokens'] / 1e9:.2f}B{over} | {spec['tokens'] / counts.non_vocab:.0f} "
            f"| {run.steps:,} | {run.flops:.2e} | {hours:.1f} | {hours * args.usd_per_hour:.0f} |"
        )
    print(
        f"\ntotal: {len(runs)} runs, {total_flops:.2e} FLOPs, ~{total_hours:.0f} GPU-h, "
        f"~${total_hours * args.usd_per_hour:.0f} at {args.eff_tflops:.0f} effective TFLOP/s "
        f"and ${args.usd_per_hour:.2f}/h (override with --eff-tflops / --usd-per-hour)"
    )


def modal_command(run: PlannedRun) -> list[str]:
    spec = run.spec
    command = [
        "modal",
        "run",
        "--detach",
        str(REPO_ROOT / "modal" / "train_spikegpt.py"),
        "--dataset",
        spec["dataset"],
        "--hf-config",
        spec["hf_config"],
        "--max-tokens",
        str(spec["corpus_tokens"]),
        "--val-holdout-tokens",
        str(spec["val_holdout_tokens"]),
        "--preset",
        "custom",
        "--context-length",
        str(spec["context_length"]),
        "--layers",
        str(spec["layers"]),
        "--embedding",
        str(spec["embedding"]),
        "--batch",
        str(spec["batch"]),
        "--steps",
        str(run.steps),
        "--lr",
        f"{spec['lr']:g}",
        "--lr-final",
        f"{run.lr_final:g}",
        "--warmup-steps",
        str(run.warmup_steps),
        "--dropout",
        f"{spec['dropout']:g}",
        "--weight-decay",
        f"{spec['weight_decay']:g}",
        "--wandb",
        "--wandb-project",
        spec["wandb_project"],
        "--wandb-run-name",
        run.name,
    ]
    if run.activation_checkpointing:
        command.append("--activation-checkpointing")
    return command


def lambda_command(run: PlannedRun) -> list[str]:
    spec = run.spec
    corpus = f"data/{spec['dataset']}_{spec['hf_config']}_{spec['corpus_tokens']}.bin"
    trainer = [
        "uv",
        "run",
        "--extra",
        "cuda",
        "--extra",
        "tracking",
        "python",
        "examples/train_tiny_spikegpt.py",
        "--train-bin",
        corpus,
        "--vocab",
        "bpe",
        "--val-holdout-tokens",
        str(spec["val_holdout_tokens"]),
        "--context-length",
        str(spec["context_length"]),
        "--layers",
        str(spec["layers"]),
        "--embedding",
        str(spec["embedding"]),
        "--batch",
        str(spec["batch"]),
        "--steps",
        str(run.steps),
        "--lr",
        f"{spec['lr']:g}",
        "--lr-final",
        f"{run.lr_final:g}",
        "--warmup-steps",
        str(run.warmup_steps),
        "--dropout",
        f"{spec['dropout']:g}",
        "--weight-decay",
        f"{spec['weight_decay']:g}",
        "--amp",
        "bf16",
        "--compile",
        "regional",
        "--wandb",
        "--wandb-project",
        spec["wandb_project"],
        "--wandb-run-name",
        run.name,
        "--best-checkpoint-out",
        f"runs/{run.name}.best.pt",
    ]
    if run.activation_checkpointing:
        trainer.append("--activation-checkpointing")
    # Fresh Lambda VMs have no corpus: build data/<tag>.bin on-VM first (ONBOARDING.md).
    return ["python3", "lambda/run.py", "run", "--name", run.name, "--", *trainer]


def cmd_emit(runs: list[PlannedRun], args: argparse.Namespace) -> None:
    build = modal_command if args.backend == "modal" else lambda_command
    for run in runs:
        print(shlex.join(build(run)))


def cmd_launch(runs: list[PlannedRun], args: argparse.Namespace) -> None:
    if args.backend != "modal":
        raise SystemExit(
            "launch supports --backend modal only (lambda runs are blocking one-shots; "
            "use `emit --backend lambda` and run them yourself)"
        )
    if not args.yes:
        raise SystemExit(f"would launch {len(runs)} paid Modal runs; re-run with --yes")
    for run in runs:
        command = modal_command(run)
        print(f"launching {run.name}: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("plan", "emit", "launch"))
    parser.add_argument("grid", type=Path, help="grid JSON file")
    parser.add_argument("--backend", choices=("modal", "lambda"), default="modal")
    parser.add_argument("--filter", help="only runs whose name contains this substring")
    parser.add_argument(
        "--eff-tflops",
        type=float,
        default=100.0,
        help="assumed effective training TFLOP/s for time/cost estimates "
        "(calibrate with python -m spikegpt.benchmarks.spikegpt_mfu)",
    )
    parser.add_argument("--usd-per-hour", type=float, default=3.95, help="GPU $/h (Modal H100)")
    parser.add_argument("--yes", action="store_true", help="confirm paid launches")
    args = parser.parse_args()

    runs = filter_runs(load_grid(args.grid), args.filter)
    if args.command == "plan":
        cmd_plan(runs, args)
    elif args.command == "emit":
        cmd_emit(runs, args)
    else:
        cmd_launch(runs, args)


if __name__ == "__main__":
    main()
