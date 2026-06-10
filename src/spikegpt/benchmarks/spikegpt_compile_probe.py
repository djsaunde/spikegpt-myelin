"""Probe torch.compile costs for SpikeGPT-style training."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import torch
from myelin.baselines import (
    max_cuda_memory_allocated,
    reset_cuda_peak_memory,
    synchronize_if_needed,
)
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name
from torch import nn

from spikegpt.benchmarks.spikegpt_training import (
    _make_batch,
    _make_model,
    _train_step,
    resolve_config,
)
from spikegpt.language import SPIKEGPT_PRESETS


@dataclass(frozen=True)
class CompileProbeRow:
    phase: str
    seconds: float | None
    peak_bytes: int | None
    loss: float | None
    error: str | None = None


def _time_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[float, int | None, float]:
    reset_cuda_peak_memory(device)
    start = time.perf_counter()
    loss = _train_step(model, optimizer, inputs, targets)
    synchronize_if_needed(device)
    return time.perf_counter() - start, max_cuda_memory_allocated(device), float(loss)


def _make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def run_probe(args: argparse.Namespace) -> list[CompileProbeRow]:
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if getattr(args, "preset", "custom") == "custom":
        if args.context_length <= 0:
            raise ValueError("--context-length must be positive")
        if args.layers <= 0:
            raise ValueError("--layers must be positive")
        if args.embedding <= 0:
            raise ValueError("--embedding must be positive")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)
    inputs, targets = _make_batch(args, device)
    rows: list[CompileProbeRow] = []

    try:
        torch.manual_seed(args.seed)
        eager_model = _make_model(args, device)
        eager_optimizer = _make_optimizer(args, eager_model)
        seconds, peak_bytes, loss = _time_step(
            eager_model, eager_optimizer, inputs, targets, device
        )
        rows.append(CompileProbeRow("eager_first_step", seconds, peak_bytes, loss))

        reset_cuda_peak_memory(device)
        start = time.perf_counter()
        last_loss = None
        for _ in range(args.repeats):
            last_loss = _train_step(eager_model, eager_optimizer, inputs, targets)
        synchronize_if_needed(device)
        rows.append(
            CompileProbeRow(
                "eager_steady_step",
                (time.perf_counter() - start) / args.repeats,
                max_cuda_memory_allocated(device),
                None if last_loss is None else float(last_loss),
            )
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        synchronize_if_needed(device)
        rows.append(CompileProbeRow("eager_path", None, None, None, f"{type(exc).__name__}: {exc}"))

    try:
        torch.manual_seed(args.seed)
        raw_model = _make_model(args, device)
        start = time.perf_counter()
        compiled_model = cast(
            nn.Module,
            torch.compile(raw_model, mode=args.compile_mode, fullgraph=args.fullgraph),
        )
        wrapper_seconds = time.perf_counter() - start
        rows.append(CompileProbeRow("compile_wrapper", wrapper_seconds, None, None))

        compiled_optimizer = _make_optimizer(args, compiled_model)
        first_seconds, first_peak, first_loss = _time_step(
            compiled_model, compiled_optimizer, inputs, targets, device
        )
        rows.append(CompileProbeRow("compiled_first_step", first_seconds, first_peak, first_loss))

        reset_cuda_peak_memory(device)
        start = time.perf_counter()
        last_loss = None
        for _ in range(args.repeats):
            last_loss = _train_step(compiled_model, compiled_optimizer, inputs, targets)
        synchronize_if_needed(device)
        steady_seconds = (time.perf_counter() - start) / args.repeats
        rows.append(
            CompileProbeRow(
                "compiled_steady_step",
                steady_seconds,
                max_cuda_memory_allocated(device),
                None if last_loss is None else float(last_loss),
            )
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report compiler failures.
        synchronize_if_needed(device)
        rows.append(
            CompileProbeRow("compiled_path", None, None, None, f"{type(exc).__name__}: {exc}")
        )
    return rows


def print_markdown(args: argparse.Namespace, rows: list[CompileProbeRow]) -> None:
    config = resolve_config(args)
    print("# SpikeGPT Compile Probe")
    print()
    print(f"Generated: {datetime.now(UTC).isoformat()}")
    print(f"Device: {args.device} ({gpu_name(args.device)})")
    print(
        "Shape: "
        f"batch={args.batch}, preset={args.preset}, context_length={config.context_length}, "
        f"layers={config.n_layer}, embedding={config.n_embd}, model_type={config.model_type}, "
        f"vocab_size={args.vocab_size}"
    )
    print(
        f"compile_mode={args.compile_mode}; fullgraph={args.fullgraph}; "
        f"matmul_precision={args.matmul_precision}; repeats={args.repeats}; seed={args.seed}"
    )
    print()
    print("| Phase | Time | Peak memory | Loss | Error |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        loss = "" if row.loss is None else f"{row.loss:.6f}"
        print(
            f"| {row.phase} | {format_ms(row.seconds)} | {format_memory(row.peak_bytes)} | "
            f"{loss} | {row.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=("custom", *SPIKEGPT_PRESETS.keys()), default="custom")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--embedding", type=int, default=16)
    parser.add_argument("--model-type", choices=("rwkv", "rwkv-ffn-pre"), default="rwkv")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lif-threshold", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dense-embedding", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print_markdown(args, run_probe(args))


if __name__ == "__main__":
    main()
