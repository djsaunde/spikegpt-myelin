"""Benchmark cached versus context-recompute SpikeGPT generation."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch
from myelin.baselines import (
    max_cuda_memory_allocated,
    reset_cuda_peak_memory,
    synchronize_if_needed,
)
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name

from spikegpt.benchmarks.spikegpt_training import resolve_config
from spikegpt.language import SPIKEGPT_PRESETS, SpikeLanguageModel


@dataclass(frozen=True)
class GenerationResult:
    path: str
    seconds: float
    peak_bytes: int | None
    tokens_per_second: float
    matches_reference: bool


def _time_generate(
    model: SpikeLanguageModel,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    use_cache: bool,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, float, int | None]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    with torch.inference_mode():
        for _ in range(warmup):
            model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                sampling="greedy",
                use_cache=use_cache,
            )
    synchronize_if_needed(prompt.device)
    reset_cuda_peak_memory(prompt.device)

    output = None
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            output = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                sampling="greedy",
                use_cache=use_cache,
            )
    synchronize_if_needed(prompt.device)
    seconds = (time.perf_counter() - start) / repeats
    if output is None:
        raise RuntimeError("generation benchmark did not produce output")
    return output, seconds, max_cuda_memory_allocated(prompt.device)


def run_benchmark(args: argparse.Namespace) -> list[GenerationResult]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")
    if args.new_tokens < 0:
        raise ValueError("--new-tokens must be non-negative")
    config = resolve_config(args)
    if args.prompt_tokens + args.new_tokens > config.context_length:
        raise ValueError(
            "--prompt-tokens + --new-tokens must be <= --context-length so cached and "
            "recompute generation have the same context contract"
        )

    model = SpikeLanguageModel(config).to(device=device)
    model.eval()
    prompt = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch, args.prompt_tokens),
        device=device,
    )

    reference, reference_seconds, reference_peak_bytes = _time_generate(
        model,
        prompt,
        max_new_tokens=args.new_tokens,
        use_cache=False,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    cached, cached_seconds, cached_peak_bytes = _time_generate(
        model,
        prompt,
        max_new_tokens=args.new_tokens,
        use_cache=True,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    generated_tokens = args.batch * args.new_tokens
    return [
        GenerationResult(
            path="recompute_context",
            seconds=reference_seconds,
            peak_bytes=reference_peak_bytes,
            tokens_per_second=generated_tokens / reference_seconds,
            matches_reference=True,
        ),
        GenerationResult(
            path="cached_recurrent_state",
            seconds=cached_seconds,
            peak_bytes=cached_peak_bytes,
            tokens_per_second=generated_tokens / cached_seconds,
            matches_reference=torch.equal(cached, reference),
        ),
    ]


def print_markdown(args: argparse.Namespace, rows: list[GenerationResult]) -> None:
    config = resolve_config(args)
    print("# SpikeGPT Generation Benchmark")
    print()
    print(f"Generated: {datetime.now(UTC).isoformat()}")
    print(f"Device: {args.device} ({gpu_name(args.device)})")
    print(
        "Shape: "
        f"batch={args.batch}, prompt_tokens={args.prompt_tokens}, "
        f"new_tokens={args.new_tokens}, preset={args.preset}, "
        f"context_length={config.context_length}, layers={config.n_layer}, "
        f"embedding={config.n_embd}, model_type={config.model_type}, "
        f"vocab_size={args.vocab_size}"
    )
    print(f"Warmup: {args.warmup}; repeats: {args.repeats}; seed: {args.seed}")
    print()
    print("| Path | Time | Tokens/s | Peak memory | Matches recompute |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.path} | {format_ms(row.seconds)} | "
            f"{row.tokens_per_second:.1f} | {format_memory(row.peak_bytes)} | "
            f"{row.matches_reference} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=("custom", *SPIKEGPT_PRESETS.keys()), default="custom")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--model-type", choices=("rwkv", "rwkv-ffn-pre"), default="rwkv")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--lif-threshold", type=float, default=0.0)
    parser.add_argument(
        "--dense-embedding",
        action="store_true",
        help="use dense token embeddings instead of hard surrogate binary embeddings",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
