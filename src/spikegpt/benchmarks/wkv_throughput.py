"""Throughput comparison: production Triton WKV vs chunked/parallel matmul forms.

The production WKV path (``spikegpt.wkv_triton.weighted_key_value_triton``) is a
sequential-over-time Triton kernel: one program per ``(batch, channel-block)``
runs a serial ``T``-step loop. At our scale (B=12-64, C=512, T=1024-3072) that is
only ``B * ceil(C/64)`` programs, each latency-bound on the recurrence -- low
occupancy on a 170-SM Blackwell card.

The chunked/parallel decay-matrix forms (``spikegpt_wkv_compare._wkv_span``)
trade that for ``T / chunk`` sequential steps, each a tensor-core batched matmul
over the chunk -- far more parallelism per step. This benchmark measures whether
that wins at production shapes, in forward-only and forward+backward, with peak
memory, all computed in fp32 internally (matching how the model runs WKV under
bf16 autocast: bf16 in, fp32 recurrence, bf16 out).

Correctness at long ``T`` is checked against the Triton kernel (itself validated
against the sequential loop oracle in ``tests/test_wkv_variants.py``); the loop
oracle is too slow to run at T=3072 here.
"""

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

from spikegpt.benchmarks.spikegpt_wkv_compare import wkv_chunked, wkv_parallel

# --- Variants ----------------------------------------------------------------
#
# Each variant is a callable (key, value, time_decay, time_first) -> out that
# computes the recurrence in fp32 internally and returns the caller's dtype,
# matching the production kernels' bf16-in / fp32-compute / bf16-out contract.


def _fp32_io(fn):
    """Wrap a pure-torch WKV so it upcasts to fp32 and casts the result back."""

    def wrapped(key, value, time_decay, time_first):
        out_dtype = key.dtype
        out = fn(
            key.float(),
            value.float(),
            time_decay.float(),
            time_first.float(),
        )
        return out.to(out_dtype)

    return wrapped


def _build_variants(chunk_sizes: list[int], *, with_triton: bool) -> dict:
    variants: dict = {
        "parallel": _fp32_io(wkv_parallel),
    }
    for size in chunk_sizes:
        variants[f"chunked{size}"] = _fp32_io(
            lambda k, v, d, f, _s=size: wkv_chunked(k, v, d, f, chunk_size=_s)
        )
    if with_triton:
        from spikegpt.wkv_triton import weighted_key_value_triton

        # The Triton custom op already does fp32-internal compute + dtype cast-back.
        variants["triton"] = weighted_key_value_triton
    return variants


# --- Measurement -------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    variant: str
    batch: int
    timesteps: int
    max_err: float | None
    fwd_ms: float | None
    fwd_bwd_ms: float | None
    peak_bytes: int | None
    error: str | None = None


def _make_inputs(batch, t, channels, device, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    key = (torch.randn(batch, t, channels, generator=g) * 0.5).to(device=device, dtype=dtype)
    value = torch.randn(batch, t, channels, generator=g).to(device=device, dtype=dtype)
    time_decay = (-5.0 + 8.0 * torch.rand(channels, generator=g)).to(device=device, dtype=dtype)
    time_first = (torch.ones(channels) * -1.2 + torch.randn(channels, generator=g) * 0.5).to(
        device=device, dtype=dtype
    )
    return key, value, time_decay, time_first


def _time(fn, inputs, *, backward: bool, repeats: int, device) -> float:
    key, value, time_decay, time_first = inputs
    for _ in range(2):  # warmup
        _run_once(fn, key, value, time_decay, time_first, backward=backward)
    synchronize_if_needed(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _run_once(fn, key, value, time_decay, time_first, backward=backward)
    synchronize_if_needed(device)
    return (time.perf_counter() - start) / repeats * 1000.0


def _run_once(fn, key, value, time_decay, time_first, *, backward: bool) -> None:
    if not backward:
        with torch.no_grad():
            fn(key, value, time_decay, time_first)
        return
    tensors = [
        t.detach().clone().requires_grad_(True) for t in (key, value, time_decay, time_first)
    ]
    out = fn(*tensors)
    out.float().square().sum().backward()


def _peak_memory(fn, inputs, device) -> int | None:
    if torch.device(device).type != "cuda":
        return None
    reset_cuda_peak_memory(device)
    _run_once(fn, *inputs, backward=True)
    synchronize_if_needed(device)
    return max_cuda_memory_allocated(device)


def _one_line(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".splitlines()[0]
    return text if len(text) <= 90 else text[:87] + "..."


def run(args) -> list[Row]:
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.set_float32_matmul_precision(args.matmul_precision)
    with_triton = device.type == "cuda" and _has_triton()
    variants = _build_variants(args.chunk_sizes, with_triton=with_triton)
    ref_name = "triton" if with_triton else "parallel"

    rows: list[Row] = []
    for batch in args.batches:
        for t in args.timesteps:
            inputs = _make_inputs(batch, t, args.channels, device, dtype, args.seed)
            ref_out = None
            try:
                with torch.no_grad():
                    ref_out = variants[ref_name](*inputs).float()
            except Exception:  # noqa: BLE001
                ref_out = None
            for name, fn in variants.items():
                max_err = fwd_ms = fwd_bwd_ms = None
                peak = None
                errors: list[str] = []
                try:
                    if ref_out is not None:
                        with torch.no_grad():
                            out = fn(*inputs).float()
                        max_err = (out - ref_out).abs().max().item()
                    fwd_ms = _time(fn, inputs, backward=False, repeats=args.repeats, device=device)
                    fwd_bwd_ms = _time(
                        fn, inputs, backward=True, repeats=args.repeats, device=device
                    )
                    peak = _peak_memory(fn, inputs, device)
                except Exception as exc:  # noqa: BLE001 - benchmark reports failures.
                    synchronize_if_needed(device)
                    errors.append(_one_line(exc))
                rows.append(
                    Row(
                        name, batch, t, max_err, fwd_ms, fwd_bwd_ms, peak, "; ".join(errors) or None
                    )
                )
    return rows


def _has_triton() -> bool:
    from myelin._optional import has_triton

    return has_triton()


def print_markdown(args, rows: list[Row]) -> None:
    print("# WKV Throughput: Triton vs chunked/parallel matmul")
    print()
    print(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    print(f"Device: {args.device} ({gpu_name(args.device)})")
    print(f"torch: {torch.__version__}")
    print(
        f"channels={args.channels}, dtype={args.dtype}, "
        f"matmul_precision={args.matmul_precision}, repeats={args.repeats}, "
        f"chunk_sizes={args.chunk_sizes}"
    )
    print()
    print(
        "`max_err` vs the Triton kernel (CUDA) or parallel (CPU). Lower fwd/fwd+bwd ms is better."
    )
    print()
    print("| Variant | B | T | Max err | Fwd ms | Fwd+Bwd ms | Peak mem | Error |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:

        def fmt_ms(x):
            return "" if x is None else format_ms(x / 1000.0)

        err = "" if r.max_err is None else f"{r.max_err:.2e}"
        print(
            f"| {r.variant} | {r.batch} | {r.timesteps} | {err} | "
            f"{fmt_ms(r.fwd_ms)} | {fmt_ms(r.fwd_bwd_ms)} | "
            f"{format_memory(r.peak_bytes)} | {r.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[12, 24, 64])
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[1024, 3072])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    args = parser.parse_args()
    print_markdown(args, run(args))


if __name__ == "__main__":
    main()
