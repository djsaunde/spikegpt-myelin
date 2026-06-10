"""Compare WKV recurrence formulations for torch.compile cost and correctness.

The SpikeGPT time-mixing recurrence (``spikegpt.language.weighted_key_value``) is a
Python ``for step in range(T)`` loop. Under ``torch.compile`` that loop is
unrolled, so the graph grows with the context length ``T`` and Inductor compile
time blows up. This benchmark compares the reference loop against three
loop-free formulations that keep the compiled graph small:

- ``scan``: the same per-step recurrence wrapped in ``torch._higher_order_ops.scan``
  (single graph node, independent of ``T``).
- ``parallel``: a fully parallel O(T^2) intra-span decay-matrix formulation.
- ``chunked``: the same decay-matrix formulation applied per chunk, threading a
  stable carry across chunks (graph unroll is ``T / chunk_size``).

For each variant and context length it reports forward/backward correctness
versus the reference, eager step time, first compiled step time (which includes
the compile), and steady compiled step time.
"""

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

from spikegpt.language import weighted_key_value as wkv_reference

# --- Variant implementations -------------------------------------------------


def _wkv_span(
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    first: torch.Tensor,
    num0: torch.Tensor,
    den0: torch.Tensor,
    q0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel WKV over a span via an O(L^2) intra-span decay matrix.

    Every exponent is <= 0 (bounded) and a span-local key max keeps ``exp(key)``
    finite, so this is overflow-free for any decay/length and autograd-safe.
    Returns ``(out, num, den, q)`` carried stable state (true acc = exp(q) * num).
    """
    _, t, _ = key.shape
    dtype, device = key.dtype, key.device
    idx = torch.arange(t, device=device, dtype=dtype)
    expo = idx[:, None] - 1.0 - idx[None, :]  # (L,L) = i-1-j
    mask = idx[None, :] < idx[:, None]  # strictly causal, exclusive of i
    expo = torch.where(mask, expo, torch.zeros_like(expo))
    decay_mat = torch.exp(decay[:, None, None] * expo[None]) * mask[None].to(dtype)  # (C,L,L)

    km = key.max(dim=1, keepdim=True).values  # (B,1,C) span-local key max
    ek = torch.exp(key - km)
    intra_num = torch.exp(km) * torch.einsum("cij,bjc->bic", decay_mat, ek * value)
    intra_den = torch.exp(km) * torch.einsum("cij,bjc->bic", decay_mat, ek)

    carry = torch.exp((idx[None, :, None] * decay[None, None, :]) + q0[:, None, :])
    acc_num = carry * num0[:, None, :] + intra_num
    acc_den = carry * den0[:, None, :] + intra_den
    bonus = torch.exp(first[None, None] + key)
    out = (acc_num + bonus * value) / (acc_den + bonus)

    km0 = km[:, 0, :]
    w_end = torch.exp(decay[:, None] * (t - 1.0 - idx[None, :]))  # (C,L), exponent <= 0
    sum_num = torch.einsum("cj,bjc->bc", w_end, ek * value)
    sum_den = torch.einsum("cj,bjc->bc", w_end, ek)
    carried_level = q0 + (t * decay[None, :])
    m = torch.maximum(carried_level, km0)
    num_new = torch.exp(carried_level - m) * num0 + torch.exp(km0 - m) * sum_num
    den_new = torch.exp(carried_level - m) * den0 + torch.exp(km0 - m) * sum_den
    return out, num_new, den_new, m


def _decay_first(key, time_decay, time_first):
    dtype, device = key.dtype, key.device
    decay = -torch.exp(time_decay.to(device=device, dtype=dtype))
    first = time_first.to(device=device, dtype=dtype)
    return decay, first


def _zero_state(key):
    b, _, c = key.shape
    z = key.new_zeros((b, c))
    q = torch.full((b, c), torch.finfo(key.dtype).min, dtype=key.dtype, device=key.device)
    return z, z.clone(), q


def wkv_parallel(key, value, time_decay, time_first):
    decay, first = _decay_first(key, time_decay, time_first)
    num0, den0, q0 = _zero_state(key)
    out, *_ = _wkv_span(key, value, decay, first, num0, den0, q0)
    return out


def wkv_chunked(key, value, time_decay, time_first, *, chunk_size=32):
    decay, first = _decay_first(key, time_decay, time_first)
    num, den, q = _zero_state(key)
    t = key.shape[1]
    outs = []
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        out, num, den, q = _wkv_span(
            key[:, start:end], value[:, start:end], decay, first, num, den, q
        )
        outs.append(out)
    return torch.cat(outs, dim=1)


def wkv_scan(key, value, time_decay, time_first):
    from torch._higher_order_ops.scan import scan

    decay, first = _decay_first(key, time_decay, time_first)
    num0, den0, q0 = _zero_state(key)
    k_tm = key.movedim(1, 0).contiguous()
    v_tm = value.movedim(1, 0).contiguous()

    def combine(carry, x):
        num, den, qq = carry
        k_t, v_t = x
        a = torch.maximum(qq, first + k_t)
        os = torch.exp(qq - a)
        ns = torch.exp(first + k_t - a)
        out = (os * num + ns * v_t) / (os * den + ns)
        dq = qq + decay
        bb = torch.maximum(dq, k_t)
        os2 = torch.exp(dq - bb)
        ns2 = torch.exp(k_t - bb)
        return (os2 * num + ns2 * v_t, os2 * den + ns2, bb), out

    _, ys = scan(combine, (num0, den0, q0), (k_tm, v_tm))
    return ys.movedim(0, 1)


# --- Measurement -------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    variant: str
    timesteps: int
    fwd_err: float | None
    bwd_err: float | None
    eager_ms: float | None
    eager_peak_bytes: int | None
    compile_first_ms: float | None
    compiled_steady_ms: float | None
    error: str | None = None


def _make_inputs(args, t, device):
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch, t, args.channels)
    key = (torch.randn(shape, generator=g) * 0.5).to(device)
    value = torch.randn(shape, generator=g).to(device)
    time_decay = (-5.0 + 8.0 * torch.rand(args.channels, generator=g)).to(device)
    time_first = (
        torch.ones(args.channels) * -1.2 + torch.randn(args.channels, generator=g) * 0.5
    ).to(device)
    return key, value, time_decay, time_first


def _grads(fn, key, value, time_decay, time_first) -> tuple[torch.Tensor, ...]:
    tensors = [t.clone().requires_grad_(True) for t in (key, value, time_decay, time_first)]
    fn(*tensors).square().sum().backward()
    return tuple(cast(torch.Tensor, t.grad) for t in tensors)


def _time_call(fn, *inputs, repeats, device):
    start = time.perf_counter()
    for _ in range(repeats):
        _grads(fn, *inputs)
    synchronize_if_needed(device)
    return (time.perf_counter() - start) / repeats


def _one_line(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".splitlines()[0]
    return text if len(text) <= 90 else text[:87] + "..."


def run(args) -> list[Row]:
    device = torch.device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.cold_compile:
        # Measure true cold-compile cost: defeat the persistent Inductor caches.
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True
    # Absorb one-time Dynamo/Inductor startup so the first variant is not penalized.
    warm = torch.compile(lambda x: x * 2.0 + 1.0)
    warm(torch.zeros(4, device=device))
    synchronize_if_needed(device)
    rows: list[Row] = []
    for t in args.timesteps:
        key, value, time_decay, time_first = _make_inputs(args, t, device)
        ref_out = wkv_reference(key, value, time_decay, time_first)
        ref_grads = _grads(wkv_reference, key, value, time_decay, time_first)
        base_fns = {
            "reference_loop": wkv_reference,
            "scan": wkv_scan,
            "parallel": wkv_parallel,
        }
        variant_list: list[tuple[str, dict]] = [(n, {}) for n in base_fns]
        variant_list += [(f"chunked{size}", {"chunk_size": size}) for size in args.chunk_sizes]
        for name, kw in variant_list:
            fn = base_fns.get(name, wkv_chunked)

            def call(k, v, d, f, _fn=fn, _kw=kw):
                return _fn(k, v, d, f, **_kw)

            # Correctness + eager timing + peak memory (independent of compile).
            fwd_err = bwd_err = eager_ms = None
            eager_peak_bytes = None
            errors: list[str] = []
            try:
                out = call(key, value, time_decay, time_first)
                fwd_err = (out - ref_out).abs().max().item()
                grads = _grads(call, key, value, time_decay, time_first)
                bwd_err = max(
                    (g - r).abs().max().item() for g, r in zip(grads, ref_grads, strict=True)
                )
                reset_cuda_peak_memory(device)
                _grads(call, key, value, time_decay, time_first)
                synchronize_if_needed(device)
                eager_peak_bytes = max_cuda_memory_allocated(device)
                eager_ms = (
                    _time_call(
                        call,
                        key,
                        value,
                        time_decay,
                        time_first,
                        repeats=args.repeats,
                        device=device,
                    )
                    * 1000.0
                )
            except Exception as exc:  # noqa: BLE001 - benchmark reports failures.
                synchronize_if_needed(device)
                errors.append(f"eager {_one_line(exc)}")

            # Compile timing (skipped for the reference loop above a cutoff: too slow).
            compile_first_ms = steady_ms = None
            if name == "reference_loop" and t > args.ref_compile_max_t:
                errors.append("compile skipped (slow)")
            else:
                try:
                    # Independent cold compile per (variant, T): clear all prior
                    # Dynamo state so guards/recompiles do not leak between rows.
                    torch._dynamo.reset()
                    compiled = torch.compile(call, fullgraph=args.fullgraph, mode=args.compile_mode)
                    start = time.perf_counter()
                    _grads(compiled, key, value, time_decay, time_first)
                    synchronize_if_needed(device)
                    compile_first_ms = (time.perf_counter() - start) * 1000.0
                    steady_ms = (
                        _time_call(
                            compiled,
                            key,
                            value,
                            time_decay,
                            time_first,
                            repeats=args.repeats,
                            device=device,
                        )
                        * 1000.0
                    )
                except Exception as exc:  # noqa: BLE001 - benchmark reports failures.
                    synchronize_if_needed(device)
                    errors.append(f"compile {_one_line(exc)}")

            rows.append(
                Row(
                    name,
                    t,
                    fwd_err,
                    bwd_err,
                    eager_ms,
                    eager_peak_bytes,
                    compile_first_ms,
                    steady_ms,
                    "; ".join(errors) or None,
                )
            )
    return rows


def print_markdown(args, rows: list[Row]) -> None:
    print("# SpikeGPT WKV Compile/Correctness Comparison")
    print()
    print(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    print(f"Device: {args.device} ({gpu_name(args.device)})")
    print(f"torch: {torch.__version__}")
    print(
        f"batch={args.batch}, channels={args.channels}, chunk_sizes={args.chunk_sizes}, "
        f"matmul_precision={args.matmul_precision}, repeats={args.repeats}, "
        f"compile_mode={args.compile_mode}, fullgraph={args.fullgraph}"
    )
    print()
    print("Timings/peak memory are forward+backward. `compile_first_ms` includes compilation.")
    print()
    print(
        "| Variant | T | Fwd err | Bwd err | Eager ms | Eager peak | "
        "Compile+1st ms | Steady ms | Error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:

        def fmt(x, sci=False):
            if x is None:
                return ""
            return f"{x:.2e}" if sci else format_ms(x / 1000.0)

        print(
            f"| {r.variant} | {r.timesteps} | {fmt(r.fwd_err, True)} | {fmt(r.bwd_err, True)} | "
            f"{fmt(r.eager_ms)} | {format_memory(r.eager_peak_bytes)} | "
            f"{fmt(r.compile_first_ms)} | {fmt(r.compiled_steady_ms)} | {r.error or ''} |"
        )
    print()
    print("## Notes")
    print()
    print(
        "- `compile_first_ms` is a cold compile (Inductor caches disabled when `--cold-compile`);"
        " `torch._dynamo.reset()` runs before each row so compiles are independent."
    )
    print(
        "- `reference_loop` unrolls the per-step recurrence, so its compile time scales"
        " (super-linearly) with `T` and is skipped above `--ref-compile-max-t`."
    )
    print(
        "- `parallel` is loop-free (single graph), so its compile time is ~flat in `T`;"
        " cost is O(T^2) intra-span memory/matmul."
    )
    print(
        "- `chunked` compile time scales with `T / chunk_size` because the chunk loop still"
        " unrolls under `fullgraph`; larger chunks reduce the unroll but raise O(chunk^2) memory."
    )
    print(
        "- `scan` (`torch._higher_order_ops.scan`) is exact in the forward but, in this torch"
        " build, returns wrong gradients eagerly and fails Inductor compilation (aliasing)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--ref-compile-max-t",
        type=int,
        default=128,
        help="skip compiling the reference loop above this T (compile is too slow)",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cold-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable Inductor caches so compile-time reflects a true cold compile",
    )
    args = parser.parse_args()
    print_markdown(args, run(args))


if __name__ == "__main__":
    main()
