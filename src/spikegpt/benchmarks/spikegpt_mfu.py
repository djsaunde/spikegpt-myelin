"""Accurate MFU + profiling for SpikeGPT training steps.

Reports Model FLOPs Utilization the honest way:

- **FLOPs**: counted op-by-op with ``torch.utils.flop_counter.FlopCounterMode``
  over a real forward+backward (not the 6N approximation). This captures exactly
  the matmul FLOPs (Linear projections, the FFN, the vocab head); the WKV
  recurrence and LIF run as elementwise/custom ops and contribute ~no matmul
  FLOPs, which is the point — they cost wall-time without tensor-core work.
  FLOPs are counted **eager** (compile-invariant; compile changes timing, not the
  FLOP count).
- **Step time**: CUDA-event timed forward+backward+optimizer on the *compiled*
  model (what the real run does).
- **Peak**: *measured empirically* on this card via a large bf16 matmul
  microbenchmark (achievable peak, not a spec sheet) — and optionally compared to
  a ``--peak-tflops`` spec figure.

MFU = achieved FLOPs/s / peak FLOPs/s. With ``--trace`` it also captures a
``torch.profiler`` Chrome trace and prints the top CUDA ops grouped into
matmul / WKV / LIF / other, which explains where the non-tensor-core time goes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import cast

import torch
from myelin.baselines import synchronize_if_needed
from myelin.benchmarks.lif import gpu_name
from torch.utils.flop_counter import FlopCounterMode

from spikegpt.language import SPIKEGPT_PRESETS, SpikeGPTConfig, SpikeLanguageModel


def count_fwd_bwd_flops(
    model: SpikeLanguageModel, inputs: torch.Tensor, targets: torch.Tensor
) -> int:
    """Exact forward+backward FLOPs (2*MAC convention), counted eager."""
    was_training = model.training
    model.train()
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        loss, _ = model(inputs, targets)
        loss.backward()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return int(flop_counter.get_total_flops())


def measure_peak_bf16_flops(
    device: torch.device, *, n: int = 8192, iters: int = 50, dtype: torch.dtype = torch.bfloat16
) -> float:
    """Achievable peak FLOP/s from a large square matmul (2*n^3 per matmul)."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    sink = a
    for _ in range(10):
        sink = a @ b
    synchronize_if_needed(device)
    start = time.perf_counter()
    for _ in range(iters):
        sink = a @ b
    synchronize_if_needed(device)
    del sink
    seconds = (time.perf_counter() - start) / iters
    return (2.0 * n**3) / seconds


def measure_step_ms(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    amp: bool,
    warmup: int = 8,
    iters: int = 30,
) -> float:
    """Median forward+backward+optimizer step time (ms)."""

    def amp_ctx():
        if amp and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            loss, _ = model(inputs, targets)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    synchronize_if_needed(device)
    times: list[float] = []
    for _ in range(iters):
        synchronize_if_needed(device)
        start = time.perf_counter()
        step()
        synchronize_if_needed(device)
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _op_group(name: str) -> str:
    low = name.lower()
    if "wkv" in low:
        return "WKV (recurrence)"
    if "lif" in low or "surrogate" in low:
        return "LIF (spiking)"
    if any(k in low for k in ("mm", "gemm", "matmul", "cutlass", "cublas")):
        return "matmul"
    if "adam" in low or "optimizer" in low or "fused_adam" in low:
        return "optimizer"
    if "elementwise" in low or "vectorized" in low or "reduce" in low or "fused" in low:
        return "elementwise/other"
    return "elementwise/other"


@dataclass(frozen=True)
class MFUResult:
    params: int
    matmul_flops_per_step: int
    step_ms: float
    tokens_per_s: float
    achieved_tflops: float
    measured_peak_tflops: float
    mfu_measured: float


def run(args) -> MFUResult:
    device = torch.device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.preset != "custom":
        spec = SPIKEGPT_PRESETS[args.preset]
        layers, embd, ctx = spec.n_layer, spec.n_embd, args.context_length or spec.context_length
    else:
        layers, embd, ctx = args.layers, args.embedding, args.context_length or 1024
    config = SpikeGPTConfig(
        vocab_size=args.vocab_size,
        context_length=ctx,
        n_layer=layers,
        n_embd=embd,
        dropout=0.0,
        gradient_checkpointing=args.activation_checkpointing,
    )
    torch.manual_seed(0)
    model = SpikeLanguageModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, args.vocab_size, (args.batch, ctx + 1), device=device)
    inputs, targets = ids[:, :ctx], ids[:, 1:]

    # FLOPs: eager (compile-invariant). Then optionally compile for timing.
    matmul_flops = count_fwd_bwd_flops(model, inputs, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")
    mode = None if args.compile_mode == "default" else args.compile_mode
    forward: torch.nn.Module = model
    if args.compile == "regional":
        for index, block in enumerate(model.blocks):
            model.blocks[index] = torch.compile(block, mode=mode)  # type: ignore[index]
        if args.compile_tail:
            model.loss_tail = torch.compile(model.loss_tail, mode=mode)  # type: ignore[method-assign]
    elif args.compile == "full":
        # Compile the whole forward incl. the ln_out/head/cross-entropy tail; the
        # WKV/LIF custom ops stay opaque (no loop unroll).
        forward = cast(torch.nn.Module, torch.compile(model, mode=mode))

    step_ms = measure_step_ms(
        forward, optimizer, inputs, targets, device=device, amp=args.amp == "bf16"
    )
    tokens_per_s = args.batch * ctx / (step_ms / 1000.0)
    achieved = matmul_flops / (step_ms / 1000.0)
    peak = measure_peak_bf16_flops(device) if device.type == "cuda" else float("nan")
    result = MFUResult(
        params=params,
        matmul_flops_per_step=matmul_flops,
        step_ms=step_ms,
        tokens_per_s=tokens_per_s,
        achieved_tflops=achieved / 1e12,
        measured_peak_tflops=peak / 1e12,
        mfu_measured=achieved / peak if peak == peak else float("nan"),
    )

    print(f"# SpikeGPT MFU — {gpu_name(args.device)}")
    print()
    print(
        f"config: {layers}L/{embd}d ctx{ctx} vocab{args.vocab_size} batch{args.batch} "
        f"dtype={args.amp} compile={args.compile}{'+tail' if args.compile_tail else ''}"
        f"/{args.compile_mode}"
    )
    print(f"params: {params / 1e6:.1f}M")
    print(f"step time: {step_ms:.2f} ms   throughput: {tokens_per_s:,.0f} tok/s")
    print(
        f"matmul FLOPs/step (fwd+bwd, exact): {matmul_flops / 1e9:.2f} GFLOP "
        f"({matmul_flops / args.batch / ctx / 1e9:.3f} GFLOP/token)"
    )
    print(f"achieved: {result.achieved_tflops:.1f} TFLOP/s")
    print(f"measured bf16 peak (matmul microbench): {result.measured_peak_tflops:.1f} TFLOP/s")
    print(f"**MFU (vs measured peak): {result.mfu_measured * 100:.1f}%**")
    if args.peak_tflops:
        print(
            f"MFU (vs --peak-tflops {args.peak_tflops}): "
            f"{result.achieved_tflops / args.peak_tflops * 100:.1f}%"
        )

    if args.trace and device.type == "cuda":
        _trace(
            forward,
            optimizer,
            inputs,
            targets,
            device,
            amp=args.amp == "bf16",
            out=args.trace_out,
        )
    return result


def _trace(model, optimizer, inputs, targets, device, *, amp, out) -> None:
    from torch.profiler import ProfilerActivity, profile

    def step():
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast("cpu", enabled=False)
        )
        with autocast:
            loss, _ = model(inputs, targets)
        loss.backward()
        optimizer.step()

    for _ in range(5):
        step()
    synchronize_if_needed(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(8):
            step()
        synchronize_if_needed(device)
    prof.export_chrome_trace(out)
    print(f"\nchrome trace -> {out} (open in chrome://tracing or perfetto.dev)")

    groups: dict[str, float] = {}
    for evt in prof.key_averages():
        cuda_us = float(
            getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        )
        if cuda_us > 0:
            groups[_op_group(evt.key)] = groups.get(_op_group(evt.key), 0.0) + cuda_us
    total = sum(groups.values()) or 1.0
    print("\n## CUDA time by op group")
    print("| group | % of GPU time |")
    print("|---|---:|")
    for name, us in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"| {name} | {us / total * 100:.1f}% |")

    # Per-kernel detail: the top GPU consumers (the actionable optimization list).
    kernels = []
    for evt in prof.key_averages():
        cuda_us = float(
            getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        )
        if cuda_us > 0:
            count = int(getattr(evt, "count", 0) or 0)
            kernels.append((evt.key, cuda_us, count))
    kernels.sort(key=lambda k: -k[1])
    print("\n## Top GPU kernels (self CUDA time)")
    print("| kernel | % | total ms | calls | us/call |")
    print("|---|---:|---:|---:|---:|")
    for name, us, count in kernels[:25]:
        per = us / count if count else 0.0
        print(f"| {name[:70]} | {us / total * 100:.1f}% | {us / 1000:.2f} | {count} | {per:.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", default="gpt2-216m", choices=("custom", *SPIKEGPT_PRESETS))
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--embedding", type=int, default=768)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--vocab-size", type=int, default=50277)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--compile", choices=("off", "regional", "full"), default="regional")
    parser.add_argument("--compile-tail", action="store_true", help="compile the head/CE tail")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--peak-tflops", type=float, help="spec peak for a second MFU figure")
    parser.add_argument("--trace", action="store_true", help="capture a torch.profiler trace")
    parser.add_argument("--trace-out", default="spikegpt_trace.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
