# WKV recurrence: why `associative_scan`

The SpikeGPT time-mixing block uses the RWKV "weighted key-value" (WKV)
recurrence (`myelin.language.weighted_key_value`). It is a first-order linear
recurrence over the token/time axis: a decay-weighted running average of past
values. This note records why the **CPU / fallback** implementation is an
`associative_scan` higher-order op rather than a Python time loop.

> **Update:** on CUDA the default is now a hand-written **fused Triton kernel**
> (`myelin.wkv_triton`, forward + backward ported from the reference RWKV
> `cuda/wkv_cuda.cu`, registered as the `myelin::wkv_forward` / `wkv_backward`
> `torch.library` custom ops). It matches the loop oracle's gradients to fp32
> (~1e-7). Measured vs `associative_scan` (6L/512d ctx1024 bf16 regional-compile):
> the **full training step is on par** (WKV is only ~36% of the step and Inductor
> already compiles the scan well); the real wins are a **~8x faster forward** in
> isolation (helps eval / forward-only paths), **0 graph breaks** (it stays
> in-graph as an opaque op), and a somewhat **faster cold compile** (~2.0s vs
> ~3.3s per block). It is also the implementation the reference SpikeGPT uses.
> `SpikeTimeMix.forward` dispatches to it when CUDA + Triton are available and
> falls back to `associative_scan` otherwise. The rest of this note explains the
> `associative_scan` fallback.

## The problem

The obvious implementation is a `for t in range(T)` loop
(`weighted_key_value_loop`, kept as the correctness oracle). It is correct and
memory-light, but `torch.compile` **unrolls** the Python loop, so the graph
grows with the context length `T` and Inductor compile time explodes
super-linearly. On an RTX 5090 (cold compile, fwd+bwd): ~107 s at `T=64`,
~294 s at `T=128`, and impractical beyond. Compiling the model — which is where
`torch.compile` pays off for SNNs — becomes the bottleneck.

## Variants considered

All are exact in the forward; the differences are compile time, peak memory, and
whether backward (training) is correct. RTX 5090, batch 8, channels 128, fp32,
cold compile, fwd+bwd (see `benchmarks/results/spikegpt_wkv_compare_5090.md` and
`spikegpt_wkv_compare.py`):

| Variant | Compile @ T=1024 | Steady runtime @ T=1024 | Peak mem @ T=1024 | Training backward |
|---|--:|--:|--:|:--:|
| loop (`weighted_key_value_loop`) | uncompilable (>>100 s) | — | low | correct |
| parallel (O(T²) decay matrix) | ~3.9 s | 4.7 ms | 2299 MB | correct |
| chunked (decay matrix per chunk) | ~49 s | 6.6 ms | 232 MB | correct |
| `scan` HOP | fails / wrong | — | low | **WRONG** |
| **`associative_scan` (generic)** | ~51 s | **1.8 ms** | **231 MB** | **correct** |

> **Throughput at production scale (C=512, T=1024-3072):** the table above is
> compile-focused at C=128. A separate throughput benchmark
> (`myelin.benchmarks.wkv_throughput`,
> [results](../benchmarks/results/wkv_throughput_rtxpro6000.md)) pits the
> production **Triton kernel** against the chunked/parallel matmul forms at real
> shapes. The Triton kernel is **5-9x faster (fwd+bwd) and lighter at every
> shape** — e.g. B=24/T=3072 bf16: 7.5 ms vs 50 ms (chunked-128). The reason is
> structural: RWKV-v4 WKV is a **per-channel diagonal** recurrence with no
> contraction (head) dimension, so the matmul form is C independent tiny L×L
> matmuls doing O(C·T·chunk) FLOPs vs the recurrence's O(C·T) — it cannot fill
> tensor cores. Chunked-matmul is therefore **not** a throughput lever here; the
> only matmul-friendly route is an architecture change (matrix-valued / multi-head
> state, à la RWKV-5/6 gated linear attention).

## Why `associative_scan` (generic mode)

WKV is a linear recurrence, so it fits an associative scan: each token is a
log-space monoid element `(acc_decay, log_scale, num, den)` with true
accumulator `exp(log_scale) * num`, and the associative `combine` decays the
earlier segment by the later segment's total decay before merging (with a shared
max-exponent for stability). This gives, at long context:

- **Low, linear memory** — 231 MB at `T=1024`, vs the parallel form's O(T²)
  2.3 GB.
- **Best compiled runtime** — flat ~1.8 ms, beating both parallel and chunked.
- **Correct training** — forward exact, backward correct *including the
  `time_decay`/carry gradient*, numerically stable to `T=1024`.

The one wart is compile time: it grows with `T` (the prototype generic-mode
autograd materializes a joint backward graph), ~51 s cold at `T=1024`. Unlike
the loop it always completes, and it is a one-time, cacheable cost.

### Rejected alternatives

- **`scan` HOP**: forward is exact and compiles (with `.clone()` on the
  `combine_fn` outputs to satisfy the no-aliasing rule), but its backward
  **drops the carry-recurrence gradient** — a minimal `s_t = 0.9·s_{t-1} + x_t`
  test returns gradient `[1,1,…]` instead of the correct `[4.69, 4.10, …]`. This
  holds eager and compiled, in torch 2.12 and 2.13.dev. Unusable for training.
- **`associative_scan` pointwise mode** (CUDA-only): faster/flatter compile, but
  produces NaN/exploding gradients and parallelizes the whole scan tree into
  ~34 GB at `T=1024`. Unusable.

## End-to-end: the next bottleneck is `SpikingSequenceLIF`

Fixing WKV does **not** by itself make the whole SpikeGPT model compile fast.
`SpikeGPTBlock.forward` also applies two `SpikingSequenceLIF` activations, each of
which is its own `for step in range(T)` loop that `torch.compile` unrolls. An
end-to-end `spikegpt_compile_probe` (context 32, 2 layers, 128 embedding, RTX
5090, `fullgraph=True`) is correct (compiled loss matches eager) and fast once
warm (compiled steady 4.6 ms), but the cold compile still takes ~57 s at `T=32`
— now dominated by the four unrolled LIF loops, not WKV.

So the WKV is no longer the compile bottleneck, but the LIF activations are.
Unlike WKV, however, LIF does **not** get the same treatment: the hard reset
folds in as a per-step multiplier `decay * (1 - spike_{t-1}) ∈ {decay, 0}` that
depends on whether the membrane crossed threshold, i.e. the recurrence is
**nonlinear / state-dependent**, not a constant-coefficient linear recurrence.
That breaks the associativity `associative_scan` needs — you cannot merge two
segments without first knowing whether the earlier one spiked. So LIF is
genuinely sequential.

Benchmarking the alternatives (loop vs the fused Triton surrogate kernel vs
checkpointed recompute), the **compiled loop wins**: Inductor shortens buffer
lifetimes so its peak memory stays low, while the Triton surrogate kernel stores
the full `[T, B, C]` pre-reset trace for its fused backward and so uses
materially more memory; recompute trades that memory back for extra FLOPs but
does not beat the loop. So the loop is the right LIF kernel — its only liability
is compile latency.

The natural loop-free tool for a sequential nonlinear recurrence is the **`scan`
higher-order op** (one un-unrolled graph node: flat compile, low memory, the
reset handled inside the per-step combine). Its forward already works, but its
**backward is broken** in current torch (it drops the carry gradient — the same
blocker WKV's `scan` hit; `associative_scan` got a correct autograd impl first).
So `scan`-LIF is the queued win: adopt it once PyTorch ships correct `scan`
autograd. Until then the loop is retained and its compile cost is made one-time
via persistent compilation caching (Inductor `fx_graph_cache` + Mega-Cache):

| `spikegpt_compile_probe` (ctx 32, 2 layers, RTX 5090) | compiled first step | steady |
|---|--:|--:|
| cold (empty cache) | 58.7 s | 4.36 ms |
| warm (shared cache dir) | 6.5 s | 4.34 ms |

So a warm cache cuts the model's cold compile ~9x (58.7 s -> 6.5 s) with identical
steady runtime and loss. The residual ~6.5 s is Dynamo re-tracing the unrolled
graph (Inductor codegen is cached), so for a fixed-shape training run the heavy
LIF compile is paid once, not every run.

## torch 2.13 requirement

`associative_scan` autograd is only correct in torch 2.13+, which at time of
writing is a **nightly** build (`pyproject.toml` pulls torch/torchvision/triton
from the PyTorch nightly CUDA index). Caveat: a pinned nightly dev build is
garbage-collected from the index after a few weeks — re-run `uv lock` if a sync
fails. Switch the index back to PyPI and pin `torch>=2.13` once 2.13 ships
stable.
