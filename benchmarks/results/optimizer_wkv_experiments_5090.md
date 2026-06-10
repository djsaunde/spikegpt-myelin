# Optimizer & WKV-variant experiments (RTX 5090)

Speedrun-style attempts to beat the tuned v4 + AdamW baseline. Proxy: 6L/384d,
ctx-512, batch-32, eager (loss-vs-step is compile-invariant), AdamW lr 2e-3 /
wd 0.1 / cosine, same init + seeded batch stream across arms. Harnesses:
`benchmarks/muon_ab.py`, `benchmarks/wkv_ab.py`.

## Muon — does *not* beat tuned AdamW at feasible scale

Corrected to the modern recipe: Moonshot RMS-matching update scale
(`0.2*sqrt(max(m,n))`, so Muon shares AdamW's LR), decoupled weight decay,
momentum warmup 0.85→0.95, nesterov + 5-step Newton-Schulz. Hidden 2D matrices on
Muon; embeddings / head / norms / WKV vectors on AdamW.

| scale | steps | AdamW | Muon | gap |
|---|---:|---:|---:|---:|
| 6L/384d (~12M) | 2000 | 1.7337 | 1.7749 | +0.041 |
| 12L/512d (~40M) | 4000 | 1.5786 | 1.6162 | +0.038 |

**The gap is stable (~0.04 BPC) across a 3× scale / 2× horizon jump**, and Muon
never reaches AdamW's final BPC (steps-to-target). It is briefly *ahead* early
(step 500) then falls behind — a consistent "fast early, worse late" pattern.

Interpretation: the documented Muon wins (modded-nanogpt 1.35× at 124M; Moonshot
~2× at B-scale) are at 124M+ over fully-converged 10B-token runs — far beyond
40M/4000 steps (~65M tokens), so "needs much more scale" can't be ruled out. But
the *stability* of the gap across scale points more toward an architecture
mismatch: SpikeGPT applies a LIF spiking nonlinearity on the activations, which
may not suit Muon's orthogonalized updates. **Verdict: not worth it here without
the full 124M+ converged regime.**

## RWKV-7 — *works* once the value residual is threaded

First attempt (MVP) disabled RWKV-7's cross-layer value residual (`layer_idx=0`
per block) and lost. Fixing it — a per-model shared dict carries block 0's
`v_first` to later blocks, no model surgery — flips the result.

| variant | no value residual | **with value residual** | params |
|---|---:|---:|---|
| v4 | 1.7341 | 1.7358 | 11.7M |
| v7 (fla chunked) | 1.7728 (behind) | **1.7301 (ahead)** | 12.9M |

v7 improves ~0.043 BPC and **crosses above v4 in the late phase** (ahead from step
1500 on) — the signal you want, not final-point noise. So RWKV-7's problem was the
handicap, not scale.

Caveats: small margin (0.006), v7 carries +10% params (LoRA decay/gate), and
`fla` warns its RWKV impl "may be buggy" vs the official repo. RWKV-7's recurrence
is also the ~3× faster tensor-core op (see `wkv_tensorcore_probe.py`), so a holding
win is quality **and** throughput.

| scale | steps | v4 | v7 | gap |
|---|---:|---:|---:|---:|
| 6L/384d | 2000 | 1.7358 | **1.7301** | −0.006 (v7 ahead) |
| 12L/512d | 2500 | **1.7111** | 1.7281 | +0.017 (v7 behind, closing) |

Scale did not simply widen the lead. v7 converges *slower early* (more params to
learn) then catches up: at 6L it crossed v4 by step ~1500; at 12L/2500 steps the
gap had shrunk from +0.066 (step 500) to +0.017 and was still closing. So v7 ≈ v4
on quality (±0.02 BPC), crossover timing pushed later at larger scale — not a
scale-driven win, and not a scale problem either.

### Throughput: the v7 op win does NOT survive integration

Compiled end-to-end step, 12L/512d ctx-512 batch-24 (`wkv_steptime.py`):

| variant | ms/step | vs v4 |
|---|---:|---:|
| v4 (fully compiled) | 39.8 | 1.00x |
| v7 (fla, partial) | 142.3 | **0.28x (3.6x slower)** |

Even though the chunked WKV *op* is ~3x faster in isolation
(`wkv_tensorcore_probe.py`), the full v7 step is 3.6x **slower** because (1) fla's
RWKV-7 kernels graph-break under `torch.compile`, so v7 misses the Inductor
speedup v4 gets, and (2) v7's extra LoRA decay/gate/delta projections add work.

**Verdict:** RWKV-7's value residual gets it to quality parity (real progress),
but it is not a wall-clock win as integrated — the throughput potential is
unrealized pending a compile-friendly or custom-fused RWKV-7 kernel.

## Making RWKV-7 compile-friendly (`src/spikegpt/wkv7_compile.py`)

fla's RWKV-7 chunk kernel (`chunk_dplr_delta_rule`) is wrapped in
`torch.compiler.disable`, so under `torch.compile` it is a hard graph break: the
recurrence never fuses and the v7 step is ~3.6x slower than fully-compiled v4.
`torch._dynamo.allow_in_graph` removes the disable break but then fails on
fake-tensor tracing (the triton kernel has no meta impl) — making fla's kernel
compile-compatible needs a full `torch.library` custom op (register_fake +
register_autograd), a real kernel-integration project.

**Existing-impl survey:** fla ships a *pure-PyTorch* DPLR reference
(`generalized_delta_rule/dplr/naive.py::dplr_chunkwise`) — only aten ops, no
`compiler.disable`. It compiles fullgraph.

**Implemented:** `RWKV7TimeMixCompile` — RWKV-7 entirely in aten (token-shift,
LoRA-gated projections, the DPLR chunked recurrence inlined from fla's reference,
per-head norm, gate/bonus). Validated: **fullgraph `torch.compile` with no graph
breaks, output matches eager to 6e-7, backward clean.** So RWKV-7 is now
compile-friendly (the goal).

**Honest caveats — compile-friendly is not the same as fast here:**

| variant (6L/384d, ctx-512, compiled) | ms/step |
|---|---:|
| v4 (fused sequential WKV) | 18.7 |
| v7 (fla, graph-breaks) | 68.2 |
| v7c (pure-torch, fullgraph, single block ×6) | ~45/block |

- The pure-torch chunked DPLR compiles but is **~14x slower per block than v4**:
  the `chunk_size`-length Python loops unroll into large kernels and everything
  runs fp32. Inductor fuses it but can't beat fla's hand-tuned triton.
- fp32 chunk intermediates are **memory-heavy** (OOM at 12L/512d batch-24).
- The cross-layer value residual mutates a Python dict inside the compiled
  region → per-step recompile; the compile-friendly variant runs self-contained
  (compile-safe threading needs model-level surgery).

**Verdict:** RWKV-7 is now compile-friendly (fullgraph, no breaks). Realizing a
throughput *win* still needs either fla's triton kernel wrapped as a proper
custom op (fast + compile-compatible) or a vectorized/bf16 rewrite of the
pure-torch chunked kernel — the pure-torch path is compile-friendly but slow.

## Bottom line ("do we need more scale?")

Scale is not the blocker for either. Muon's gap is stable across a 3x scale jump
(architecture-fit issue, or needs 124M+ converged). RWKV-7 reaches quality parity
regardless of scale; its blocker is kernel integration (fla + compile), not scale.
The one lever that did improve wall-clock at matched quality remains context-length
warmup (`ctx_warmup_5090.md`).
