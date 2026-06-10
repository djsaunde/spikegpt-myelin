# Context-length warmup: wall-clock reduction at matched quality (RTX 5090)

**Lever:** train the early phase at a shorter context length, then grow to the
full context. v4's WKV recurrence is *sequential over time*, so a step's
wall-clock is latency-bound in the context length `T`. Holding tokens-per-step
constant (`batch * ctx` fixed, so batch grows as ctx shrinks) means the
matmul/FFN/head work is unchanged while the WKV sequential depth shrinks — a
strictly cheaper step on the same number of tokens. The model is **identical**
(v4, unchanged), and RWKV length-generalizes, so evaluation is at the full
context regardless of the training schedule.

Harness: `benchmarks/ctx_warmup_ab.py` (same model init + seeded batch stream for
both arms; val BPC on a fixed held-out enwik8 slice at the full eval context vs
cumulative training seconds). Proxy model 6L/384d, eager, AdamW lr 2e-3.
Schedule `0.25:0.4, 0.5:0.3, 1.0:0.3` (ctx fraction : step fraction).

| eval ctx | arm | final val BPC | wall-clock | Δ wall-clock |
|---:|---|---:|---:|---:|
| 512 | constant | 1.6986 | 116.7s | — |
| 512 | warmup 128→256→512 | 1.6991 | **106.7s** | **−8.6%** |
| 1024 | constant | 1.6447 | 372.1s | — |
| 1024 | warmup 256→512→1024 | **1.6371** | **345.4s** | **−7.2%** |
| 3072 | constant | 1.8938 | 396.7s | — |
| 3072 | warmup 768→1536→3072 | **1.8679** | **336.9s** | **−15.1%** |

### Aggressive schedule at ctx-3072 (50% of steps at ctx-384, 20% at full)

Schedule `0.125:0.5, 0.25:0.3, 1.0:0.2` (separate 800-step run; constant baseline
1.8956 BPC / 375.9s):

| arm | final BPC | wall-clock | iso-quality time to 1.8956 |
|---|---:|---:|---:|
| constant ctx-3072 | 1.8956 | 375.9s | 375.9s |
| aggressive warmup 384→768→3072 | **1.8484** | **312.6s** (−16.8%) | **222.7s (−41%)** |

The aggressive schedule is better on both axes at equal steps. The iso-quality
−41% conflates two effects: the WKV-depth throughput saving (~15%, robust) **and**
faster convergence from the larger early batch (ctx-384 runs at B128). On an
800-step proxy the batch-size convergence boost is amplified and only 20% of steps
see full context; a full converged run will shrink the upside and may need more
full-context training for long-range quality. **Safe number: ~15% (throughput);
~17–41% is upside pending a full-run confirmation.**

**Findings**

- A consistent **~7–9% wall-clock reduction at matched quality** (BPC equal within
  eval noise; at ctx-1024 the warmup arm was actually slightly *better*, from the
  larger early batches + easy-context-first curriculum).
- The saving **grows with context once WKV dominates the step**: ~7–9% at
  ctx-512/1024 (where the constant matmul/FFN/head work still dominates), rising
  to **15% at ctx-3072** — the headline repro config — where the sequential WKV is
  the bulk of the step time and cutting its early depth saves the most. (At
  ctx-3072 the warmup arm was also *better* on BPC, 1.8679 vs 1.8938.)
- Zero architecture/quality risk (identical v4 model), no new dependencies, and
  composes with any other lever.

**Shipped:** `examples/train_tiny_spikegpt.py --ctx-schedule "256:0.4,512:0.3,1024:0.3"`.

> **Update (speedrun):** these numbers are **eager**. In the compiled production path, ctx-warmup needs `dynamic=True` shape-polymorphic compile, whose slower kernels negate the saving — and at a *meaningful* BPC target it shows no convergence advantage (same step-to-target as baseline). See `speedrun_46m.md`. The eager saving here is real but does not transfer to the compiled run to a real quality bar.

## Context: levers that did *not* beat the tuned v4+AdamW baseline (proxy)

Same 6L/384d / 2000-step proxy, val BPC at fixed steps:

| lever | result | baseline | notes |
|---|---:|---:|---|
| Muon (RMS-matched scale, lr 2e-3) | 1.7749 | 1.7337 | tied through step ~400, then drifts back; 1e-3/4e-3 worse |
| RWKV-5 retention (per-head scalar decay) | 1.7963 | 1.7357 | weakest matrix-state variant |
| RWKV-7 (fla chunked, MVP) | 1.7728 | 1.7341 | handicapped: value-residual disabled, v4 recipe, fla "may be buggy" disclaimer |

The tuned baseline is hard to beat on quality-per-step at proxy scale; the pure
throughput lever (context warmup) is the one that wins. Muon's documented edge is
at 124M+ over full converged runs (modded-nanogpt 1.35x; Moonshot ~2x), measured
as steps-to-target — not fixed-step BPC on a 12M / 2000-step proxy.
