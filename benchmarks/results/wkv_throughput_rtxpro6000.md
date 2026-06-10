# WKV Throughput: Triton kernel vs chunked/parallel matmul

Device: NVIDIA RTX PRO 6000 Blackwell (sm_120, same arch as the local RTX 5090)
torch: 2.13.0.dev20260521+cu130 · channels=512 · matmul_precision=high · repeats=20
Source: `myelin.benchmarks.wkv_throughput` (Modal `wkv_throughput`)

## Question

Is a chunked / parallel **matmul** formulation of the RWKV-v4 WKV recurrence
faster than the production sequential Triton kernel
(`myelin.wkv_triton.weighted_key_value_triton`) at our real training shapes
(C=512, T=1024/3072, B=12/24/64)?

Hypothesis going in: the Triton kernel is a serial T-loop with only
`B * ceil(C/64)` programs (~96-512), latency-bound on the recurrence; a chunked
form exposes tensor-core GEMMs per step and might win on a 170-SM card.

## Answer: No — Triton wins by 5-9x everywhere, and uses less memory.

bf16, forward+backward (training path):

| Config | Triton | best chunked | Triton speedup | Triton peak | chunked peak |
|---|---:|---:|---:|---:|---:|
| B=12 T=1024 | 1.35 ms | 11.7 ms (c128) | 8.7x | 0.38 GB | 1.02 GB |
| B=24 T=1024 | 2.14 ms | 11.4 ms (c128) | 5.3x | 0.69 GB | 1.47 GB |
| B=64 T=1024 | 4.51 ms | 25.0 ms (c128) | 5.5x | 1.73 GB | 2.94 GB |
| B=12 T=3072 | 5.38 ms | 32.9 ms (c128) | 6.1x | 1.00 GB | 2.95 GB |
| B=24 T=3072 | 7.48 ms | 50.6 ms (c128) | 6.8x | 1.94 GB | 4.28 GB |
| B=64 T=3072 | 13.7 ms | 107  ms (c256) | 7.8x | 5.06 GB | 10.2 GB |

(fp32 results are within ~5% of bf16 and tell the same story. The `parallel`
O(T^2) form is non-viable: 75 GB peak at T=3072.)

Best chunk size is ~128. Smaller chunks (32/64) are far slower; chunk 256 starts
losing to chunk 128 as O(chunk^2) intra-chunk work dominates.

Accuracy: in bf16 the matmul forms carry ~1.5e-2 to 3e-2 max error vs Triton
(tensor-core bf16 accumulation through the decay matrix) — a training-stability
concern on top of being slower. In fp32 they match to ~1e-6 (chunked) but
parallel still shows ~3e-3 (tf32 matmul under `matmul_precision=high`).

## Why — structural, not an implementation gap

RWKV-v4 WKV is a **per-channel diagonal recurrence**: there is no key/value
contraction (head) dimension. So the "matmul" form is C=512 *independent tiny*
L x L matmuls (one per channel), which do not fill tensor cores, and it
materializes a (C, L, L) decay matrix doing **O(C·T·chunk)** FLOPs versus the
recurrence's **O(C·T)**. The Triton kernel's serial T-loop is a handful of cheap
elementwise ops per step, register-resident and coalesced; even at B=12 it is so
lightweight it dominates. No implementation tweak closes a 5-9x algorithmic-FLOP
gap.

## Recommendation

- **Keep the Triton kernel as the CUDA WKV path. Do not wire in chunked-matmul.**
- The only route to a genuinely matmul-friendly WKV is an **architecture change**
  — matrix-valued / multi-head state (RWKV-5/6 "gated linear attention"), which
  has a real contraction dimension that maps to large GEMMs. That changes the
  model and requires retraining; it is a research direction, not a kernel swap.
- The chunked/parallel forms retain value only as the CPU / no-Triton fallback
  and as a correctness oracle (they match the loop to fp32).
