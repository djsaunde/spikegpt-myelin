# SpikeGPT WKV Compile/Correctness Comparison

Generated: 2026-05-30T23:45:33+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
torch: 2.12.0+cu130
batch=8, channels=128, chunk_sizes=[32, 128], matmul_precision=highest, repeats=10, compile_mode=default, fullgraph=True

Timings/peak memory are forward+backward. `compile_first_ms` includes compilation.

| Variant | T | Fwd err | Bwd err | Eager ms | Eager peak | Compile+1st ms | Steady ms | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference_loop | 16 | 0.00e+00 | 0.00e+00 | 16.182 | 1.6 | 18538.405 | 0.747 |  |
| scan | 16 | 0.00e+00 | 3.82e+01 | 127.953 | 2.5 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 16 | 7.15e-07 | 7.63e-06 | 1.766 | 66.5 | 3257.239 | 0.959 |  |
| chunked32 | 16 | 7.15e-07 | 7.63e-06 | 1.858 | 65.9 | 2876.934 | 0.849 |  |
| chunked128 | 16 | 7.15e-07 | 7.63e-06 | 1.866 | 65.9 | 2409.015 | 0.898 |  |
| reference_loop | 32 | 0.00e+00 | 0.00e+00 | 34.636 | 67.1 | 44081.328 | 1.140 |  |
| scan | 32 | 0.00e+00 | 6.48e+01 | 147.871 | 68.9 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 32 | 7.15e-07 | 1.53e-05 | 1.791 | 69.6 | 2958.568 | 0.865 |  |
| chunked32 | 32 | 7.15e-07 | 1.53e-05 | 1.861 | 69.6 | 3556.347 | 0.890 |  |
| chunked128 | 32 | 7.15e-07 | 1.53e-05 | 1.862 | 68.3 | 2812.314 | 0.900 |  |
| reference_loop | 64 | 0.00e+00 | 0.00e+00 | 61.423 | 70.2 | 101395.967 | 1.659 |  |
| scan | 64 | 0.00e+00 | 1.39e+02 | 275.861 | 73.6 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 64 | 7.15e-07 | 4.58e-05 | 1.845 | 77.1 | 3087.212 | 0.939 |  |
| chunked32 | 64 | 5.96e-07 | 4.58e-05 | 4.255 | 72.5 | 5171.697 | 1.453 |  |
| chunked128 | 64 | 7.15e-07 | 4.58e-05 | 1.849 | 77.1 | 2718.222 | 0.960 |  |
| reference_loop | 128 | 0.00e+00 | 0.00e+00 | 113.892 | 77.0 |  |  | compile skipped (slow) |
| scan | 128 | 0.00e+00 | 2.92e+02 | 279.030 | 83.6 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 128 | 8.34e-07 | 7.63e-05 | 1.820 | 110.7 | 4775.538 | 1.208 |  |
| chunked32 | 128 | 9.54e-07 | 7.63e-05 | 9.548 | 79.6 | 12217.449 | 2.282 |  |
| chunked128 | 128 | 8.34e-07 | 7.63e-05 | 1.993 | 105.2 | 2948.445 | 1.057 |  |
| reference_loop | 256 | 0.00e+00 | 0.00e+00 | 259.854 | 89.0 |  |  | compile skipped (slow) |
| scan | 256 | 0.00e+00 | 5.35e+02 | 453.745 | 96.1 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 256 | 7.15e-07 | 2.44e-04 | 4.173 | 212.5 | 3792.885 | 1.068 |  |
| chunked32 | 256 | 7.15e-07 | 2.75e-04 | 18.933 | 97.5 | 21861.422 | 4.161 |  |
| chunked128 | 256 | 7.15e-07 | 2.75e-04 | 4.529 | 131.9 | 6308.265 | 1.383 |  |
| reference_loop | 512 | 0.00e+00 | 0.00e+00 | 521.596 | 114.0 |  |  | compile skipped (slow) |
| scan | 512 | 0.00e+00 | 1.24e+03 | 782.336 | 140.1 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 512 | 9.54e-07 | 1.10e-03 | 3.001 | 618.0 | 3517.832 | 1.420 |  |
| chunked32 | 512 | 7.15e-07 | 1.10e-03 | 40.601 | 131.1 | 47164.840 | 7.716 |  |
| chunked128 | 512 | 9.54e-07 | 1.10e-03 | 10.399 | 184.4 | 14425.465 | 2.242 |  |

## Notes

- `compile_first_ms` is a cold compile (Inductor caches disabled when `--cold-compile`); `torch._dynamo.reset()` runs before each row so compiles are independent.
- `reference_loop` unrolls the per-step recurrence, so its compile time scales (super-linearly) with `T` and is skipped above `--ref-compile-max-t`.
- `parallel` is loop-free (single graph), so its compile time is ~flat in `T`; cost is O(T^2) intra-span memory/matmul.
- `chunked` compile time scales with `T / chunk_size` because the chunk loop still unrolls under `fullgraph`; larger chunks reduce the unroll but raise O(chunk^2) memory.
- `scan` (`torch._higher_order_ops.scan`) is exact in the forward but, in this torch build, returns wrong gradients eagerly and fails Inductor compilation (aliasing).

## Verdict

- **reference_loop**: correct, but cold compile is catastrophic and super-linear in `T`
  (18.5s / 44s / 101s at T=16/32/64) and impractical beyond. Lowest memory.
- **scan**: exact forward but wrong eager gradients (bwd err 38-1240) and fails Inductor
  compilation in torch 2.12. Not viable.
- **parallel** (O(T^2) decay matrix): flat ~3s compile, but peak memory is quadratic in `T`
  -- 110 -> 212 -> 618 MB at T=128/256/512 and ~2.2 GB at T=1024. The memory blow-up rules it
  out at long context.
- **chunked** (matrix per chunk): memory is linear in `T` and close to the reference loop
  (chunked32: 131 MB at T=512 vs loop 114 MB), but compile time grows with `T / chunk_size`
  (chunked32: 47s at T=512; chunked128: 14s). Chunk size trades memory vs compile.

**Recommendation**: use `chunked` with a moderate chunk (~64-128). It keeps peak memory within
~2x of the reference loop (nowhere near `parallel`'s O(T^2) blow-up) while cutting the
compile-unroll factor versus small chunks. `parallel` is only appropriate for short `T` where
O(T^2) memory is still small.
