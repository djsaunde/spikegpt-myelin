# SpikeGPT Compile Probe

Generated: 2026-05-25T18:39:03.276992+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
Shape: batch=2, context_length=32, layers=2, embedding=32, vocab_size=256
compile_mode=reduce-overhead; fullgraph=True; matmul_precision=high; repeats=3; seed=0

| Phase | Time | Peak memory | Loss | Error |
|---|---:|---:|---:|---|
| eager_first_step | 775.147 | 65.4 | 5.545857 |  |
| eager_steady_step | 108.469 | 65.8 | 5.411427 |  |
| compile_wrapper | 88.572 |  |  |  |
| compiled_first_step | 89155.817 | 129.5 | 5.545857 |  |
| compiled_steady_step | 11.340 | 33.9 | 5.411427 |  |
