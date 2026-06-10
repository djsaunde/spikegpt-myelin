# SpikeGPT Training Benchmark

Generated: 2026-05-25T18:30:17.126689+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
Shape: batch=4, context_length=64, layers=4, embedding=64, vocab_size=512
Warmup: 1; repeats: 3; seed: 0; compile=False; activation_checkpointing=True; matmul_precision=high

| Path | Step time | Tokens/s | Peak memory | Loss | Error |
|---|---:|---:|---:|---:|---|
| eager | 463.598 | 552.2 | 81.0 | 6.022974 |  |
| eager_activation_checkpointed | 738.892 | 346.5 | 72.1 | 6.022974 |  |
