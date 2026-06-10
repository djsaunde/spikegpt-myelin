# SpikeGPT Generation Benchmark

Generated: 2026-05-25T18:13:10.456021+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
Shape: batch=8, prompt_tokens=64, new_tokens=32, context_length=128, layers=4, embedding=128, vocab_size=512
Warmup: 2; repeats: 5; seed: 0

| Path | Time | Tokens/s | Peak memory | Matches recompute |
|---|---:|---:|---:|---:|
| recompute_context | 5389.239 | 47.5 | 42.7 | True |
| cached_recurrent_state | 425.038 | 602.3 | 36.1 | True |
