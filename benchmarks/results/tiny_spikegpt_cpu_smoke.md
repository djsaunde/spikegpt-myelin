# Tiny SpikeGPT CPU Smoke

Date: 2026-05-25

Command:

```bash
uv run python examples/train_tiny_spikegpt.py \
  --device cpu \
  --compile off \
  --context-length 32 \
  --layers 2 \
  --embedding 64 \
  --batch 16 \
  --steps 80 \
  --log-every 20 \
  --eval-every 20 \
  --eval-batches 8 \
  --sample-prompt spik \
  --sample-tokens 48
```

Setup:

| Setting | Value |
|---|---:|
| Device | CPU |
| Compile | off |
| Context length | 32 |
| Layers | 2 |
| Embedding | 64 |
| Batch | 16 |
| Steps | 80 |
| Learning rate | 0.003 |
| LIF threshold | 0.0 |
| Spike embedding | true |
| Vocab size | 23 |
| Train tokens | 70 |
| Validation tokens | 64 |
| Parameters | 111,104 |

Results:

| Step | Train Loss | Val Loss | Val BPC | Val PPL | Emb Spike Rate | Mean Block Spike Rate | Step ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.126677 | 3.021503 | 4.3591 | 20.5221 | 0.4962 | 0.4311 | 77.603 |
| 20 | 1.188763 | 2.686323 | 3.8755 | 14.6776 | 0.4814 | 0.4584 | 37.406 |
| 40 | 0.265923 | 2.956408 | 4.2652 | 19.2288 | 0.4837 | 0.4756 | 56.213 |
| 60 | 0.130832 | 3.602291 | 5.1970 | 36.6822 | 0.4847 | 0.4735 | 60.111 |
| 80 | 0.064754 | 3.861815 | 5.5714 | 47.5516 | 0.4742 | 0.4837 | 52.528 |

Summary:

| Metric | Value |
|---|---:|
| Initial train loss | 3.126677 |
| Final train loss | 0.064754 |
| Best validation BPC | 3.8755 |
| Final validation BPC | 5.5714 |
| Average step time | 48.909 ms |
| Post-warmup average step time | 48.546 ms |
| Steady-state average step time | 48.637 ms |

Generated sample:

```text
spiking neural networks trade dense activations for 
```

Takeaways:

- The torch-native SpikeGPT-style model can overfit a tiny character language-model workload.
- The validation metrics expose that overfitting: validation BPC is best around step 20 and degrades as train loss keeps falling.
- Spike-rate diagnostics are nonzero and stable in this smoke, which confirms that the residual LIF activations are active rather than dead.
- This is a correctness and API smoke only. It is not a meaningful language modeling benchmark.
