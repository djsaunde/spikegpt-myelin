"""Gradient accumulation must reproduce the full-batch gradient exactly.

This is the property the memory strategy rests on: --batch stays the EFFECTIVE
batch, so a batch-dependent LR calibration still applies and d1024/d1280 can fit
the 5090 without changing the optimization.
"""

from __future__ import annotations

import pytest
import torch

from spikegpt import SpikeGPTConfig, SpikeLanguageModel

BATCH, CTX = 8, 16


def _model() -> SpikeLanguageModel:
    torch.manual_seed(0)
    return SpikeLanguageModel(
        SpikeGPTConfig(vocab_size=64, context_length=CTX, n_layer=2, n_embd=32, dropout=0.0)
    )


def _grads(model: SpikeLanguageModel, inputs, targets, accum: int) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    micro = BATCH // accum
    for start in range(0, BATCH, micro):
        window = slice(start, start + micro)
        loss, _ = model(inputs[window], targets[window])
        (loss / accum).backward()
    return {name: p.grad.clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("accum", [2, 4, 8])
def test_accumulated_gradient_matches_full_batch(accum: int) -> None:
    model = _model()
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (BATCH, CTX + 1))
    inputs, targets = ids[:, :CTX], ids[:, 1:]

    full = _grads(model, inputs, targets, accum=1)
    accumulated = _grads(model, inputs, targets, accum=accum)

    scale = max(g.abs().max().item() for g in full.values())
    for name, grad in full.items():
        worst = (grad - accumulated[name]).abs().max().item()
        assert worst / scale < 1e-5, f"{name}: accumulation changed the gradient ({worst:.2e})"
