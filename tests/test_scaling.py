"""Exactness tests for spikegpt.scaling: closed-form counts vs ``sum(p.numel())``
and 6N FLOPs vs ``FlopCounterMode`` on a real fwd+bwd."""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from spikegpt import SpikeGPTConfig, SpikeLanguageModel
from spikegpt.scaling import (
    count_spikegpt_params,
    flops_per_token,
    tokens_for_flop_budget,
    training_flops,
)

SIZE_CASES = [
    dict(vocab_size=277, n_layer=2, n_embd=64, model_type="rwkv"),
    dict(vocab_size=277, n_layer=3, n_embd=96, model_type="rwkv-ffn-pre"),
    dict(vocab_size=50277, n_layer=4, n_embd=128, model_type="rwkv"),
    dict(vocab_size=256, n_layer=12, n_embd=512, model_type="rwkv"),
]


def _build(case: dict) -> SpikeLanguageModel:
    config = SpikeGPTConfig(context_length=32, dropout=0.0, **case)
    return SpikeLanguageModel(config)


@pytest.mark.parametrize("case", SIZE_CASES, ids=lambda c: f"{c['n_layer']}L{c['n_embd']}d")
def test_param_counts_match_instantiated_model(case: dict) -> None:
    model = _build(case)
    counts = count_spikegpt_params(**case)
    assert counts.total == sum(p.numel() for p in model.parameters())
    assert counts.input_embedding == model.embedding.weight.numel()
    assert counts.lm_head == model.head.weight.numel()
    assert counts.ln_out == sum(p.numel() for p in model.ln_out.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    assert counts.block_matmul + counts.block_other == block_params


def test_gpt2_216m_headline_count() -> None:
    counts = count_spikegpt_params(vocab_size=50277, n_layer=18, n_embd=768)
    assert counts.total == 215_399_424  # the repo's "215.4M" gpt2-216m preset
    assert counts.non_vocab == 138_173_952
    # N = 2VC + L(13C^2 + 11C) + 4C, the closed form in the module docstring.
    v, layers, c = 50277, 18, 768
    assert counts.total == 2 * v * c + layers * (13 * c * c + 11 * c) + 4 * c


@pytest.mark.parametrize(
    "case",
    [
        dict(vocab_size=277, n_layer=2, n_embd=64, model_type="rwkv"),
        dict(vocab_size=277, n_layer=3, n_embd=96, model_type="rwkv-ffn-pre"),
    ],
    ids=lambda c: c["model_type"],
)
def test_training_flops_match_flop_counter(case: dict) -> None:
    model = _build(case)
    model.train()
    batch, ctx = 2, 32
    ids = torch.randint(0, case["vocab_size"], (batch, ctx + 1))
    inputs, targets = ids[:, :ctx], ids[:, 1:]
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        loss, _ = model(inputs, targets)
        loss.backward()
    counts = count_spikegpt_params(**case)
    tokens = batch * ctx
    assert flop_counter.get_total_flops() == training_flops(counts, tokens)
    assert training_flops(counts, tokens) == flops_per_token(counts) * tokens
    # Forward is exactly a third of fwd+bwd for bias-free Linear stacks.
    assert flops_per_token(counts, backward=False) * 3 == flops_per_token(counts)


def test_tokens_for_flop_budget_round_trips() -> None:
    counts = count_spikegpt_params(vocab_size=50277, n_layer=12, n_embd=512)
    budget = 3.0e17
    tokens = tokens_for_flop_budget(counts, budget)
    assert training_flops(counts, tokens) <= budget
    assert training_flops(counts, tokens + 1) > budget
