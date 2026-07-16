"""Exactness tests for spikegpt.scaling: closed-form counts vs ``sum(p.numel())``
and 6N FLOPs vs ``FlopCounterMode`` on a real fwd+bwd."""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from spikegpt import SpikeGPTConfig, SpikeLanguageModel
from spikegpt.scaling import (
    attention_flops_per_token,
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


def test_vanilla_attention_is_parameter_matched_to_rwkv() -> None:
    """The mixers differ only by RWKV's 5*C time-mix vectors per layer -- the parity
    that makes an rwkv-vs-vanilla comparison controlled."""
    kwargs = dict(vocab_size=50277, n_layer=12, n_embd=512)
    rwkv = count_spikegpt_params(**kwargs, attention="rwkv")
    vanilla = count_spikegpt_params(**kwargs, attention="vanilla", context_length=1024)
    assert rwkv.block_matmul == vanilla.block_matmul
    assert rwkv.flop_matmul == vanilla.flop_matmul
    assert rwkv.total - vanilla.total == 12 * 5 * 512


def test_vanilla_attention_adds_quadratic_flops() -> None:
    """Attention's QK^T/AV term must be priced (12*L*T*C/token fwd+bwd) or the isoFLOP
    comparison hands the attention arm ~16-20% free compute."""
    counts = count_spikegpt_params(
        vocab_size=50277, n_layer=12, n_embd=512, attention="vanilla", context_length=1024
    )
    expected_attn = 12 * 12 * 1024 * 512
    assert attention_flops_per_token(counts) == expected_attn
    assert flops_per_token(counts) == 6 * counts.flop_matmul + expected_attn
    # forward-only is a third of fwd+bwd
    assert attention_flops_per_token(counts, backward=False) == 4 * 12 * 1024 * 512

    rwkv = count_spikegpt_params(vocab_size=50277, n_layer=12, n_embd=512)
    assert attention_flops_per_token(rwkv) == 0
    assert flops_per_token(rwkv) == 6 * rwkv.flop_matmul  # unchanged for RWKV


def test_vanilla_attention_flops_need_context_length() -> None:
    """Must fail loudly: silently defaulting would rig the isoFLOP fit."""
    counts = count_spikegpt_params(
        vocab_size=50277, n_layer=12, n_embd=512, attention="vanilla", context_length=None
    )
    with pytest.raises(ValueError, match="context_length is required"):
        flops_per_token(counts)


def test_vanilla_attention_param_count_matches_model() -> None:
    """Closed-form counts equal the instantiated vanilla model."""
    config = SpikeGPTConfig(
        vocab_size=256, context_length=32, n_layer=3, n_embd=64, attention="vanilla", n_head=4
    )
    model = SpikeLanguageModel(config)
    counts = count_spikegpt_params(
        vocab_size=256, n_layer=3, n_embd=64, attention="vanilla", context_length=32
    )
    assert counts.total == sum(p.numel() for p in model.parameters())
