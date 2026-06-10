"""Tests for spike-statistics collection over a SpikeGPT model."""

from __future__ import annotations

import math
from typing import cast

import torch

from spikegpt import (
    SpikeGPTConfig,
    SpikeLanguageModel,
    SpikingSequenceLIF,
    collect_spike_statistics,
    compute_spiking_fraction,
)


def _tiny_model(*, spike_embedding: bool = True, n_layer: int = 3, n_embd: int = 64):
    config = SpikeGPTConfig(
        vocab_size=256,
        context_length=32,
        n_layer=n_layer,
        n_embd=n_embd,
        dropout=0.0,
        spike_embedding=spike_embedding,
    )
    torch.manual_seed(0)
    return SpikeLanguageModel(config).eval()


def test_collect_spike_statistics_shapes_and_ranges() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 256, (32 * 10,))
    stats = collect_spike_statistics(model, tokens, batch_size=4)

    expected = {"embedding"}
    for i in range(model.config.n_layer):
        expected |= {f"blocks.{i}.time", f"blocks.{i}.channel"}
    assert set(stats.populations) == expected

    for pop in stats.populations.values():
        assert pop.num_channels == model.config.n_embd
        assert pop.per_neuron_rate.numel() == model.config.n_embd
        assert pop.per_position_rate.numel() == model.config.context_length
        assert 0.0 <= pop.density <= 1.0
        assert float(pop.per_neuron_rate.min()) >= 0.0
        assert float(pop.per_neuron_rate.max()) <= 1.0
        assert 0.0 <= pop.per_position_rate.min() <= pop.per_position_rate.max() <= 1.0

    assert len(stats.per_layer_profile()) == model.config.n_layer
    assert 0.0 <= stats.overall_density() <= 1.0


def test_collector_density_matches_spike_rates() -> None:
    # The collector hooks the real forward; spike_rates re-runs a partial forward.
    # On a single window they must agree exactly (the collector is the truth).
    model = _tiny_model()
    window = torch.randint(0, 256, (32,))
    rates = model.spike_rates(window.unsqueeze(0))
    stats = collect_spike_statistics(model, window, batch_size=1)
    for key, pop in stats.populations.items():
        assert math.isclose(pop.density, rates[key], abs_tol=1e-5)


def test_dead_and_saturated_counts_are_consistent() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 256, (32 * 8,))
    # Untrained Linear init keeps currents tiny, so LIF neurons never cross the
    # threshold: every lif neuron is "dead". The embedding (random sign) fires.
    stats = collect_spike_statistics(model, tokens, batch_size=4, saturated_threshold=0.5)
    time_pop = stats.populations["blocks.0.time"]
    assert time_pop.density == 0.0
    assert time_pop.dead_count == time_pop.num_channels
    assert time_pop.dead_fraction == 1.0

    for pop in stats.populations.values():
        assert pop.dead_count == int((pop.per_neuron_rate <= stats.dead_threshold).sum())
        assert pop.saturated_count == int((pop.per_neuron_rate >= stats.saturated_threshold).sum())


def test_hooks_are_removed_after_collection() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 256, (32 * 4,))
    collect_spike_statistics(model, tokens, batch_size=2)
    for block in model.blocks:
        assert len(cast(SpikingSequenceLIF, block.lif1)._forward_hooks) == 0
        assert len(cast(SpikingSequenceLIF, block.lif2)._forward_hooks) == 0


def test_no_embedding_population_when_spike_embedding_disabled() -> None:
    model = _tiny_model(spike_embedding=False)
    tokens = torch.randint(0, 256, (32 * 4,))
    stats = collect_spike_statistics(model, tokens, batch_size=2)
    assert "embedding" not in stats.populations
    assert not stats.spike_embedding


def test_compute_spiking_fraction_lenses() -> None:
    model = _tiny_model()
    fraction = compute_spiking_fraction(model)
    # LIF holds no parameters.
    assert fraction.spiking_params == 0
    assert fraction.param_fraction_spiking == 0.0
    # Every residual write is a spike: 2 per block + 1 embedding.
    assert fraction.residual_writes_spiking == 2 * model.config.n_layer + 1
    assert fraction.residual_writes_spiking == fraction.total_residual_writes
    # Compute is dominated by the continuous projections.
    assert 0.0 < fraction.flop_fraction_spiking_estimate < 0.1
