"""Spike-statistics collection and reporting for SpikeGPT models.

This characterizes the *spiking nature* of a trained ``SpikeLanguageModel``:
which populations spike, how densely, where (per layer / per token position), and
whether neurons fall into undesirable states (dead = never fire, saturated = fire
almost always).

The only binary-spiking activations are the two ``SpikingSequenceLIF`` modules in
each block (``lif1`` = the "time"/attention population, ``lif2`` = the
"channel"/FFN population) plus the spike embedding when
``config.spike_embedding`` is set. Everything else (WKV time-mixing, the relu^2
FFN, layernorms, the head) is continuous.

Statistics are gathered by registering forward hooks on the real LIF modules and
running the model over strided corpus windows, so the captured spikes are exactly
what the model computes (including the fused Triton path) -- unlike
``SpikeLanguageModel.spike_rates``, which re-runs a partial forward. Only ``[C]``
and ``[T]`` accumulators are retained (never the raw ``[B, T, C]`` activations), so
memory is bounded regardless of corpus length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from spikegpt.language import SpikeLanguageModel, SpikingSequenceLIF

_LN2 = 0.6931471805599453


@dataclass(frozen=True)
class PopulationSpikeStats:
    """Spike statistics for one spiking population (embedding / lif1 / lif2)."""

    name: str
    num_channels: int
    total_steps: int  # B*T summed over all windows
    density: float  # mean firing probability per neuron per step
    per_neuron_rate: Tensor  # [C] firing rate of each neuron over the corpus
    per_position_rate: Tensor  # [T] firing rate at each within-window position
    dead_count: int
    dead_fraction: float
    saturated_count: int
    saturated_fraction: float


@dataclass(frozen=True)
class SpikeStatistics:
    """Spike statistics across every spiking population of a model."""

    populations: dict[str, PopulationSpikeStats]
    context_length: int
    windows_processed: int
    tokens_processed: int
    spike_embedding: bool
    dead_threshold: float
    saturated_threshold: float

    def overall_density(self) -> float:
        """Element-weighted mean firing rate across all spiking populations."""
        spikes = sum(p.density * p.num_channels * p.total_steps for p in self.populations.values())
        elems = sum(p.num_channels * p.total_steps for p in self.populations.values())
        return spikes / elems if elems else 0.0

    def per_layer_profile(self) -> list[tuple[int, float, float]]:
        """``(layer, time_density, channel_density)`` for each block, in order."""
        profile: list[tuple[int, float, float]] = []
        index = 0
        while True:
            time = self.populations.get(f"blocks.{index}.time")
            channel = self.populations.get(f"blocks.{index}.channel")
            if time is None or channel is None:
                break
            profile.append((index, time.density, channel.density))
            index += 1
        return profile


@dataclass(frozen=True)
class SpikingFractionReport:
    """How much of the model is "spiking", measured three different ways."""

    total_params: int
    spiking_params: int
    param_fraction_spiking: float
    n_layer: int
    spike_embedding: bool
    spike_activation_sites: int  # 2 per block (+1 for the embedding spike)
    residual_writes_spiking: int  # residual-stream writes that are spike tensors
    total_residual_writes: int
    flop_fraction_spiking_estimate: float
    continuous_flops_estimate: int
    spiking_flops_estimate: int


class _Accumulator:
    """Running spike sums for one population, kept on CPU in float64."""

    def __init__(self) -> None:
        self.spike_sum: float = 0.0
        self.elem_count: int = 0
        self.bt_steps: int = 0  # B*T summed over windows (per-neuron denominator)
        self.bc_per_position: int = 0  # B*C summed over windows (per-position denom)
        self.neuron_sum: Tensor | None = None  # [C]
        self.position_sum: Tensor | None = None  # [T]

    def update(self, spikes: Tensor) -> None:
        # Spikes may be bf16/fp16 (still exactly 0/1); upcast before summing.
        values = spikes.detach().float()
        batch, time, channels = values.shape
        self.spike_sum += float(values.sum())
        self.elem_count += batch * time * channels
        self.bt_steps += batch * time
        self.bc_per_position += batch * channels
        neuron = values.sum(dim=(0, 1)).double().cpu()  # [C]
        position = values.sum(dim=(0, 2)).double().cpu()  # [T]
        self.neuron_sum = neuron if self.neuron_sum is None else self.neuron_sum + neuron
        self.position_sum = position if self.position_sum is None else self.position_sum + position


def _finalize(
    acc: _Accumulator, name: str, dead_threshold: float, saturated_threshold: float
) -> PopulationSpikeStats:
    assert acc.neuron_sum is not None and acc.position_sum is not None
    per_neuron = (acc.neuron_sum / acc.bt_steps).float()
    per_position = (acc.position_sum / acc.bc_per_position).float()
    channels = per_neuron.numel()
    dead = int((per_neuron <= dead_threshold).sum())
    saturated = int((per_neuron >= saturated_threshold).sum())
    return PopulationSpikeStats(
        name=name,
        num_channels=channels,
        total_steps=acc.bt_steps,
        density=acc.spike_sum / acc.elem_count if acc.elem_count else 0.0,
        per_neuron_rate=per_neuron,
        per_position_rate=per_position,
        dead_count=dead,
        dead_fraction=dead / channels if channels else 0.0,
        saturated_count=saturated,
        saturated_fraction=saturated / channels if channels else 0.0,
    )


@torch.no_grad()
def collect_spike_statistics(
    model: SpikeLanguageModel,
    tokens: Tensor,
    *,
    context_length: int | None = None,
    stride: int | None = None,
    batch_size: int = 8,
    max_windows: int | None = None,
    device: torch.device | str | None = None,
    dead_threshold: float = 0.0,
    saturated_threshold: float = 0.99,
) -> SpikeStatistics:
    """Collect per-population spike statistics over strided windows of ``tokens``.

    ``tokens`` is a 1-D token-id tensor. Windows of ``context_length`` are taken
    every ``stride`` tokens (non-overlapping by default). Forward hooks on the
    ``SpikingSequenceLIF`` modules capture the real spike tensors; the embedding
    spike (when enabled) is captured via ``model.embed_tokens``.
    """
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be one-dimensional; got {tuple(tokens.shape)}")
    ctx = context_length if context_length is not None else model.config.context_length
    step = stride if stride is not None else ctx
    if step <= 0:
        raise ValueError("stride must be positive")
    if tokens.numel() < ctx:
        raise ValueError(f"corpus ({tokens.numel()}) shorter than context_length ({ctx})")
    resolved_device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    starts = list(range(0, tokens.numel() - ctx + 1, step))
    if max_windows is not None:
        starts = starts[:max_windows]

    was_training = model.training
    model.eval()
    spike_embedding = model.config.spike_embedding
    embedding_acc = _Accumulator() if spike_embedding else None
    lif_accs: dict[str, _Accumulator] = {}
    handles = []

    def make_hook(acc: _Accumulator):
        def hook(_module: object, _inputs: object, output: Tensor) -> None:
            acc.update(output)

        return hook

    for index, block in enumerate(model.blocks):
        for module, name in (
            (block.lif1, f"blocks.{index}.time"),
            (block.lif2, f"blocks.{index}.channel"),
        ):
            acc = _Accumulator()
            lif_accs[name] = acc
            assert isinstance(module, SpikingSequenceLIF)
            handles.append(module.register_forward_hook(make_hook(acc)))

    try:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = torch.stack([tokens[s : s + ctx] for s in batch_starts]).to(
                device=resolved_device
            )
            if embedding_acc is not None:
                embedding_acc.update(model.embed_tokens(batch))
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    populations: dict[str, PopulationSpikeStats] = {}
    if embedding_acc is not None:
        populations["embedding"] = _finalize(
            embedding_acc, "embedding", dead_threshold, saturated_threshold
        )
    for index in range(len(model.blocks)):
        for name in (f"blocks.{index}.time", f"blocks.{index}.channel"):
            populations[name] = _finalize(lif_accs[name], name, dead_threshold, saturated_threshold)

    return SpikeStatistics(
        populations=populations,
        context_length=ctx,
        windows_processed=len(starts),
        tokens_processed=len(starts) * ctx,
        spike_embedding=spike_embedding,
        dead_threshold=dead_threshold,
        saturated_threshold=saturated_threshold,
    )


def compute_spiking_fraction(model: SpikeLanguageModel) -> SpikingFractionReport:
    """Static accounting of how much of the model is spiking (params/writes/FLOPs).

    The three lenses disagree by design: by parameters the model is ~0% spiking
    (LIF holds no weights), by residual-stream writes it is ~100% spiking (every
    block writes its output to the residual through a spike tensor), and by FLOPs
    it is mostly continuous (the spike is a cheap elementwise step on top of dense
    Linear / WKV / FFN compute). FLOP figures are order-of-magnitude estimates.
    """
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    spiking_params = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, SpikingSequenceLIF)
        for p in module.parameters(recurse=False)
    )
    n_layer = config.n_layer
    embd = config.n_embd
    spike_activation_sites = 2 * n_layer + (1 if config.spike_embedding else 0)
    residual_writes_spiking = 2 * n_layer + (1 if config.spike_embedding else 0)

    # Per-token MAC estimate. Continuous: WKV projections (~4*embd^2), the relu^2
    # channel-mix FFN (~9*embd^2 with the 4x hidden), and the output head
    # (embd*vocab). Spiking: the LIF membrane recurrence is a handful of
    # elementwise ops per neuron (~6*embd per LIF), plus the embedding surrogate.
    continuous = n_layer * (4 * embd * embd + 9 * embd * embd) + embd * config.vocab_size
    spiking = n_layer * 2 * 6 * embd + (embd if config.spike_embedding else 0)
    flop_fraction = spiking / (spiking + continuous) if (spiking + continuous) else 0.0

    return SpikingFractionReport(
        total_params=total_params,
        spiking_params=spiking_params,
        param_fraction_spiking=spiking_params / total_params if total_params else 0.0,
        n_layer=n_layer,
        spike_embedding=config.spike_embedding,
        spike_activation_sites=spike_activation_sites,
        residual_writes_spiking=residual_writes_spiking,
        total_residual_writes=residual_writes_spiking,
        flop_fraction_spiking_estimate=flop_fraction,
        continuous_flops_estimate=continuous,
        spiking_flops_estimate=spiking,
    )


def _bar(value: float, max_value: float, width: int = 32) -> str:
    filled = 0 if max_value <= 0 else round(width * value / max_value)
    return "█" * filled + "·" * (width - filled)


def _downsample(values: Tensor, buckets: int) -> list[tuple[int, float]]:
    """Mean within each of ``buckets`` contiguous segments → (start_index, mean)."""
    length = values.numel()
    buckets = min(buckets, length)
    edges = torch.linspace(0, length, buckets + 1).round().long().tolist()
    out: list[tuple[int, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi > lo:
            out.append((lo, float(values[lo:hi].mean())))
    return out


def _histogram(rates: Tensor, bins: int = 20) -> str:
    counts = torch.histc(rates.clamp(0.0, 1.0), bins=bins, min=0.0, max=1.0).long().tolist()
    peak = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        lo, hi = i / bins, (i + 1) / bins
        lines.append(f"  [{lo:.2f},{hi:.2f}) {_bar(count, peak, 28)} {count}")
    return "\n".join(lines)


def format_spike_report(
    stats: SpikeStatistics,
    fraction: SpikingFractionReport,
    *,
    title: str,
    meta_lines: list[str],
    position_buckets: int = 24,
    histogram_bins: int = 20,
) -> str:
    """Render a Markdown spike-analysis report."""
    out: list[str] = [f"# {title}", ""]
    out += meta_lines + [""]

    out += [
        "## 1. How much of the model is spiking?",
        "",
        "| Lens | Spiking | Total | Fraction |",
        "|---|---:|---:|---:|",
        f"| Trainable params | {fraction.spiking_params:,} | {fraction.total_params:,} | "
        f"{fraction.param_fraction_spiking * 100:.3f}% |",
        f"| Residual-stream writes | {fraction.residual_writes_spiking} | "
        f"{fraction.total_residual_writes} | "
        f"{fraction.residual_writes_spiking / max(fraction.total_residual_writes, 1) * 100:.0f}% |",
        f"| Compute (FLOPs, approx) | {fraction.spiking_flops_estimate:,} | "
        f"{fraction.spiking_flops_estimate + fraction.continuous_flops_estimate:,} | "
        f"{fraction.flop_fraction_spiking_estimate * 100:.3f}% |",
        "",
        f"LIF activations hold **no parameters** (≈0% by weights) yet **every one of "
        f"the {fraction.residual_writes_spiking} residual-stream writes is a binary "
        f"spike tensor** ({2}/block"
        + (" + 1 embedding" if fraction.spike_embedding else "")
        + "). The dense Linear / WKV / FFN compute that *produces* those spikes "
        "dominates FLOPs, so spiking is a property of the information flow, not the "
        "weights or the arithmetic.",
        "",
    ]

    out += [
        "## 2. Spike density",
        "",
        f"Overall firing rate (all spiking populations): **{stats.overall_density() * 100:.2f}%** "
        f"of neuron-steps. Windows={stats.windows_processed}, "
        f"tokens={stats.tokens_processed:,}, ctx={stats.context_length}.",
        "",
        "| Population | Channels | Density | Dead | Saturated |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, pop in stats.populations.items():
        out.append(
            f"| {name} | {pop.num_channels} | {pop.density * 100:.2f}% | "
            f"{pop.dead_count} ({pop.dead_fraction * 100:.1f}%) | "
            f"{pop.saturated_count} ({pop.saturated_fraction * 100:.1f}%) |"
        )
    out.append("")

    out += ["## 3. Per-layer profile (firing density by depth)", ""]
    profile = stats.per_layer_profile()
    if profile:
        peak = max(max(t, c) for _, t, c in profile) or 1.0
        out += ["| Layer | time (lif1) | channel (lif2) |", "|---:|---|---|"]
        for layer, time_d, channel_d in profile:
            out.append(
                f"| {layer} | `{_bar(time_d, peak, 20)}` {time_d * 100:.1f}% | "
                f"`{_bar(channel_d, peak, 20)}` {channel_d * 100:.1f}% |"
            )
    out.append("")

    out += [
        "## 4. Per-position profile (firing vs. token position in window)",
        "",
        "Window-aligned (position 0 = first token of each window). Mean over the "
        "embedding + all block populations, bucketed:",
        "",
        "```",
    ]
    # Mean per-position across populations, weighted by channels.
    total_channels = sum(p.num_channels for p in stats.populations.values())
    if total_channels:
        weighted = torch.stack(
            [p.per_position_rate * p.num_channels for p in stats.populations.values()]
        )
        mean_position = weighted.sum(dim=0) / total_channels
        peak = float(mean_position.max()) or 1.0
        for start, value in _downsample(mean_position, position_buckets):
            out.append(f"  pos {start:>5} {_bar(value, peak, 36)} {value * 100:.1f}%")
    out += ["```", ""]

    out += ["## 5. Dead / saturated neurons", ""]
    out.append(
        f"Dead = per-neuron firing rate ≤ {stats.dead_threshold}; "
        f"saturated = rate ≥ {stats.saturated_threshold}. "
        "Per-neuron firing-rate histograms:"
    )
    out.append("")
    for name, pop in stats.populations.items():
        out += [
            f"**{name}** — dead {pop.dead_count}/{pop.num_channels} "
            f"({pop.dead_fraction * 100:.1f}%), saturated {pop.saturated_count}/"
            f"{pop.num_channels} ({pop.saturated_fraction * 100:.1f}%)",
            "```",
            _histogram(pop.per_neuron_rate, histogram_bins),
            "```",
            "",
        ]

    out += [
        "## 6. Recommendations (loss terms for undesirable states)",
        "",
        "`myelin.losses.spike_rate_loss` / `SpikeRateLoss` exist but are **not** "
        "wired into language-model training, and they target the **global mean** "
        "firing rate — they cannot fix per-neuron dead/saturation (a population can "
        "hit a 10% mean with half its neurons dead and half saturated). The "
        "targeted fix is a **per-neuron homeostatic** penalty "
        "`((spikes.mean(dim=(0,1)) - target)**2).mean()` that pushes each channel's "
        "own rate toward a target, optionally with an EMA for a stabler estimate. "
        "Deferred until the analysis shows it is needed.",
        "",
    ]
    return "\n".join(out)
