"""Chinchilla-style parameter and FLOP accounting for SpikeGPT models.

Closed-form counts derived from ``spikegpt.language``, verified exactly
against instantiated models in ``tests/test_scaling.py`` (C = n_embd,
L = n_layer, V = vocab_size):

* regular block: ``13*C^2 + 11*C`` — time-mix ``4*C^2 + 5*C``, channel-mix
  ``9*C^2 + 2*C``, ``ln1``+``ln2`` ``4*C``; LIF neurons carry no parameters.
  Layer 0 adds ``2*C`` (``ln0``); ``model_type="rwkv-ffn-pre"`` swaps its
  time-mix for a second channel-mix.
* outside blocks: ``V*C`` embedding + ``V*C`` untied head + ``2*C`` ``ln_out``.

Two different "N" matter for a scaling-law fit:

* ``non_vocab`` — vocabulary-independent capacity. Fit L(N, D) on this; at
  small widths the two ``V*C`` vocab matrices otherwise dominate N.
* ``flop_matmul`` — matmul-participating parameters (blocks + head; the
  embedding lookup is a gather, 0 FLOPs). This is the N in C = 6*N*D.

FLOPs use the 2*MAC matmul-only convention, matching ``FlopCounterMode`` as
used by ``spikegpt.benchmarks.spikegpt_mfu`` — the WKV recurrence and LIF
gates are elementwise and contribute none: ``2 * flop_matmul`` per token
forward, ``6 *`` forward+backward (the "6N" in C = 6*N*D).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpikeGPTParamCounts:
    """Exact SpikeGPT parameter counts, split by role in the scaling law."""

    vocab_size: int
    n_layer: int
    n_embd: int
    model_type: str
    input_embedding: int
    lm_head: int
    block_matmul: int
    block_other: int
    ln_out: int

    @property
    def total(self) -> int:
        """Every trainable parameter (matches ``sum(p.numel())``)."""
        return (
            self.input_embedding + self.lm_head + self.block_matmul + self.block_other + self.ln_out
        )

    @property
    def non_vocab(self) -> int:
        """Vocabulary-independent capacity — the N to fit L(N, D) on."""
        return self.block_matmul + self.block_other + self.ln_out

    @property
    def flop_matmul(self) -> int:
        """Matmul-participating parameters — the N in C = 6*N*D."""
        return self.block_matmul + self.lm_head


def count_spikegpt_params(
    *,
    vocab_size: int,
    n_layer: int,
    n_embd: int,
    model_type: str = "rwkv",
) -> SpikeGPTParamCounts:
    """Exact parameter counts for a ``SpikeLanguageModel`` of these dimensions."""
    if model_type not in ("rwkv", "rwkv-ffn-pre"):
        raise ValueError("model_type must be 'rwkv' or 'rwkv-ffn-pre'")
    c = n_embd
    n_time_mix = n_layer if model_type == "rwkv" else n_layer - 1
    n_channel_mix = n_layer if model_type == "rwkv" else n_layer + 1
    block_matmul = n_time_mix * 4 * c * c + n_channel_mix * 9 * c * c
    # Mixing vectors (time-mix 5C, channel-mix 2C), ln1+ln2 per block, ln0 on layer 0.
    block_other = n_time_mix * 5 * c + n_channel_mix * 2 * c + n_layer * 4 * c + 2 * c
    return SpikeGPTParamCounts(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        model_type=model_type,
        input_embedding=vocab_size * c,
        lm_head=vocab_size * c,
        block_matmul=block_matmul,
        block_other=block_other,
        ln_out=2 * c,
    )


def flops_per_token(counts: SpikeGPTParamCounts, *, backward: bool = True) -> int:
    """Matmul FLOPs per token (2*MAC), forward or forward+backward."""
    return (6 if backward else 2) * counts.flop_matmul


def training_flops(counts: SpikeGPTParamCounts, tokens: int) -> int:
    """Total training compute C for a run of ``tokens`` tokens (fwd+bwd)."""
    return flops_per_token(counts) * tokens


def tokens_for_flop_budget(counts: SpikeGPTParamCounts, flop_budget: float) -> int:
    """The D that spends ``flop_budget`` on this model: D = C / (6 * N_flop)."""
    return int(flop_budget // flops_per_token(counts))
