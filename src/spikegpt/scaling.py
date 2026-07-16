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
    # "vanilla" is matmul-parameter-identical but adds a quadratic FLOP term that
    # scales with context, so context_length must be recorded. See flops_per_token.
    attention: str = "rwkv"
    context_length: int | None = None

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
    attention: str = "rwkv",
    context_length: int | None = None,
) -> SpikeGPTParamCounts:
    """Exact parameter counts for a ``SpikeLanguageModel`` of these dimensions.

    ``attention="vanilla"`` swaps the RWKV time-mix for causal softmax attention: both
    are 4*C^2, so matmul parameters are identical and only RWKV's 5*C time-mix vectors
    per layer differ. ``context_length`` is unused for parameters but recorded so
    ``flops_per_token`` can price attention's quadratic term.
    """
    if model_type not in ("rwkv", "rwkv-ffn-pre"):
        raise ValueError("model_type must be 'rwkv' or 'rwkv-ffn-pre'")
    if attention not in ("rwkv", "vanilla"):
        raise ValueError("attention must be 'rwkv' or 'vanilla'")
    c = n_embd
    n_time_mix = n_layer if model_type == "rwkv" else n_layer - 1
    n_channel_mix = n_layer if model_type == "rwkv" else n_layer + 1
    block_matmul = n_time_mix * 4 * c * c + n_channel_mix * 9 * c * c
    # Mixing vectors (time-mix 5C, channel-mix 2C), ln1+ln2 per block, ln0 on layer 0.
    # Vanilla attention carries no time-mix vectors (and RoPE adds no parameters).
    time_mix_vectors = 0 if attention == "vanilla" else n_time_mix * 5 * c
    block_other = time_mix_vectors + n_channel_mix * 2 * c + n_layer * 4 * c + 2 * c
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
        attention=attention,
        context_length=context_length,
    )


def attention_flops_per_token(counts: SpikeGPTParamCounts, *, backward: bool = True) -> int:
    """FLOPs/token from attention's quadratic QK^T and AV products (0 for RWKV).

    Activation-activation matmuls, so they scale with context, not parameters -- the
    ``6*N*D`` approximation misses them. 4*T*C per token per layer forward, ~3x for
    backward. RWKV's recurrence has no such cost, so an isoFLOP comparison ignoring
    this would hand the attention arm ~16-20% free compute at ctx 1024.
    """
    if counts.attention != "vanilla":
        return 0
    if counts.context_length is None:
        raise ValueError("context_length is required to price attention FLOPs")
    n_attn_layers = counts.n_layer if counts.model_type == "rwkv" else counts.n_layer - 1
    return (12 if backward else 4) * n_attn_layers * counts.context_length * counts.n_embd


def flops_per_token(counts: SpikeGPTParamCounts, *, backward: bool = True) -> int:
    """FLOPs per token (2*MAC): weight matmuls, plus attention's quadratic term when
    attention="vanilla". Exactly ``6*N_flop`` for RWKV (its recurrence is elementwise)."""
    return (6 if backward else 2) * counts.flop_matmul + attention_flops_per_token(
        counts, backward=backward
    )


def training_flops(counts: SpikeGPTParamCounts, tokens: int) -> int:
    """Total training compute C for a run of ``tokens`` tokens (fwd+bwd)."""
    return flops_per_token(counts) * tokens


def tokens_for_flop_budget(counts: SpikeGPTParamCounts, flop_budget: float) -> int:
    """The D that spends ``flop_budget`` on this model: D = C / (6 * N_flop)."""
    return int(flop_budget // flops_per_token(counts))
