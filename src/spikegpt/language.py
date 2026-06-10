"""SpikeGPT-style language-model modules.

The modules here intentionally start as ordinary PyTorch code. SpikeGPT's most
important architectural idea is to use the token sequence as the spiking time
axis and replace quadratic self-attention with an RWKV-style recurrent mixer.
Keeping the first version torch-native gives us a correctness target before we
specialize the WKV recurrence or spike operators.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Any, Literal, Protocol, TypeAlias, cast

import torch
from myelin._optional import has_triton
from myelin.autograd import (
    triton_surrogate_lif_function,
    triton_surrogate_lif_recompute_function,
)
from myelin.neurons import LIFParams, LIFState
from myelin.surrogates import (
    SURROGATE_NAMES,
    SurrogateFn,
    SurrogateName,
    atan_surrogate,
    hard_surrogate_spike,
    surrogate_from_name,
)
from myelin.triton.lif import surrogate_id as lif_surrogate_id
from myelin.triton.lif_bf16 import surrogate_lif_bf16io
from torch import nn
from torch._higher_order_ops.associative_scan import associative_scan
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from spikegpt.wkv_bf16 import weighted_key_value_triton_bf16io
from spikegpt.wkv_triton import weighted_key_value_triton

# Reverse lookup (built-in surrogate callable -> name) so SpikingSequenceLIF can
# dispatch to the fused Triton kernels, which take a surrogate name.
_SURROGATE_NAME_BY_FN: dict[SurrogateFn, SurrogateName] = {
    surrogate_from_name(name): name for name in SURROGATE_NAMES
}

SpikeGPTModelType = Literal["rwkv", "rwkv-ffn-pre"]
SpikeGPTPreset = Literal["micro", "tiny", "small", "base", "gpt2-216m"]
SamplingMode = Literal["multinomial", "greedy"]


@dataclass(frozen=True)
class CharacterVocabulary:
    """Stable character-level vocabulary for tiny SpikeGPT-style experiments."""

    tokens: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharacterVocabulary:
        if not text:
            raise ValueError("text must not be empty")
        return cls(tuple(sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str) -> torch.Tensor:
        token_to_id = self.token_to_id
        missing = sorted(set(text) - set(token_to_id))
        if missing:
            raise ValueError(f"text contains out-of-vocabulary characters: {missing}")
        return torch.tensor([token_to_id[token] for token in text], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        return "".join(self.tokens[index] for index in token_ids.tolist())


@dataclass(frozen=True)
class ByteVocabulary:
    """Fixed byte-level vocabulary for arbitrary UTF-8 text corpora."""

    @classmethod
    def from_text(cls, text: str) -> ByteVocabulary:
        if not text:
            raise ValueError("text must not be empty")
        return cls()

    @property
    def size(self) -> int:
        return 256

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            raise ValueError("text must not be empty")
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        values = token_ids.detach().cpu().tolist()
        return bytes(values).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class BPEVocabulary:
    """Subword BPE vocabulary backed by a HuggingFace ``tokenizers`` tokenizer.

    Used for the SpikeGPT 216M setting (GPT-NeoX BPE, ~50k vocab). The full
    ``tokenizer.json`` is stored inline so checkpoints are self-contained — a saved
    model reloads its exact tokenizer with no network access. Requires the
    ``tokenization`` extra (``tokenizers``); the import is lazy so the core package
    does not depend on it.
    """

    tokenizer_json: str
    vocab_size: int
    _tokenizer: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        from tokenizers import Tokenizer

        object.__setattr__(self, "_tokenizer", Tokenizer.from_str(self.tokenizer_json))

    @classmethod
    def from_pretrained(cls, name: str = "EleutherAI/gpt-neox-20b") -> BPEVocabulary:
        """Build from a Hub tokenizer (downloads ``tokenizer.json`` once)."""
        from huggingface_hub import hf_hub_download

        return cls.from_tokenizer_file(hf_hub_download(name, "tokenizer.json"))

    @classmethod
    def from_tokenizer_file(cls, path: str | PathLike[str]) -> BPEVocabulary:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer_json=tokenizer.to_str(), vocab_size=tokenizer.get_vocab_size())

    @property
    def size(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            raise ValueError("text must not be empty")
        return torch.tensor(self._tokenizer.encode(text).ids, dtype=torch.long)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many texts at once (the Rust tokenizer parallelizes across cores)."""
        return [encoding.ids for encoding in self._tokenizer.encode_batch(texts)]

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids.detach().cpu().tolist())


LanguageVocabulary: TypeAlias = CharacterVocabulary | ByteVocabulary | BPEVocabulary


@dataclass(frozen=True)
class LanguageEval:
    """Evaluation metrics for autoregressive character language modeling."""

    loss: float
    bits_per_character: float
    perplexity: float


@dataclass(frozen=True)
class SpikeLanguageCheckpoint:
    """Loaded SpikeGPT-style model checkpoint."""

    model: SpikeLanguageModel
    vocabulary: LanguageVocabulary
    metadata: dict[str, object]
    optimizer_state_dict: dict[str, Any] | None = None


@dataclass(frozen=True)
class SpikeTimeMixState:
    """Cached recurrent state for one ``SpikeTimeMix`` module."""

    previous: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    log_scale: torch.Tensor


@dataclass(frozen=True)
class SpikeChannelMixState:
    """Cached previous-token state for one ``SpikeChannelMix`` module."""

    previous: torch.Tensor


@dataclass(frozen=True)
class SpikingSequenceLIFState:
    """Cached membrane state for one sequence LIF module."""

    membrane: torch.Tensor


@dataclass(frozen=True)
class SpikeGPTBlockState:
    """Cached recurrent state for one ``SpikeGPTBlock``."""

    time_mix: SpikeTimeMixState | None
    ffn_pre: SpikeChannelMixState | None
    channel_mix: SpikeChannelMixState
    lif1: SpikingSequenceLIFState
    lif2: SpikingSequenceLIFState


@dataclass(frozen=True)
class SpikeLanguageModelState:
    """Cached recurrent state for autoregressive SpikeGPT-style inference."""

    blocks: tuple[SpikeGPTBlockState, ...]


@dataclass(frozen=True)
class SpikeGPTConfig:
    """Configuration for a compact SpikeGPT-style recurrent language model."""

    vocab_size: int
    context_length: int
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.03
    model_type: SpikeGPTModelType = "rwkv"
    lif_tau: float = 2.0
    lif_threshold: float = 1.0
    lif_reset: float = 0.0
    surrogate_slope: float = 2.0
    spike_embedding: bool = True
    spiking: bool = True
    spike_input: bool = False
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.model_type not in ("rwkv", "rwkv-ffn-pre"):
            raise ValueError("model_type must be 'rwkv' or 'rwkv-ffn-pre'")
        if self.lif_tau <= 0.0:
            raise ValueError("lif_tau must be positive")


@dataclass(frozen=True)
class SpikeGPTPresetSpec:
    """Named SpikeGPT model dimensions for reproducible experiments."""

    context_length: int
    n_layer: int
    n_embd: int


SPIKEGPT_PRESETS: dict[SpikeGPTPreset, SpikeGPTPresetSpec] = {
    "micro": SpikeGPTPresetSpec(context_length=32, n_layer=1, n_embd=32),
    "tiny": SpikeGPTPresetSpec(context_length=64, n_layer=2, n_embd=64),
    "small": SpikeGPTPresetSpec(context_length=128, n_layer=4, n_embd=128),
    "base": SpikeGPTPresetSpec(context_length=256, n_layer=6, n_embd=256),
    # SpikeGPT paper's 216M at-scale config (18L/768d); ~215M params at vocab 50277.
    "gpt2-216m": SpikeGPTPresetSpec(context_length=1024, n_layer=18, n_embd=768),
}


def spikegpt_config_from_preset(
    preset: SpikeGPTPreset,
    *,
    vocab_size: int,
    dropout: float = 0.03,
    model_type: SpikeGPTModelType = "rwkv",
    lif_tau: float = 2.0,
    lif_threshold: float = 1.0,
    lif_reset: float = 0.0,
    surrogate_slope: float = 2.0,
    spike_embedding: bool = True,
    spiking: bool = True,
    spike_input: bool = False,
    gradient_checkpointing: bool = False,
) -> SpikeGPTConfig:
    """Create a ``SpikeGPTConfig`` from a named model-size preset."""

    spec = SPIKEGPT_PRESETS[preset]
    return SpikeGPTConfig(
        vocab_size=vocab_size,
        context_length=spec.context_length,
        n_layer=spec.n_layer,
        n_embd=spec.n_embd,
        dropout=dropout,
        model_type=model_type,
        lif_tau=lif_tau,
        lif_threshold=lif_threshold,
        lif_reset=lif_reset,
        surrogate_slope=surrogate_slope,
        spike_embedding=spike_embedding,
        spiking=spiking,
        spike_input=spike_input,
        gradient_checkpointing=gradient_checkpointing,
    )


def _time_shift(inputs: torch.Tensor) -> torch.Tensor:
    return F.pad(inputs[:, :-1], (0, 0, 1, 0))


def _mix_with_previous(
    inputs: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    previous = _time_shift(inputs)
    return inputs * mix + previous * (1.0 - mix)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    sampling: SamplingMode,
) -> torch.Tensor:
    next_logits = logits / temperature
    if top_k is not None:
        limit = min(top_k, next_logits.shape[-1])
        values, _indices = torch.topk(next_logits, limit, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        next_logits = torch.where(
            next_logits < threshold,
            torch.full_like(next_logits, -torch.inf),
            next_logits,
        )
    if sampling == "greedy":
        return next_logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _validate_wkv_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
) -> None:
    if key.shape != value.shape:
        msg = f"key and value must have the same shape; got {key.shape} and {value.shape}"
        raise ValueError(msg)
    if key.ndim != 3:
        msg = f"key and value must have shape [B, T, C]; got {key.shape}"
        raise ValueError(msg)
    if time_decay.shape != (key.shape[-1],):
        msg = f"time_decay must have shape [{key.shape[-1]}]; got {time_decay.shape}"
        raise ValueError(msg)
    if time_first.shape != (key.shape[-1],):
        msg = f"time_first must have shape [{key.shape[-1]}]; got {time_first.shape}"
        raise ValueError(msg)


def weighted_key_value_loop(
    key: torch.Tensor,
    value: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
) -> torch.Tensor:
    """Sequential RWKV weighted key-value recurrence (correctness oracle).

    ``key`` and ``value`` are batch-major tensors with shape ``[B, T, C]``. This
    readable per-timestep loop is the reference ``weighted_key_value`` is
    validated against. It is correct but ``torch.compile`` unrolls the time loop,
    so its compile cost grows (super-linearly) with ``T``; prefer the default
    associative-scan ``weighted_key_value`` for compiled training.
    """

    _validate_wkv_inputs(key, value, time_decay, time_first)
    batch, timesteps, channels = key.shape
    dtype = key.dtype
    device = key.device
    decay = -torch.exp(time_decay.to(device=device, dtype=dtype))
    first = time_first.to(device=device, dtype=dtype)
    numerator = torch.zeros((batch, channels), dtype=dtype, device=device)
    denominator = torch.zeros((batch, channels), dtype=dtype, device=device)
    log_scale = torch.full(
        (batch, channels),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    outputs: list[torch.Tensor] = []

    for step in range(timesteps):
        key_t = key[:, step]
        value_t = value[:, step]

        next_log_scale = torch.maximum(log_scale, first + key_t)
        old_scale = torch.exp(log_scale - next_log_scale)
        new_scale = torch.exp(first + key_t - next_log_scale)
        outputs.append(
            (old_scale * numerator + new_scale * value_t) / (old_scale * denominator + new_scale)
        )

        decayed_log_scale = log_scale + decay
        next_log_scale = torch.maximum(decayed_log_scale, key_t)
        old_scale = torch.exp(decayed_log_scale - next_log_scale)
        new_scale = torch.exp(key_t - next_log_scale)
        numerator = old_scale * numerator + new_scale * value_t
        denominator = old_scale * denominator + new_scale
        log_scale = next_log_scale

    return torch.stack(outputs, dim=1)


def weighted_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
) -> torch.Tensor:
    """RWKV weighted key-value recurrence via ``associative_scan`` (default).

    Numerically identical to :func:`weighted_key_value_loop`, but expressed as a
    single ``associative_scan`` higher-order op so ``torch.compile`` sees one
    graph node instead of unrolling the per-timestep loop. The result: compile
    time is roughly flat in ``T`` (vs. super-linear for the loop), peak memory is
    linear in ``T`` (vs. the O(T^2) dense decay-matrix form), and compiled
    runtime is the best of the variants at long context. Requires torch >= 2.13
    for correct ``associative_scan`` autograd. See
    ``benchmarks/results/spikegpt_wkv_compare_5090.md``.

    Each token is summarized as a log-space monoid element
    ``(acc_decay, log_scale, num, den)`` whose true accumulator is
    ``exp(log_scale) * num``; the associative ``combine`` decays the earlier
    segment by the later segment's total decay before merging, with a shared
    max-exponent for numerical stability.
    """

    _validate_wkv_inputs(key, value, time_decay, time_first)
    batch, timesteps, channels = key.shape
    # The log-space recurrence (exp of accumulated decay) is numerically
    # sensitive, so it always runs in float32 — matching the reference SpikeGPT
    # WKV CUDA kernel, which upcasts w/u/k/v to float32 internally regardless of
    # the model's autocast dtype. The result is cast back to the caller's dtype
    # so the surrounding bf16 matmuls are unaffected. No-op in a pure-fp32 run.
    orig_dtype = key.dtype
    # Upcast only reduced-precision floats (bf16/fp16); preserve float32/float64.
    dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    device = key.device
    decay = -torch.exp(time_decay.to(device=device, dtype=dtype))  # [C], < 0
    first = time_first.to(device=device, dtype=dtype)

    keys = key.movedim(1, 0).contiguous().to(dtype)  # [T, B, C]
    values = value.movedim(1, 0).contiguous().to(dtype)
    # Per-token element: acc_decay=decay (len 1), log_scale=key (true num=exp(k)*v), num=v, den=1.
    acc_decay = decay[None, None, :].expand(timesteps, batch, channels).contiguous()
    ones = torch.ones_like(keys)

    def combine(left, right):
        decay_l, scale_l, num_l, den_l = left
        decay_r, scale_r, num_r, den_r = right
        # The earlier (left) segment decays by exp(decay_r) before merging.
        merged = torch.maximum(scale_l + decay_r, scale_r)
        weight_l = torch.exp(scale_l + decay_r - merged)
        weight_r = torch.exp(scale_r - merged)
        return (
            decay_l + decay_r,
            merged,
            weight_l * num_l + weight_r * num_r,
            weight_l * den_l + weight_r * den_r,
        )

    _, scale_inc, num_inc, den_inc = associative_scan(
        combine, (acc_decay, keys, values, ones), dim=0, combine_mode="generic"
    )

    # Exclusive prefix: the accumulator strictly before each token (shift by one).
    neg_inf = torch.full_like(scale_inc[:1], torch.finfo(dtype).min)
    zeros = torch.zeros_like(num_inc[:1])
    scale_ex = torch.cat([neg_inf, scale_inc[:-1]], dim=0)
    num_ex = torch.cat([zeros, num_inc[:-1]], dim=0)
    den_ex = torch.cat([zeros, den_inc[:-1]], dim=0)

    # Combine the exclusive prefix with the current-token ``time_first`` bonus.
    bonus = first[None, None, :] + keys
    merged = torch.maximum(scale_ex, bonus)
    weight_prefix = torch.exp(scale_ex - merged)
    weight_bonus = torch.exp(bonus - merged)
    out = (weight_prefix * num_ex + weight_bonus * values) / (weight_prefix * den_ex + weight_bonus)
    return out.movedim(0, 1).to(orig_dtype)


def split_token_sequence(
    tokens: torch.Tensor,
    *,
    validation_fraction: float,
    min_validation_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a one-dimensional token sequence into train and validation tails."""

    if tokens.ndim != 1:
        raise ValueError(f"tokens must be one-dimensional; got {tokens.shape}")
    if tokens.numel() < 3:
        raise ValueError("tokens must contain at least 3 elements")
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if min_validation_tokens < 0:
        raise ValueError("min_validation_tokens must be non-negative")

    validation_count = max(min_validation_tokens, int(tokens.numel() * validation_fraction))
    validation_count = min(validation_count, tokens.numel() - 2)
    if validation_count == 0:
        return tokens, tokens
    return tokens[:-validation_count], tokens[-validation_count:]


def split_train_val_test(
    tokens: torch.Tensor,
    *,
    validation_fraction: float,
    min_validation_tokens: int,
    test_fraction: float = 0.0,
    test_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three-way tail split: train (head) / val (middle) / test (last tail).

    The **test** tail is held out *first*, so it is never trained on nor used for
    checkpoint selection; the remainder is then split into train (head) and val
    (tail) by :func:`split_token_sequence`. This is the clean enwik8 protocol
    (e.g. 90M / 5M / 5M) where the reported test number comes from data the model
    neither trained on nor was selected against.

    ``test_tokens`` overrides ``test_fraction`` when positive. With no test
    holdout the behaviour matches :func:`split_token_sequence` and the returned
    ``test`` tensor is empty.
    """
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be one-dimensional; got {tokens.shape}")
    if test_fraction < 0.0 or test_fraction >= 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if test_tokens < 0:
        raise ValueError("test_tokens must be non-negative")

    test_count = test_tokens if test_tokens > 0 else int(tokens.numel() * test_fraction)
    # Leave at least 3 tokens for the train/val split of the remainder.
    test_count = max(0, min(test_count, tokens.numel() - 3))
    if test_count == 0:
        train_tokens, val_tokens = split_token_sequence(
            tokens,
            validation_fraction=validation_fraction,
            min_validation_tokens=min_validation_tokens,
        )
        return train_tokens, val_tokens, tokens[:0]

    test_split = tokens[-test_count:]
    remainder = tokens[:-test_count]
    train_tokens, val_tokens = split_token_sequence(
        remainder,
        validation_fraction=validation_fraction,
        min_validation_tokens=min_validation_tokens,
    )
    return train_tokens, val_tokens, test_split


class TokenSource(Protocol):
    """Anything that can serve token windows by slicing (a ``torch.Tensor`` or a
    memory-mapped corpus). ``__getitem__`` of an int slice returns a 1-D long
    tensor; ``numel`` is the total token count."""

    def numel(self) -> int: ...

    def __getitem__(self, index: slice) -> torch.Tensor: ...


def sample_token_batch(
    tokens: torch.Tensor | TokenSource,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random next-token prediction windows from a token sequence.

    ``tokens`` may be an in-memory ``torch.Tensor`` or a memmap-backed
    :class:`~spikegpt.token_corpus.MemmapTokenCorpus` (anything matching
    :class:`TokenSource`), so the same IID random-window sampler drives both the
    small enwik8 runs and OpenWebText2-scale corpora.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    max_start = tokens.numel() - context_length - 1
    if max_start <= 0:
        raise ValueError("tokens are too short for the requested context length")
    starts = torch.randint(0, max_start, (batch_size,)).tolist()
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return inputs.to(device=device), targets.to(device=device)


@torch.no_grad()
def evaluate_language_model(
    model: nn.Module,
    tokens: torch.Tensor | TokenSource,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    batches: int,
) -> LanguageEval:
    """Evaluate random language-model windows and report loss, BPC, and perplexity."""

    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    try:
        for _ in range(batches):
            inputs, targets = sample_token_batch(
                tokens,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("evaluate_language_model expected logits-only model output")
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            total_loss += float(loss)
            total_tokens += targets.numel()
    finally:
        if was_training:
            model.train()
    mean_loss = total_loss / total_tokens
    return LanguageEval(
        loss=mean_loss,
        bits_per_character=mean_loss / 0.6931471805599453,
        perplexity=float(torch.exp(torch.tensor(mean_loss))),
    )


@torch.no_grad()
def evaluate_language_model_strided(
    model: nn.Module,
    tokens: torch.Tensor | TokenSource,
    *,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    stride: int | None = None,
    count_last: int | None = None,
) -> LanguageEval:
    """Evaluate deterministic next-token windows over a corpus.

    By default every position of each window contributes to the loss. That
    *inflates* BPC, because the first positions of each window are scored with
    far less than ``context_length`` tokens of preceding context. For a faithful
    full-context BPC (the metric the SpikeGPT/enwik8 literature reports), pass
    ``count_last`` together with ``stride == count_last``: only the last
    ``count_last`` targets of each window are scored (each with at least
    ``context_length - count_last`` tokens of context), and the first window is
    scored in full (its prefix is unavoidable). Counted targets then tile the
    corpus exactly once.
    """

    if isinstance(tokens, torch.Tensor) and tokens.ndim != 1:
        raise ValueError(f"tokens must be one-dimensional; got {tokens.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if count_last is not None and not 0 < count_last <= context_length:
        raise ValueError("count_last must be in (0, context_length]")
    resolved_stride = context_length if stride is None else stride
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")
    starts = tuple(range(0, tokens.numel() - context_length, resolved_stride))
    if not starts:
        raise ValueError("tokens are too short for the requested context length")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    reduction = "sum" if count_last is None else "none"
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    try:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            inputs = torch.stack(
                [tokens[start : start + context_length] for start in batch_starts]
            ).to(device=device)
            targets = torch.stack(
                [tokens[start + 1 : start + context_length + 1] for start in batch_starts]
            ).to(device=device)
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("evaluate_language_model_strided expected logits-only output")
            if count_last is None:
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                total_loss += float(loss)
                total_tokens += targets.numel()
            else:
                # per-position losses; score only the last ``count_last`` targets
                # of each window (the whole first window, whose prefix is forced).
                losses = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).reshape(
                    len(batch_starts), context_length
                )
                for row, start in enumerate(batch_starts):
                    lo = 0 if start == 0 else context_length - count_last
                    total_loss += float(losses[row, lo:].sum())
                    total_tokens += context_length - lo
    finally:
        if was_training:
            model.train()
    mean_loss = total_loss / total_tokens
    return LanguageEval(
        loss=mean_loss,
        bits_per_character=mean_loss / 0.6931471805599453,
        perplexity=float(torch.exp(torch.tensor(mean_loss))),
    )


class SpikingSequenceLIF(nn.Module):
    """LIF unroll over a batch-major token sequence ``[B, T, C]``."""

    def __init__(
        self,
        *,
        tau: float = 2.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 2.0,
        hard_forward: bool = True,
        fused: bool = True,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.surrogate = surrogate
        self.surrogate_slope = surrogate_slope
        self.hard_forward = hard_forward
        # When `fused`, use the Triton fused-time kernel on CUDA (store-all backward
        # by default; `recompute` trades a backward recompute for lower peak memory).
        # Falls back to the per-step loop on CPU / without Triton / for custom
        # surrogates. The loop stays the correctness oracle.
        self.fused = fused
        self.recompute = recompute

    @property
    def decay(self) -> float:
        return 1.0 - (1.0 / self.tau)

    def initial_state(
        self,
        *,
        batch_size: int,
        features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikingSequenceLIFState:
        return SpikingSequenceLIFState(
            membrane=torch.zeros((batch_size, features), device=device, dtype=dtype)
        )

    def step(
        self,
        inputs: torch.Tensor,
        state: SpikingSequenceLIFState,
    ) -> tuple[torch.Tensor, SpikingSequenceLIFState]:
        if inputs.ndim != 2:
            msg = f"inputs must have shape [B, C]; got {inputs.shape}"
            raise ValueError(msg)
        if state.membrane.shape != inputs.shape:
            msg = (
                "state.membrane and inputs must have the same shape; "
                f"got {state.membrane.shape} and {inputs.shape}"
            )
            raise ValueError(msg)
        # Membrane update uses myelin's standard LIF convention `v = decay*v + x`
        # (decay = 1 - 1/tau), consistent with `neurons.py` and the Triton kernels.
        # The SpikeGPT/SpikingJelly reference instead uses the Euler form
        # `v = decay*v + (1-decay)*x` (decay_input=True); the two differ only by a
        # constant 1/tau gain on the input, which is fully absorbed by the learned
        # Linear feeding this activation, so a trained model is equivalent either way.
        membrane = state.membrane * self.decay + inputs
        centered = self.surrogate_slope * (membrane - self.threshold)
        if self.hard_forward:
            spike = hard_surrogate_spike(centered, self.surrogate)
        else:
            spike = self.surrogate(centered)
        membrane = membrane * (1.0 - spike) + self.reset * spike
        return spike, SpikingSequenceLIFState(membrane=membrane)

    def _fused_surrogate_name(self) -> SurrogateName | None:
        """Surrogate name for the Triton kernel, or None if the fused path can't apply."""
        if not (self.fused and self.hard_forward):
            return None
        return _SURROGATE_NAME_BY_FN.get(self.surrogate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            msg = f"inputs must have shape [B, T, C]; got {inputs.shape}"
            raise ValueError(msg)

        surrogate_name = self._fused_surrogate_name()
        params = LIFParams(tau_mem=self.tau, threshold=self.threshold, reset=self.reset)

        # Fast path: reduced-precision (bf16/fp16) store-all on CUDA uses the fused
        # bf16-I/O kernel — bf16 inputs/spikes with an fp32 membrane carry (~1.5x the
        # LIF op vs upcasting; convergence-validated to match the fp32 path).
        if (
            surrogate_name is not None
            and inputs.is_cuda
            and has_triton()
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and not self.recompute
        ):
            currents = inputs.movedim(1, 0).contiguous()  # [T, B, C]
            initial = LIFState(membrane=currents.new_zeros(currents.shape[1:]))
            spikes = surrogate_lif_bf16io(
                currents,
                initial,
                params,
                surrogate_slope=self.surrogate_slope,
                surrogate_id=lif_surrogate_id(surrogate_name),
            )
            return spikes.movedim(0, 1)

        # Otherwise keep the membrane recurrence in fp32 by upcasting bf16/fp16
        # inputs (no-op for a pure-fp32 run); covers fp32, the recompute variant,
        # and the CPU loop fallback below.
        orig_dtype = inputs.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            inputs = inputs.float()

        # Fused-time Triton kernel on CUDA: the time loop lives in the kernel, so it
        # needs no torch.compile and does not unroll. Falls back to the loop below.
        if surrogate_name is not None and inputs.is_cuda and has_triton():
            currents = inputs.movedim(1, 0).contiguous()  # [T, B, C]
            initial = LIFState(membrane=currents.new_zeros(currents.shape[1:]))
            run = (
                triton_surrogate_lif_recompute_function
                if self.recompute
                else triton_surrogate_lif_function
            )
            _, spikes = run(
                currents,
                initial,
                params,
                surrogate=surrogate_name,
                surrogate_slope=self.surrogate_slope,
                hard_forward=True,
            )
            return spikes.movedim(0, 1).to(orig_dtype)

        state = self.initial_state(
            batch_size=inputs.shape[0],
            features=inputs.shape[2],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        spikes_list: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            spike, state = self.step(inputs[:, step], state)
            spikes_list.append(spike)
        return torch.stack(spikes_list, dim=1).to(orig_dtype)


class SpikeTimeMix(nn.Module):
    """RWKV time-mixing block followed by a spiking residual activation."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
    ) -> None:
        super().__init__()
        if n_embd <= 0:
            raise ValueError("n_embd must be positive")
        attn_size = n_embd
        layer_denominator = max(1, n_layer - 1)
        ratio_0_to_1 = layer_id / layer_denominator
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

        position = torch.arange(n_embd, dtype=torch.float32) / n_embd
        decay_speed = -5.0 + 8.0 * torch.pow(
            torch.arange(attn_size, dtype=torch.float32) / max(1, attn_size - 1),
            0.7 + 1.3 * ratio_0_to_1,
        )
        zigzag = torch.tensor([(index + 1) % 3 - 1 for index in range(attn_size)])
        self.time_decay = nn.Parameter(decay_speed)
        self.time_first = nn.Parameter(torch.ones(attn_size) * -1.2039728 + zigzag * 0.5)
        self.time_mix_k = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_v = nn.Parameter(
            (torch.pow(position, ratio_1_to_almost0) + 0.3 * ratio_0_to_1).view(1, 1, -1)
        )
        self.time_mix_r = nn.Parameter(torch.pow(position, 0.5 * ratio_1_to_almost0).view(1, 1, -1))

        self.key = nn.Linear(n_embd, attn_size, bias=False)
        self.value = nn.Linear(n_embd, attn_size, bias=False)
        self.receptance = nn.Linear(n_embd, attn_size, bias=False)
        self.output = nn.Linear(attn_size, n_embd, bias=False)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikeTimeMixState:
        features = self.time_decay.numel()
        return SpikeTimeMixState(
            previous=torch.zeros((batch_size, features), device=device, dtype=dtype),
            numerator=torch.zeros((batch_size, features), device=device, dtype=dtype),
            denominator=torch.zeros((batch_size, features), device=device, dtype=dtype),
            log_scale=torch.full(
                (batch_size, features),
                torch.finfo(dtype).min,
                device=device,
                dtype=dtype,
            ),
        )

    def step(
        self,
        inputs: torch.Tensor,
        state: SpikeTimeMixState,
    ) -> tuple[torch.Tensor, SpikeTimeMixState]:
        if inputs.ndim != 2:
            msg = f"inputs must have shape [B, C]; got {inputs.shape}"
            raise ValueError(msg)
        previous = state.previous
        key_input = inputs * self.time_mix_k[0, 0] + previous * (1.0 - self.time_mix_k[0, 0])
        value_input = inputs * self.time_mix_v[0, 0] + previous * (1.0 - self.time_mix_v[0, 0])
        receptance_input = inputs * self.time_mix_r[0, 0] + previous * (1.0 - self.time_mix_r[0, 0])
        key = self.key(key_input)
        value = self.value(value_input)
        receptance = torch.sigmoid(self.receptance(receptance_input))

        dtype = key.dtype
        decay = -torch.exp(self.time_decay.to(device=key.device, dtype=dtype))
        first = self.time_first.to(device=key.device, dtype=dtype)
        next_log_scale = torch.maximum(state.log_scale, first + key)
        old_scale = torch.exp(state.log_scale - next_log_scale)
        new_scale = torch.exp(first + key - next_log_scale)
        mixed = (old_scale * state.numerator + new_scale * value) / (
            old_scale * state.denominator + new_scale
        )

        decayed_log_scale = state.log_scale + decay
        recurrent_log_scale = torch.maximum(decayed_log_scale, key)
        old_scale = torch.exp(decayed_log_scale - recurrent_log_scale)
        new_scale = torch.exp(key - recurrent_log_scale)
        next_state = SpikeTimeMixState(
            previous=inputs,
            numerator=old_scale * state.numerator + new_scale * value,
            denominator=old_scale * state.denominator + new_scale,
            log_scale=recurrent_log_scale,
        )
        return self.output(receptance * mixed), next_state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key_input = _mix_with_previous(inputs, self.time_mix_k)
        value_input = _mix_with_previous(inputs, self.time_mix_v)
        receptance_input = _mix_with_previous(inputs, self.time_mix_r)

        key = self.key(key_input)
        value = self.value(value_input)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        # Fused Triton WKV on CUDA: ~10% faster step / ~25% less memory than the
        # associative_scan path with exact gradients (parity vs the loop oracle to
        # fp32). Under bf16/fp16 autocast the bf16-I/O kernel is bit-identical but
        # ~1.5x faster (skips the fp32 materialization, halves input bandwidth).
        # Falls back to associative_scan on CPU or without Triton.
        if key.is_cuda and has_triton():
            if key.dtype in (torch.bfloat16, torch.float16):
                wkv = weighted_key_value_triton_bf16io(key, value, self.time_decay, self.time_first)
            else:
                wkv = weighted_key_value_triton(key, value, self.time_decay, self.time_first)
        else:
            wkv = weighted_key_value(key, value, self.time_decay, self.time_first)
        mixed = receptance * wkv
        return self.output(mixed)


class SpikeChannelMix(nn.Module):
    """RWKV channel-mixing feed-forward block."""

    def __init__(self, n_embd: int, n_layer: int, layer_id: int) -> None:
        super().__init__()
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
        position = torch.arange(n_embd, dtype=torch.float32) / n_embd
        hidden_size = 4 * n_embd
        self.time_mix_k = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_r = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.key = nn.Linear(n_embd, hidden_size, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_size, n_embd, bias=False)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikeChannelMixState:
        return SpikeChannelMixState(
            previous=torch.zeros(
                (batch_size, self.receptance.out_features),
                device=device,
                dtype=dtype,
            )
        )

    def step(
        self,
        inputs: torch.Tensor,
        state: SpikeChannelMixState,
    ) -> tuple[torch.Tensor, SpikeChannelMixState]:
        if inputs.ndim != 2:
            msg = f"inputs must have shape [B, C]; got {inputs.shape}"
            raise ValueError(msg)
        previous = state.previous
        key_input = inputs * self.time_mix_k[0, 0] + previous * (1.0 - self.time_mix_k[0, 0])
        receptance_input = inputs * self.time_mix_r[0, 0] + previous * (1.0 - self.time_mix_r[0, 0])
        key = torch.relu(self.key(key_input)).square()
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        return receptance * value, SpikeChannelMixState(previous=inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key_input = _mix_with_previous(inputs, self.time_mix_k)
        receptance_input = _mix_with_previous(inputs, self.time_mix_r)
        key = torch.relu(self.key(key_input)).square()
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        return receptance * value


class _IdentityLIF(nn.Module):
    """Continuous passthrough with the SpikingSequenceLIF interface (config.spiking=False).

    The continuous-twin ablation: outputs its input unchanged (no binarization, no
    membrane), so each sublayer's contribution joins the residual in full precision
    (= vanilla RWKV-v4). Implements initial_state/step so the recurrent generation
    path also works on a continuous model (state is a carried no-op)."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def initial_state(
        self, *, batch_size: int, features: int, device: torch.device, dtype: torch.dtype
    ) -> SpikingSequenceLIFState:
        return SpikingSequenceLIFState(
            membrane=torch.zeros((batch_size, features), device=device, dtype=dtype)
        )

    def step(
        self, inputs: torch.Tensor, state: SpikingSequenceLIFState
    ) -> tuple[torch.Tensor, SpikingSequenceLIFState]:
        return inputs, state


class SpikeGPTBlock(nn.Module):
    """SpikeGPT-style block with RWKV time/channel mixing and LIF activations."""

    def __init__(self, config: SpikeGPTConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        else:
            self.ln0 = None
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if layer_id == 0 and config.model_type == "rwkv-ffn-pre":
            self.ffn_pre = SpikeChannelMix(config.n_embd, config.n_layer, 0)
            self.att = None
        else:
            self.ffn_pre = None
            self.att = SpikeTimeMix(config.n_embd, config.n_layer, layer_id)
        self.ffn = SpikeChannelMix(config.n_embd, config.n_layer, layer_id)

        # spiking=False is the continuous "standard decoder" twin (ablation): the LIF
        # binarization gates become a passthrough, so each sublayer's output joins the
        # residual in full precision (= vanilla RWKV-v4). Same params (LIF has none).
        def _make_lif() -> SpikingSequenceLIF | _IdentityLIF:
            if not config.spiking:
                return _IdentityLIF()
            return SpikingSequenceLIF(
                tau=config.lif_tau,
                threshold=config.lif_threshold,
                reset=config.lif_reset,
                surrogate_slope=config.surrogate_slope,
            )

        self.lif1 = _make_lif()
        self.lif2 = _make_lif()
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikeGPTBlockState:
        return SpikeGPTBlockState(
            time_mix=None
            if self.att is None
            else self.att.initial_state(batch_size=batch_size, device=device, dtype=dtype),
            ffn_pre=None
            if self.ffn_pre is None
            else self.ffn_pre.initial_state(batch_size=batch_size, device=device, dtype=dtype),
            channel_mix=self.ffn.initial_state(batch_size=batch_size, device=device, dtype=dtype),
            lif1=self.lif1.initial_state(
                batch_size=batch_size,
                features=self.config.n_embd,
                device=device,
                dtype=dtype,
            ),
            lif2=self.lif2.initial_state(
                batch_size=batch_size,
                features=self.config.n_embd,
                device=device,
                dtype=dtype,
            ),
        )

    def step(
        self,
        inputs: torch.Tensor,
        state: SpikeGPTBlockState,
    ) -> tuple[torch.Tensor, SpikeGPTBlockState]:
        if inputs.ndim != 2:
            msg = f"inputs must have shape [B, C]; got {inputs.shape}"
            raise ValueError(msg)
        if self.config.spike_input:
            # The recurrent generation path for the LIF->Linear variant isn't wired
            # yet; the training/eval path (forward) is. Add it before generating.
            raise NotImplementedError("spike_input does not support the recurrent step() path yet")
        residual = self.ln0(inputs) if self.ln0 is not None else inputs
        if self.ffn_pre is not None:
            if state.ffn_pre is None:
                raise RuntimeError("SpikeGPTBlockState missing ffn_pre state")
            mixed, ffn_pre_state = self.ffn_pre.step(self.ln1(residual), state.ffn_pre)
            time_mix_state = None
        else:
            if self.att is None or state.time_mix is None:
                raise RuntimeError("SpikeGPTBlockState missing time_mix state")
            mixed, time_mix_state = self.att.step(self.ln1(residual), state.time_mix)
            ffn_pre_state = None
        spike1, lif1_state = self.lif1.step(mixed, state.lif1)
        residual = residual + spike1
        channel_mixed, channel_state = self.ffn.step(self.ln2(residual), state.channel_mix)
        spike2, lif2_state = self.lif2.step(channel_mixed, state.lif2)
        residual = self.dropout(residual + spike2)
        return residual, SpikeGPTBlockState(
            time_mix=time_mix_state,
            ffn_pre=ffn_pre_state,
            channel_mix=channel_state,
            lif1=lif1_state,
            lif2=lif2_state,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.ln0(inputs) if self.ln0 is not None else inputs
        # spike_input=True is the "hardware-faithful" LIF->Linear variant: the LIF
        # spikes each sub-block's INPUT, so the projections consume spikes (the
        # AC-capable placement the paper's energy table assumes) instead of spiking
        # the output (Linear->LIF, the canonical default). See spikegpt_216m_energy.md.
        spike_in = self.config.spike_input
        if self.ffn_pre is not None:
            normed = self.ln1(residual)
            residual = residual + (
                self.ffn_pre(self.lif1(normed)) if spike_in else self.lif1(self.ffn_pre(normed))
            )
        else:
            if self.att is None:
                raise RuntimeError("SpikeGPTBlock has no time-mix module")
            normed = self.ln1(residual)
            residual = residual + (
                self.att(self.lif1(normed)) if spike_in else self.lif1(self.att(normed))
            )
        normed2 = self.ln2(residual)
        residual = residual + (
            self.ffn(self.lif2(normed2)) if spike_in else self.lif2(self.ffn(normed2))
        )
        return self.dropout(residual)


class SpikeLanguageModel(nn.Module):
    """SpikeGPT-like autoregressive language model."""

    def __init__(self, config: SpikeGPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [SpikeGPTBlock(config, layer_id) for layer_id in range(config.n_layer)]
        )
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if not self.config.spike_embedding:
            return embeddings
        return hard_surrogate_spike(
            self.config.surrogate_slope * embeddings,
            atan_surrogate,
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable or disable block-level activation checkpointing for training."""

        self.gradient_checkpointing = enabled

    def _run_block(self, block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return cast(torch.Tensor, checkpoint(block, hidden, use_reentrant=False))
        return cast(torch.Tensor, block(hidden))

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> SpikeLanguageModelState:
        parameter = self.embedding.weight
        resolved_device = parameter.device if device is None else device
        resolved_dtype = parameter.dtype if dtype is None else dtype
        return SpikeLanguageModelState(
            blocks=tuple(
                cast(SpikeGPTBlock, block).initial_state(
                    batch_size=batch_size,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
                for block in self.blocks
            )
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: SpikeLanguageModelState,
    ) -> tuple[torch.Tensor, SpikeLanguageModelState]:
        """Run one autoregressive token step with cached recurrent state."""

        if input_ids.ndim != 1:
            msg = f"input_ids must have shape [B]; got {input_ids.shape}"
            raise ValueError(msg)
        hidden = self.embed_tokens(input_ids)
        next_block_states: list[SpikeGPTBlockState] = []
        for raw_block, block_state in zip(self.blocks, state.blocks, strict=True):
            block = cast(SpikeGPTBlock, raw_block)
            hidden, next_block_state = block.step(hidden, block_state)
            next_block_states.append(next_block_state)
        logits = self.head(self.ln_out(hidden))
        return logits, SpikeLanguageModelState(blocks=tuple(next_block_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            msg = f"input_ids must have shape [B, T]; got {input_ids.shape}"
            raise ValueError(msg)
        if input_ids.shape[1] > self.config.context_length:
            msg = (
                "input sequence length exceeds context_length; "
                f"got {input_ids.shape[1]} and {self.config.context_length}"
            )
            raise ValueError(msg)

        hidden = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden = self._run_block(block, hidden)

        if targets is None:
            return self.head(self.ln_out(hidden))
        return self.loss_tail(hidden, targets)

    def loss_tail(
        self, hidden: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Final ``ln_out -> head -> cross_entropy`` from the block output.

        Split out so it can be ``torch.compile``d as its own region (fusing the
        large vocab head + cross-entropy), which is most of the win of a
        full-model compile without the whole-model graph-break fragility.
        """
        logits = self.head(self.ln_out(hidden))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: SamplingMode = "multinomial",
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids``.

        ``use_cache=True`` uses the recurrent RWKV/LIF state path. The fallback
        path recomputes the context window and is kept as a simple correctness
        reference.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if sampling not in ("multinomial", "greedy"):
            raise ValueError("sampling must be 'multinomial' or 'greedy'")
        if input_ids.ndim != 2:
            msg = f"input_ids must have shape [B, T]; got {input_ids.shape}"
            raise ValueError(msg)
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one context token")

        was_training = self.training
        self.eval()
        if use_cache:
            output = input_ids
            state = self.initial_state(
                batch_size=input_ids.shape[0],
                device=input_ids.device,
                dtype=self.embedding.weight.dtype,
            )
            logits = None
            for step in range(input_ids.shape[1]):
                logits, state = self.forward_step(input_ids[:, step], state)
            for _ in range(max_new_tokens):
                if logits is None:
                    raise RuntimeError("generate requires at least one context token")
                next_token = _sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    sampling=sampling,
                )
                output = torch.cat((output, next_token), dim=1)
                logits, state = self.forward_step(next_token[:, 0], state)
            if was_training:
                self.train()
            return output

        output = input_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.config.context_length :]
            logits = self(context)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("generate expected logits-only forward output")
            next_token = _sample_next_token(
                logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                sampling=sampling,
            )
            output = torch.cat((output, next_token), dim=1)
        if was_training:
            self.train()
        return output

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: SamplingMode = "multinomial",
    ) -> Iterator[torch.Tensor]:
        """Yield generated tokens one at a time via the recurrent cached path.

        Yields one ``[B, 1]`` token tensor per step (``max_new_tokens`` total),
        sampled exactly as :meth:`generate` with ``use_cache=True``. ``no_grad`` is
        applied inside the body (a ``@torch.no_grad()`` decorator would not cover a
        generator's lazily-executed yields).
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if sampling not in ("multinomial", "greedy"):
            raise ValueError("sampling must be 'multinomial' or 'greedy'")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T]; got {input_ids.shape}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one context token")

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                state = self.initial_state(
                    batch_size=input_ids.shape[0],
                    device=input_ids.device,
                    dtype=self.embedding.weight.dtype,
                )
                logits: torch.Tensor | None = None
                for step in range(input_ids.shape[1]):
                    logits, state = self.forward_step(input_ids[:, step], state)
                for _ in range(max_new_tokens):
                    if logits is None:
                        raise RuntimeError("generate_stream requires at least one context token")
                    next_token = _sample_next_token(
                        logits, temperature=temperature, top_k=top_k, sampling=sampling
                    )
                    yield next_token
                    logits, state = self.forward_step(next_token[:, 0], state)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def spike_rates(self, input_ids: torch.Tensor) -> dict[str, float]:
        """Return coarse spike-rate diagnostics for embeddings and block activations."""

        rates: dict[str, float] = {}
        hidden = self.embed_tokens(input_ids)
        rates["embedding"] = float(hidden.mean())
        for index, raw_block in enumerate(self.blocks):
            block = cast(SpikeGPTBlock, raw_block)
            att_input = block.ln1(block.ln0(hidden) if block.ln0 is not None else hidden)
            if block.ffn_pre is not None:
                att_spikes = block.lif1(block.ffn_pre(att_input))
            else:
                if block.att is None:
                    raise RuntimeError("SpikeGPTBlock has no time-mix module")
                att_spikes = block.lif1(block.att(att_input))
            ffn_spikes = block.lif2(block.ffn(block.ln2(hidden + att_spikes)))
            rates[f"blocks.{index}.time"] = float(att_spikes.mean())
            rates[f"blocks.{index}.channel"] = float(ffn_spikes.mean())
            hidden = block(hidden)
        return rates


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def spikegpt_config_to_dict(config: SpikeGPTConfig) -> dict[str, object]:
    """Serialize a ``SpikeGPTConfig`` to a plain checkpoint-safe dictionary."""

    return cast(dict[str, object], asdict(config))


def spikegpt_config_from_dict(data: Mapping[str, object]) -> SpikeGPTConfig:
    """Deserialize a ``SpikeGPTConfig`` from a checkpoint dictionary."""

    config = SpikeGPTConfig(
        vocab_size=_require_int(data.get("vocab_size"), "config.vocab_size"),
        context_length=_require_int(data.get("context_length"), "config.context_length"),
        n_layer=_require_int(data.get("n_layer", 4), "config.n_layer"),
        n_embd=_require_int(data.get("n_embd", 128), "config.n_embd"),
        dropout=_require_float(data.get("dropout", 0.03), "config.dropout"),
        model_type=cast(
            SpikeGPTModelType,
            _require_str(data.get("model_type", "rwkv"), "config.model_type"),
        ),
        lif_tau=_require_float(data.get("lif_tau", 2.0), "config.lif_tau"),
        lif_threshold=_require_float(data.get("lif_threshold", 1.0), "config.lif_threshold"),
        lif_reset=_require_float(data.get("lif_reset", 0.0), "config.lif_reset"),
        surrogate_slope=_require_float(data.get("surrogate_slope", 2.0), "config.surrogate_slope"),
        spike_embedding=_require_bool(data.get("spike_embedding", True), "config.spike_embedding"),
        spiking=_require_bool(data.get("spiking", True), "config.spiking"),
        spike_input=_require_bool(data.get("spike_input", False), "config.spike_input"),
        gradient_checkpointing=_require_bool(
            data.get("gradient_checkpointing", False),
            "config.gradient_checkpointing",
        ),
    )
    config.validate()
    return config


def language_vocabulary_to_dict(vocabulary: LanguageVocabulary) -> dict[str, object]:
    """Serialize a SpikeGPT vocabulary to a plain checkpoint-safe dictionary."""

    if isinstance(vocabulary, ByteVocabulary):
        return {"type": "byte"}
    if isinstance(vocabulary, BPEVocabulary):
        return {
            "type": "bpe",
            "tokenizer_json": vocabulary.tokenizer_json,
            "vocab_size": vocabulary.vocab_size,
        }
    return {"type": "character", "tokens": list(vocabulary.tokens)}


def language_vocabulary_from_dict(data: Mapping[str, object]) -> LanguageVocabulary:
    """Deserialize a SpikeGPT vocabulary from a checkpoint dictionary."""

    vocabulary_type = _require_str(data.get("type"), "vocabulary.type")
    if vocabulary_type == "byte":
        return ByteVocabulary()
    if vocabulary_type == "bpe":
        return BPEVocabulary(
            tokenizer_json=_require_str(data.get("tokenizer_json"), "vocabulary.tokenizer_json"),
            vocab_size=_require_int(data.get("vocab_size"), "vocabulary.vocab_size"),
        )
    if vocabulary_type != "character":
        raise ValueError(f"unsupported vocabulary type: {vocabulary_type}")
    raw_tokens = data.get("tokens")
    if not isinstance(raw_tokens, list) or not all(isinstance(item, str) for item in raw_tokens):
        raise ValueError("character vocabulary tokens must be a list of strings")
    if not raw_tokens:
        raise ValueError("character vocabulary tokens must not be empty")
    return CharacterVocabulary(tokens=tuple(raw_tokens))


def save_spike_language_checkpoint(
    path: str | PathLike[str],
    model: SpikeLanguageModel,
    vocabulary: LanguageVocabulary,
    *,
    metadata: Mapping[str, object] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a SpikeGPT-style model, config, vocabulary, and metadata."""

    payload = {
        "format": "myelin.spike_language_checkpoint.v1",
        "config": spikegpt_config_to_dict(model.config),
        "vocabulary": language_vocabulary_to_dict(vocabulary),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_spike_language_checkpoint(
    path: str | PathLike[str],
    *,
    map_location: str | torch.device | None = None,
) -> SpikeLanguageCheckpoint:
    """Load a SpikeGPT-style model checkpoint saved by ``save_spike_language_checkpoint``."""

    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("format") != "myelin.spike_language_checkpoint.v1":
        raise ValueError("unsupported SpikeLanguageModel checkpoint format")
    config = spikegpt_config_from_dict(_require_mapping(payload.get("config"), "config"))
    vocabulary = language_vocabulary_from_dict(
        _require_mapping(payload.get("vocabulary"), "vocabulary")
    )
    raw_state_dict = payload.get("state_dict")
    if not isinstance(raw_state_dict, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    # Checkpoints saved from a regionally-compiled model carry torch.compile's
    # `_orig_mod.` wrapper segment on each compiled submodule's keys
    # (e.g. `blocks.0._orig_mod.att.key.weight`). Strip it so the state_dict
    # loads into a plain, uncompiled SpikeLanguageModel.
    state_dict = {key.replace("._orig_mod.", "."): value for key, value in raw_state_dict.items()}
    model = SpikeLanguageModel(config)
    model.load_state_dict(cast(Mapping[str, Any], state_dict))
    metadata = dict(_require_mapping(payload.get("metadata", {}), "metadata"))
    raw_optimizer_state = payload.get("optimizer_state_dict")
    optimizer_state_dict = (
        None
        if raw_optimizer_state is None
        else dict(_require_mapping(raw_optimizer_state, "optimizer_state_dict"))
    )
    return SpikeLanguageCheckpoint(
        model=model,
        vocabulary=vocabulary,
        metadata=metadata,
        optimizer_state_dict=optimizer_state_dict,
    )
