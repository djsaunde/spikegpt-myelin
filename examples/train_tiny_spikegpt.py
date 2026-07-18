"""Train a tiny SpikeGPT-style character language model."""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
from example_utils import (
    add_compile_policy_arg,
    add_grad_clip_arg,
    add_matmul_precision_arg,
    add_wandb_args,
    clip_gradients,
    compile_training_model,
    configure_matmul_precision,
    finish_wandb,
    init_wandb,
    log_wandb,
    print_model_summary,
    print_step_time_summary,
    resolve_compile_policy,
)

from spikegpt import (
    SPIKEGPT_PRESETS,
    BPEVocabulary,
    ByteVocabulary,
    CharacterVocabulary,
    LanguageVocabulary,
    MemmapTokenCorpus,
    SpikeGPTConfig,
    SpikeLanguageModel,
    TokenArrayView,
    evaluate_language_model,
    evaluate_language_model_strided,
    load_spike_language_checkpoint,
    sample_token_batch,
    save_spike_language_checkpoint,
    spikegpt_config_from_preset,
    split_train_val_test,
)
from spikegpt.scaling import count_spikegpt_params, flops_per_token

DEFAULT_TEXT = (
    "spiking neural networks trade dense activations for sparse events. "
    "myelin explores fast training paths for those event driven models. "
)


def vocabulary_name(vocabulary: LanguageVocabulary) -> str:
    if isinstance(vocabulary, ByteVocabulary):
        return "byte"
    if isinstance(vocabulary, BPEVocabulary):
        return "bpe"
    return "char"


def metadata_nonnegative_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def compile_spikegpt_regions(
    model: SpikeLanguageModel,
    *,
    fullgraph: bool,
    options: dict[str, object] | None = None,
    mode: str | None = None,
) -> SpikeLanguageModel:
    """Compile repeated SpikeGPT blocks while keeping the top-level loop eager.

    ``mode`` (e.g. ``max-autotune-no-cudagraphs``) and ``options`` are mutually
    exclusive in ``torch.compile``; pass at most one.
    """

    for index, block in enumerate(model.blocks):
        model.blocks[index] = torch.compile(
            block,
            fullgraph=fullgraph,
            options=options,  # type: ignore[arg-type]
            mode=mode,
        )
    return model


REGIONAL_LITE_COMPILE_OPTIONS: dict[str, object] = {
    "max_autotune": False,
    "max_autotune_gemm": False,
    "max_autotune_pointwise": False,
    "triton.autotune_at_compile_time": False,
    "triton.autotune_cublasLt": False,
    "triton.cudagraphs": False,
    "triton.cudagraph_trees": False,
}


def cosine_lr(
    step: int, total_steps: int, lr_init: float, lr_final: float, warmup_steps: int
) -> float:
    """Linear warmup then cosine decay from lr_init to lr_final over total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr_init * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))


def wsd_lr(
    step: int,
    total_steps: int,
    lr_init: float,
    lr_final: float,
    warmup_steps: int,
    decay_steps: int,
    decay_shape: str = "cosine",
) -> float:
    """Warmup-Stable-Decay LR: linear warmup, constant ``lr_init``, then a final
    ``decay_steps`` window annealing to ``lr_final``.

    The stable phase is length-agnostic, so a constant-LR run (``decay_steps=0``)
    can be checkpointed and "decay-branched" later: resume from a stable
    checkpoint with ``decay_steps == total_steps - previous_steps`` and the whole
    sub-run becomes the decay, annealing to ``lr_final`` without committing the
    main run to a fixed length.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return lr_init * (step + 1) / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return lr_init
    progress = min(max((step - decay_start) / max(1, decay_steps), 0.0), 1.0)
    if decay_shape == "linear":
        return lr_init + (lr_final - lr_init) * progress
    if decay_shape == "sqrt":  # MiniCPM-style 1-sqrt
        return lr_init - (lr_init - lr_final) * math.sqrt(progress)
    return lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))


def mark_compiled_invocation_boundary(enabled: bool) -> None:
    if not enabled:
        return
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--train-bin",
        type=Path,
        help="pre-tokenized uint16 corpus (memmap) for train; bypasses text read + split. "
        "Use with --vocab bpe/byte and a corpus from examples/prepare_token_corpus.py.",
    )
    parser.add_argument("--val-bin", type=Path, help="pre-tokenized val corpus (memmap)")
    parser.add_argument("--test-bin", type=Path, help="pre-tokenized test corpus (memmap)")
    parser.add_argument(
        "--val-holdout-tokens",
        type=int,
        default=0,
        help="with --train-bin, hold out this many tokens from the END of the train "
        "corpus as an in-domain validation slice (measures the true generalization "
        "gap). Overrides --val-bin. 0 disables.",
    )
    parser.add_argument(
        "--vocab",
        choices=("char", "byte", "bpe"),
        default="char",
        help="tokenization mode; byte=256-token UTF-8, bpe=subword (--bpe-tokenizer)",
    )
    parser.add_argument(
        "--bpe-tokenizer",
        default="EleutherAI/gpt-neox-20b",
        help="HuggingFace tokenizer for --vocab bpe (downloaded once, stored in the checkpoint)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-tokens", type=int, default=64)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help=(
            "hold out this fraction of the corpus as a test tail FIRST (never "
            "trained on nor used for checkpoint selection); the remainder is then "
            "split into train/val. Use with --test-tokens for the clean 3-way "
            "enwik8 split (e.g. 90M/5M/5M). 0 keeps the legacy train/val-only split "
            "where val IS the held-out tail."
        ),
    )
    parser.add_argument(
        "--test-tokens",
        type=int,
        default=0,
        help="explicit test-tail token count; overrides --test-fraction when > 0",
    )
    parser.add_argument(
        "--preset",
        choices=("custom", *SPIKEGPT_PRESETS.keys()),
        default="custom",
        help="named SpikeGPT size preset; custom uses --context-length/--layers/--embedding",
    )
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument(
        "--model-type",
        choices=("rwkv", "rwkv-ffn-pre"),
        default="rwkv",
        help="SpikeGPT block variant for fresh runs; checkpoints keep their saved model type",
    )
    parser.add_argument(
        "--attention",
        choices=("rwkv", "vanilla"),
        default="rwkv",
        help="token mixer: 'rwkv' (linear WKV recurrence) or 'vanilla' (quadratic softmax "
        "attention + RoPE, ~GPT-2). Parameter-matched; vanilla costs extra context-scaled "
        "FLOPs (priced by spikegpt.scaling). Independent of --no-spiking.",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=8,
        help="attention heads for --attention vanilla; must divide --embedding",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="split each --batch into this many micro-batches, accumulating gradients "
        "before the optimizer step. --batch stays the EFFECTIVE batch, so the step/token "
        "budget and any batch-dependent LR calibration are unchanged; only peak "
        "activation memory shrinks (it tracks the micro-batch). Must divide --batch.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="final LR for the decay schedule over the run; unset keeps a fixed LR",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup steps")
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "wsd"),
        default="cosine",
        help=(
            "LR decay schedule (when --lr-final is set). cosine: warmup->cosine over the "
            "whole run. wsd: warmup->constant->decay over the last --decay-steps; with "
            "--decay-steps 0 it is constant-after-warmup (decay-branch from a checkpoint by "
            "resuming with --decay-steps == remaining steps)."
        ),
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=0,
        help="wsd: length of the final decay window (0 = no decay, pure stable phase)",
    )
    parser.add_argument(
        "--decay-shape",
        choices=("cosine", "linear", "sqrt"),
        default="cosine",
        help="wsd decay curve shape",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save --checkpoint-out every N steps (0 = only at the end)",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--lif-threshold",
        type=float,
        default=1.0,
        help=(
            "LIF firing threshold; 1.0 matches the SpikeGPT/SpikingJelly reference "
            "(v_threshold=1.0). Lower values (e.g. 0.0) fire denser spikes."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--val-eval",
        choices=("strided", "random"),
        default="strided",
        help=(
            "in-loop validation metric driving the log and best-checkpoint "
            "selection. 'strided' (default) is the deterministic full-context BPC "
            "over the whole val set (no sampling noise; matches the final eval). "
            "'random' is the legacy noisy random-window subsample (--eval-batches)."
        ),
    )
    parser.add_argument(
        "--val-eval-count-last",
        type=int,
        default=0,
        help=(
            "for --val-eval strided: score only the last N targets of each window "
            "(full-context BPC), stride matched to N. 0 (default) uses "
            "context_length // 4, matching evaluate_spikegpt_checkpoint.py."
        ),
    )
    parser.add_argument(
        "--val-eval-tokens",
        type=int,
        default=0,
        help=(
            "for --val-eval strided: cap the in-loop eval to the first N val "
            "tokens (deterministic prefix) to bound cost. 0 (default) uses the "
            "whole val set."
        ),
    )
    parser.add_argument(
        "--compile-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run and report one compiled training step before timed logging when compile is enabled"
        ),
    )
    parser.add_argument("--sample-prompt", default="spik")
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument(
        "--mfu-peak-tflops",
        type=float,
        default=209.5,
        help=(
            "device peak for train/mfu, in TFLOP/s. Default 209.5 = RTX 5090 dense BF16 "
            "with FP32 accumulate (the training-relevant peak; GeForce runs FP32-accumulate "
            "at half the FP16-accumulate rate, and we use no structured sparsity). Set the "
            "matching dense bf16/fp32-accum peak for other hardware, or 0 to skip MFU."
        ),
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        help=(
            "optional SpikeGPT checkpoint to resume from; model config and vocabulary come from it"
        ),
    )
    parser.add_argument(
        "--reset-schedule",
        action="store_true",
        help=(
            "fine-tune mode: load only the model weights from --checkpoint-in, but reset the "
            "step counter (previous_steps=0) so the LR schedule is a fresh warmup->cosine over "
            "--steps, and start with a fresh optimizer (do not load the pretrain Adam state). "
            "Use when adapting a pretrained checkpoint to a new corpus; omit to resume a run."
        ),
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        help="optional path for saving the trained SpikeGPT model checkpoint",
    )
    parser.add_argument(
        "--best-checkpoint-out",
        type=Path,
        help=(
            "optional path to save the best-val-BPC checkpoint seen during training "
            "(keeps the best generalizing point, not just the latest)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dense-embedding",
        action="store_true",
        help="use ordinary dense token embeddings instead of hard surrogate binary embeddings",
    )
    parser.add_argument(
        "--no-spiking",
        action="store_true",
        help="ABLATION: build the continuous 'standard decoder' twin — LIF gates become "
        "identity (vanilla RWKV-v4) and the embedding is dense. Same params; isolates "
        "what the spiking binarization costs. Only applies to fresh runs (not --checkpoint-in).",
    )
    parser.add_argument(
        "--spike-input",
        action="store_true",
        help="ABLATION: 'hardware-faithful' LIF->Linear variant — spike each sub-block's "
        "INPUT so the projections consume spikes (the AC-capable placement the paper's "
        "energy table assumes), vs the canonical Linear->LIF. Measures the accuracy cost "
        "of making SpikeGPT actually energy-efficient. Fresh runs only; no generation yet.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="checkpoint SpikeGPT blocks during training to reduce saved activations",
    )
    add_compile_policy_arg(parser, extra_policies=("regional", "regional-lite"))
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
        help=(
            "torch.compile mode for --compile regional. Default max-autotune-no-cudagraphs "
            "is ~7%% faster than default at the cost of a longer (autotuning) compile; "
            "use 'default' for fast iteration. Ignored by regional-lite (fast-compile preset)."
        ),
    )
    parser.add_argument(
        "--compile-tail",
        action="store_true",
        help="also compile the ln_out/head/cross-entropy tail (regional only). Fuses the "
        "large-vocab head + CE — most of a full-model compile's win (~+6%% on the 216M) "
        "without the whole-model graph-break fragility.",
    )
    add_grad_clip_arg(parser)
    add_matmul_precision_arg(parser, default="high")
    parser.add_argument(
        "--amp",
        choices=("off", "bf16"),
        default="bf16",
        help=(
            "autocast mixed precision for the training forward pass (default bf16); ~1.6x "
            "faster and ~25%% less memory under torch.compile on CUDA, and a no-op on CPU. "
            "The WKV recurrence and LIF membrane stay in float32 internally regardless. "
            "Use 'off' for a pure-float32 run."
        ),
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.batch % args.grad_accum != 0:
        parser.error(
            f"--batch ({args.batch}) must be divisible by --grad-accum ({args.grad_accum})"
        )
    # Peak activation memory tracks this; the optimizer still sees --batch.
    micro_batch = args.batch // args.grad_accum

    torch.manual_seed(args.seed)
    configure_matmul_precision(args.matmul_precision)
    # enwik8 and other byte-level corpora contain raw bytes that are not valid
    # UTF-8, so read them as bytes when using the byte vocabulary; otherwise read
    # UTF-8 text.
    use_bins = args.train_bin is not None
    byte_file_mode = (not use_bins) and args.text_file is not None and args.vocab == "byte"
    raw_bytes: bytes = b""
    text = ""
    if not use_bins:
        if byte_file_mode:
            assert args.text_file is not None
            raw_bytes = args.text_file.read_bytes()
            if not raw_bytes:
                raise ValueError("text file is empty")
        else:
            text = (
                args.text_file.read_text(encoding="utf-8")
                if args.text_file is not None
                else args.text
            )
    checkpoint = (
        load_spike_language_checkpoint(args.checkpoint_in, map_location=args.device)
        if args.checkpoint_in is not None
        else None
    )
    if checkpoint is None:
        if args.vocab == "char":
            if use_bins:
                raise ValueError("--vocab char is unsupported with --train-bin; use bpe or byte")
            vocabulary = CharacterVocabulary.from_text(text)
        elif args.vocab == "byte":
            vocabulary = ByteVocabulary()
        else:
            vocabulary = BPEVocabulary.from_pretrained(args.bpe_tokenizer)
        # --no-spiking builds the continuous twin: no LIF binarization and a dense
        # embedding (a spiking embedding on a continuous model makes no sense).
        # --spike-input flips the LIF placement to LIF->Linear (hardware-faithful).
        spiking = not args.no_spiking
        spike_embedding = (not args.dense_embedding) and spiking
        spike_input = args.spike_input and spiking
        config = (
            SpikeGPTConfig(
                vocab_size=vocabulary.size,
                context_length=args.context_length,
                n_layer=args.layers,
                n_embd=args.embedding,
                dropout=args.dropout,
                model_type=args.model_type,
                attention=args.attention,
                n_head=args.n_head,
                lif_threshold=args.lif_threshold,
                spike_embedding=spike_embedding,
                spiking=spiking,
                spike_input=spike_input,
                gradient_checkpointing=args.activation_checkpointing,
            )
            if args.preset == "custom"
            else spikegpt_config_from_preset(
                args.preset,
                vocab_size=vocabulary.size,
                dropout=args.dropout,
                model_type=args.model_type,
                attention=args.attention,
                n_head=args.n_head,
                lif_threshold=args.lif_threshold,
                spike_embedding=spike_embedding,
                spiking=spiking,
                spike_input=spike_input,
                gradient_checkpointing=args.activation_checkpointing,
            )
        )
        raw_model = SpikeLanguageModel(config).to(device=args.device)
        checkpoint_metadata: dict[str, object] = {}
    else:
        vocabulary = checkpoint.vocabulary
        config = checkpoint.model.config
        raw_model = checkpoint.model.to(device=args.device)
        checkpoint_metadata = checkpoint.metadata
    if use_bins:
        assert args.train_bin is not None
        corpus = MemmapTokenCorpus.open(args.train_bin)
        if corpus.vocab_size != vocabulary.size:
            raise ValueError(
                f"--train-bin vocab_size {corpus.vocab_size} != model vocab "
                f"{vocabulary.size}; tokenize with the matching --bpe-tokenizer/--vocab"
            )
        train_tokens: torch.Tensor | MemmapTokenCorpus | TokenArrayView
        val_tokens: torch.Tensor | MemmapTokenCorpus | TokenArrayView
        if args.val_holdout_tokens > 0:
            # Hold out an in-domain tail of the train corpus as validation, so the
            # in-loop eval measures the true generalization gap (not a different
            # domain). --val-bin, if given, is ignored in this mode.
            train_tokens, val_tokens = corpus.split_tail(args.val_holdout_tokens)
        elif args.val_bin is not None:
            train_tokens, val_tokens = corpus, MemmapTokenCorpus.open(args.val_bin)
        else:
            train_tokens, val_tokens = corpus, corpus
        test_tokens: torch.Tensor | MemmapTokenCorpus | TokenArrayView = (
            MemmapTokenCorpus.open(args.test_bin)
            if args.test_bin is not None
            else torch.empty(0, dtype=torch.long)
        )
    else:
        tokens = (
            torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).to(torch.long)
            if byte_file_mode
            else vocabulary.encode(text)
        )
        train_tokens, val_tokens, test_tokens = split_train_val_test(
            tokens,
            validation_fraction=args.val_fraction,
            min_validation_tokens=args.min_val_tokens,
            test_fraction=args.test_fraction,
            test_tokens=args.test_tokens,
        )
    actual_vocab = vocabulary_name(vocabulary)
    actual_activation_checkpointing = raw_model.gradient_checkpointing
    # "previous_steps" records the global steps actually completed (set on every
    # periodic and final save), so it is the correct resume point for both partial
    # and finished runs. Fall back to total_steps/steps for older checkpoints that
    # predate the field.
    previous_steps = (
        0
        if args.reset_schedule
        else (
            metadata_nonnegative_int(checkpoint_metadata, "previous_steps")
            or metadata_nonnegative_int(checkpoint_metadata, "total_steps")
            or metadata_nonnegative_int(checkpoint_metadata, "steps")
        )
    )
    total_steps = previous_steps + args.steps
    compile_model = (
        True
        if args.compile in ("regional", "regional-lite")
        else resolve_compile_policy(args.compile, args.device)
    )
    wsd_cfg = (
        f"decay_steps:{args.decay_steps},decay_shape:{args.decay_shape},"
        if args.lr_schedule == "wsd"
        else ""
    )
    print(
        "config="
        f"device:{args.device},compile:{compile_model},compile_policy:{args.compile},"
        f"compile_mode:{args.compile_mode},"
        f"vocab:{actual_vocab},preset:{args.preset},model_type:{config.model_type},"
        f"attention:{config.attention},n_head:{config.n_head},"
        f"context_length:{config.context_length},layers:{config.n_layer},"
        f"embedding:{config.n_embd},"
        f"batch:{args.batch},steps:{args.steps},lr:{args.lr},"
        f"weight_decay:{args.weight_decay},dropout:{config.dropout},"
        f"lif_threshold:{config.lif_threshold},"
        f"grad_clip:{args.grad_clip},"
        f"lr_schedule:{args.lr_schedule},{wsd_cfg}"
        f"amp:{args.amp},matmul_precision:{args.matmul_precision},"
        f"val_eval:{args.val_eval},"
        f"compile_warmup:{args.compile_warmup},"
        f"spike_embedding:{config.spike_embedding},spiking:{config.spiking},"
        f"spike_input:{config.spike_input},"
        f"activation_checkpointing:{actual_activation_checkpointing},"
        f"vocab_size:{vocabulary.size},"
        f"train_tokens:{train_tokens.numel()},val_tokens:{val_tokens.numel()},"
        f"test_tokens_heldout:{test_tokens.numel()},"
        f"previous_steps:{previous_steps},total_steps:{total_steps},"
        f"checkpoint_in:{'' if args.checkpoint_in is None else args.checkpoint_in}",
        flush=True,
    )
    if checkpoint is not None:
        print(f"checkpoint_loaded={args.checkpoint_in}", flush=True)
        if checkpoint_metadata:
            print(f"checkpoint_metadata={checkpoint_metadata}", flush=True)
    print()
    print_model_summary(raw_model)
    print()

    # regional-lite keeps its fast-compile options preset; regional uses --compile-mode
    # (mode and options are mutually exclusive in torch.compile).
    regional_lite = args.compile == "regional-lite"
    compile_mode = None if (regional_lite or args.compile_mode == "default") else args.compile_mode
    model = (
        compile_spikegpt_regions(
            raw_model,
            fullgraph=args.compile == "regional",
            options=REGIONAL_LITE_COMPILE_OPTIONS if regional_lite else None,
            mode=compile_mode,
        )
        if args.compile in ("regional", "regional-lite")
        else compile_training_model(raw_model, compile_model)
    )
    if args.compile_tail and args.compile in ("regional", "regional-lite"):
        # Compile just the head/CE tail as its own region (regional model is
        # raw_model, compiled in place); keeps the recurrent body on the validated
        # regional path while fusing the large-vocab head + cross-entropy.
        raw_model.loss_tail = torch.compile(raw_model.loss_tail, mode=compile_mode)  # type: ignore[method-assign]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        # Single fused CUDA kernel for the optimizer step (fewer launches than the
        # default foreach path); CPU AdamW has no fused impl, so guard on device.
        fused=torch.device(args.device).type == "cuda",
    )
    optimizer_loaded = False
    if (
        checkpoint is not None
        and checkpoint.optimizer_state_dict is not None
        and not args.reset_schedule
    ):
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        optimizer_loaded = True
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        print("optimizer_loaded=True", flush=True)
    # MFU accounting. fwd+bwd FLOPs/token from the study's exact matmul count (the same
    # accounting behind C = 6*N_flop*D, incl. vanilla attention's quadratic term when
    # applicable), so train/mfu is self-consistent with the scaling FLOPs. It's constant
    # for a run; MFU = flops_per_tok * tokens_per_s / peak.
    mfu_counts = count_spikegpt_params(
        vocab_size=vocabulary.size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        model_type=config.model_type,
        attention=config.attention,
        context_length=config.context_length,
    )
    flops_per_tok = flops_per_token(mfu_counts, backward=True)
    mfu_peak_flops = args.mfu_peak_tflops * 1e12
    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config={
            "device": args.device,
            "compile": compile_model,
            "compile_policy": args.compile,
            "vocab": actual_vocab,
            "preset": args.preset,
            "model_type": config.model_type,
            "attention": config.attention,
            "n_head": config.n_head,
            "spiking": config.spiking,
            "context_length": config.context_length,
            "layers": config.n_layer,
            "embedding": config.n_embd,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "micro_batch": args.batch // args.grad_accum,
            "steps": args.steps,
            "seed": args.seed,
            # LR schedule (was only logging the peak lr) -- needed to reproduce or
            # compare runs, and to interpret the loss curve's tail.
            "lr": args.lr,
            "lr_final": args.lr_final,
            "lr_schedule": args.lr_schedule,
            "warmup_steps": args.warmup_steps,
            "decay_steps": args.decay_steps,
            "decay_shape": args.decay_shape,
            "weight_decay": args.weight_decay,
            "dropout": config.dropout,
            "grad_clip": args.grad_clip,
            "amp": args.amp,
            "matmul_precision": args.matmul_precision,
            "lif_threshold": config.lif_threshold,
            "rope_base": config.rope_base,
            "compile_warmup": args.compile_warmup,
            "spike_embedding": config.spike_embedding,
            "spike_input": config.spike_input,
            "activation_checkpointing": actual_activation_checkpointing,
            "vocab_size": vocabulary.size,
            # Parameter counts (so W&B/analysis need not recompute; N for the scaling
            # law is total incl. embedding -- the Chinchilla convention).
            "n_params": sum(p.numel() for p in raw_model.parameters()),
            "n_params_nonvocab": sum(p.numel() for p in raw_model.parameters())
            - 2 * vocabulary.size * config.n_embd,
            # MFU provenance: the FLOPs/token and device peak that define train/mfu.
            "flops_per_token": flops_per_tok,
            "mfu_peak_tflops": args.mfu_peak_tflops,
            # Data provenance -- WHICH corpus this run trained on (was only logging the
            # token COUNT). Essential for a scaling study to be reproducible.
            "train_bin": "" if args.train_bin is None else str(args.train_bin),
            "bpe_tokenizer": args.bpe_tokenizer if actual_vocab == "bpe" else "",
            "val_holdout_tokens": args.val_holdout_tokens,
            "val_eval": args.val_eval,
            "val_eval_tokens": args.val_eval_tokens,
            "train_tokens": train_tokens.numel(),
            "val_tokens": val_tokens.numel(),
            "test_tokens_heldout": test_tokens.numel(),
            "previous_steps": previous_steps,
            "total_steps": total_steps,
            "checkpoint_in": "" if args.checkpoint_in is None else str(args.checkpoint_in),
            "optimizer_loaded": optimizer_loaded,
        },
    )
    step_times: list[float] = []
    best_val_bpc = float("inf")

    def save_run_checkpoint(steps_completed: int, path: Path | None = None) -> None:
        target = path if path is not None else args.checkpoint_out
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        save_spike_language_checkpoint(
            target,
            raw_model,
            vocabulary,
            metadata={
                "vocab": actual_vocab,
                "preset": args.preset,
                "model_type": config.model_type,
                "attention": config.attention,
                "n_head": config.n_head,
                "steps": args.steps,
                "previous_steps": steps_completed,
                "total_steps": total_steps,
                "batch": args.batch,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "train_tokens": train_tokens.numel(),
                "val_tokens": val_tokens.numel(),
                "test_tokens_heldout": test_tokens.numel(),
            },
            optimizer=optimizer,
        )

    # Bits per modeling unit: "BPC" (bits/char) for byte/char vocabularies, "BPT"
    # (bits/token) for subword BPE — same value (loss/ln2), unit-correct label.
    bits_label = "BPT" if isinstance(vocabulary, BPEVocabulary) else "BPC"
    print(
        f"| Step | Train Loss | Val Loss | Val {bits_label} | Val PPL | "
        "Emb Spike Rate | Mean Block Spike Rate | Grad Norm | Step ms |",
        flush=True,
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    try:
        model.train()
        torch_device = torch.device(args.device)

        def amp_context() -> contextlib.AbstractContextManager:
            if args.amp == "bf16" and torch_device.type == "cuda":
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            return contextlib.nullcontext()

        if compile_model and args.compile_warmup:
            warmup_inputs, warmup_targets = sample_token_batch(
                train_tokens,
                batch_size=micro_batch,
                context_length=config.context_length,
                device=args.device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mark_compiled_invocation_boundary(compile_model)
            with amp_context():
                warmup_loss, _warmup_logits = model(warmup_inputs, warmup_targets)
            warmup_loss.backward()
            warmup_grad_norm = clip_gradients(model, args.grad_clip)
            optimizer.step()
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            warmup_seconds = time.perf_counter() - start
            print(f"compile_warmup_step_ms={warmup_seconds * 1000:.3f}", flush=True)
            print(f"compile_warmup_loss={float(warmup_loss.detach()):.6f}", flush=True)
            if warmup_grad_norm is not None:
                print(f"compile_warmup_grad_norm={warmup_grad_norm:.4f}", flush=True)
            log_wandb(
                wandb_run,
                {
                    "compile/warmup_step_ms": warmup_seconds * 1000,
                    "compile/warmup_loss": float(warmup_loss.detach()),
                    **(
                        {}
                        if warmup_grad_norm is None
                        else {"compile/warmup_grad_norm": warmup_grad_norm}
                    ),
                },
                step=0,
            )

        for step in range(1, args.steps + 1):
            global_step = previous_steps + step
            if args.lr_final is not None:
                if args.lr_schedule == "wsd":
                    lr_now = wsd_lr(
                        global_step - 1,
                        total_steps,
                        args.lr,
                        args.lr_final,
                        args.warmup_steps,
                        args.decay_steps,
                        args.decay_shape,
                    )
                else:
                    lr_now = cosine_lr(
                        global_step - 1, total_steps, args.lr, args.lr_final, args.warmup_steps
                    )
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
            inputs, targets = sample_token_batch(
                train_tokens,
                batch_size=args.batch,
                context_length=config.context_length,
                device=args.device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # Accumulate over micro-batches: each one's activations are freed after its
            # backward, so peak memory tracks micro_batch while the step still sees the
            # full effective batch. The loss is mean-reduced and the micro-batches are
            # equal-sized, so summing grads of loss/grad_accum reproduces the full-batch
            # gradient exactly -- the LR calibration (batch-dependent) still holds.
            loss_value = 0.0
            for micro_start in range(0, args.batch, micro_batch):
                micro_slice = slice(micro_start, micro_start + micro_batch)
                mark_compiled_invocation_boundary(compile_model)
                with amp_context():
                    loss, _logits = model(inputs[micro_slice], targets[micro_slice])
                (loss / args.grad_accum).backward()
                loss_value += float(loss.detach()) / args.grad_accum
            grad_norm = clip_gradients(model, args.grad_clip)
            optimizer.step()
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            step_seconds = time.perf_counter() - start
            step_times.append(step_seconds)

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                # Cheap, frequent logging: train loss, lr, step time, grad norm are
                # already computed above. The spike-rate forward and the val eval are
                # expensive, so they run only on the coarser eval cadence.
                eval_metrics = None
                rates = None
                mean_block_rate = None
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    mark_compiled_invocation_boundary(compile_model)
                    rates = raw_model.spike_rates(inputs)
                    block_rates = [value for key, value in rates.items() if key != "embedding"]
                    mean_block_rate = sum(block_rates) / len(block_rates) if block_rates else 0.0
                    mark_compiled_invocation_boundary(compile_model)
                    if args.val_eval == "strided":
                        count_last = args.val_eval_count_last or config.context_length // 4
                        eval_slice = (
                            val_tokens[: args.val_eval_tokens]
                            if args.val_eval_tokens > 0
                            else val_tokens
                        )
                        eval_metrics = evaluate_language_model_strided(
                            raw_model,
                            eval_slice,
                            batch_size=args.batch,
                            context_length=config.context_length,
                            device=args.device,
                            stride=count_last,
                            count_last=count_last,
                        )
                    else:
                        eval_metrics = evaluate_language_model(
                            raw_model,
                            val_tokens,
                            batch_size=args.batch,
                            context_length=config.context_length,
                            device=args.device,
                            batches=args.eval_batches,
                        )
                emb_rate_str = "" if rates is None else f"{rates['embedding']:.4f}"
                block_rate_str = "" if mean_block_rate is None else f"{mean_block_rate:.4f}"
                print(
                    f"| {global_step} | {loss_value:.6f} | "
                    f"{'' if eval_metrics is None else f'{eval_metrics.loss:.6f}'} | "
                    f"{'' if eval_metrics is None else f'{eval_metrics.bits_per_character:.4f}'} | "
                    f"{'' if eval_metrics is None else f'{eval_metrics.perplexity:.4f}'} | "
                    f"{emb_rate_str} | {block_rate_str} | "
                    f"{'' if grad_norm is None else f'{grad_norm:.4f}'} | "
                    f"{step_seconds * 1000:.3f} |",
                    flush=True,
                )
                toks_per_s = args.batch * config.context_length / step_seconds
                achieved_flops = flops_per_tok * toks_per_s
                wandb_metrics = {
                    "train/loss": loss_value,
                    "train/step_ms": step_seconds * 1000,
                    "train/tokens_per_s": toks_per_s,
                    "train/tflops": achieved_flops / 1e12,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/completion_pct": 100.0 * global_step / total_steps,
                }
                if mfu_peak_flops > 0:
                    wandb_metrics["train/mfu"] = achieved_flops / mfu_peak_flops
                if rates is not None:
                    wandb_metrics["train/embedding_spike_rate"] = rates["embedding"]
                    wandb_metrics["train/mean_block_spike_rate"] = mean_block_rate
                if grad_norm is not None:
                    wandb_metrics["train/grad_norm"] = grad_norm
                if eval_metrics is not None:
                    wandb_metrics.update(
                        {
                            "val/loss": eval_metrics.loss,
                            f"val/{bits_label.lower()}": eval_metrics.bits_per_character,
                            "val/perplexity": eval_metrics.perplexity,
                        }
                    )
                log_wandb(wandb_run, wandb_metrics, step=global_step)

                if (
                    args.best_checkpoint_out is not None
                    and eval_metrics is not None
                    and eval_metrics.bits_per_character < best_val_bpc
                ):
                    best_val_bpc = eval_metrics.bits_per_character
                    save_run_checkpoint(global_step, path=args.best_checkpoint_out)
                    print(
                        f"best_checkpoint={args.best_checkpoint_out} "
                        f"(step {global_step}, val {bits_label} {best_val_bpc:.4f})",
                        flush=True,
                    )

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_run_checkpoint(global_step)
                print(f"checkpoint={args.checkpoint_out} (step {global_step})", flush=True)
    finally:
        finish_wandb(wandb_run)

    print()
    print_step_time_summary(step_times)
    save_run_checkpoint(total_steps)
    if args.checkpoint_out is not None:
        print(f"checkpoint={args.checkpoint_out}", flush=True)

    prompt = args.sample_prompt
    try:
        prompt_token_ids = vocabulary.encode(prompt)
    except ValueError as exc:
        print(
            f"sample_skipped={exc}",
            flush=True,
        )
        return
    prompt_tokens = prompt_token_ids.unsqueeze(0).to(device=args.device)
    mark_compiled_invocation_boundary(compile_model)
    generated = raw_model.generate(
        prompt_tokens,
        max_new_tokens=args.sample_tokens,
        top_k=min(8, vocabulary.size),
        sampling="greedy",
    )
    print(f"sample={vocabulary.decode(generated[0].cpu())!r}", flush=True)


if __name__ == "__main__":
    main()
