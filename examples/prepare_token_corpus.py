"""Tokenize a corpus to a uint16 ``.bin`` memmap for large-scale pretraining.

Two sources:
  * local text files (``--text-file`` repeatable) — e.g. WikiText train/valid/test;
  * a streaming HuggingFace dataset — either a named ``--dataset`` preset
    (``fineweb-edu``, ``openwebtext``) or an arbitrary ``--hf-dataset`` repo id.

Tokens are written with the same vocabulary the model will train under (default
the GPT-NeoX BPE), so the ``.bin`` ids match the checkpoint's tokenizer exactly.
Output is a flat ``uint16`` ``.bin`` + a ``.json`` sidecar (read by
``spikegpt.token_corpus.MemmapTokenCorpus``).

Examples:
  # WikiText-103 splits -> three bins
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --text-file wikitext-103/wiki.train.tokens --output data/wikitext103_train.bin
  # FineWeb-Edu (streaming, 10B-token sample), capped to a 1B-token slice
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --dataset fineweb-edu --max-tokens 1_000_000_000 --output data/fineweb_edu_1b.bin
  # OpenWebText (streaming), capped for a validation slice
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --dataset openwebtext --max-tokens 1_000_000_000 --output data/owt_1b.bin
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from spikegpt import BPEVocabulary, ByteVocabulary, TokenCorpusWriter


@dataclass(frozen=True)
class DatasetPreset:
    """A named streaming HuggingFace source with sensible defaults."""

    hf_dataset: str
    hf_config: str | None
    hf_split: str
    text_column: str


# Curated streaming-dataset choices for pretraining. A convenience wrapper over
# the raw --hf-* flags so a corpus can be prepared by name; any --hf-* flag still
# overrides the matching preset field (e.g. --hf-config sample-100BT for more data).
DATASET_PRESETS: dict[str, DatasetPreset] = {
    # FineWeb-Edu: educational-quality filtering of FineWeb CommonCrawl (1.3T tokens
    # total; Apache-2.0 / ODC-By, ungated). Defaults to the 10B-token sample — pass
    # --hf-config sample-100BT / sample-350BT / default for larger slices.
    "fineweb-edu": DatasetPreset("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    # OpenWebText: ~8M-doc open reproduction of GPT-2's WebText (the SpikeGPT paper's
    # OpenWebText2 pretraining stand-in used for the 216M WikiText result).
    "openwebtext": DatasetPreset("Skylion007/openwebtext", None, "train", "text"),
}


def resolve_hf_source(args: argparse.Namespace) -> tuple[str | None, str | None, str, str]:
    """Resolve (hf_dataset, hf_config, hf_split, text_column) from a --dataset
    preset and/or explicit --hf-* overrides. Explicit flags always win over the
    preset; the preset wins over the built-in train/text fallbacks."""
    preset = DATASET_PRESETS[args.dataset] if args.dataset else None
    hf_dataset = args.hf_dataset or (preset.hf_dataset if preset else None)
    hf_config = args.hf_config or (preset.hf_config if preset else None)
    hf_split = args.hf_split or (preset.hf_split if preset else None) or "train"
    text_column = args.text_column or (preset.text_column if preset else None) or "text"
    return hf_dataset, hf_config, hf_split, text_column


def build_vocab(args: argparse.Namespace):
    if args.vocab == "byte":
        return ByteVocabulary()
    return BPEVocabulary.from_pretrained(args.bpe_tokenizer)


def iter_texts(args: argparse.Namespace):
    """Yield text chunks from local files or a streaming HF dataset."""
    if args.text_file:
        for path in args.text_file:
            yield Path(path).read_text(encoding="utf-8")
        return
    from datasets import load_dataset

    dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, streaming=True)
    for example in dataset:
        text = example.get(args.text_column)
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="output .bin path")
    parser.add_argument("--vocab", choices=("bpe", "byte"), default="bpe")
    parser.add_argument("--bpe-tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--text-file", action="append", help="local text file(s); repeatable")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PRESETS),
        help="named streaming dataset preset (fills --hf-dataset/--hf-config/--hf-split/"
        "--text-column); any --hf-* flag overrides the matching preset field",
    )
    parser.add_argument("--hf-dataset", help="streaming HuggingFace dataset name")
    parser.add_argument("--hf-config", help="HuggingFace dataset config (e.g. wikitext-103-raw-v1)")
    parser.add_argument("--hf-split", help="HuggingFace split (default: train)")
    parser.add_argument("--text-column", help="text field in the dataset (default: text)")
    parser.add_argument(
        "--eos-id",
        type=int,
        default=0,
        help="token id inserted between documents (GPT-NeoX <|endoftext|>=0); -1 to disable",
    )
    parser.add_argument("--max-tokens", type=int, help="stop after this many tokens")
    parser.add_argument("--log-every", type=int, default=20000, help="log every ~N documents")
    parser.add_argument(
        "--batch-docs", type=int, default=1024, help="documents per batched encode call"
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20_000_000,
        help="re-write the .json sidecar every N tokens (crash-resilience for streaming)",
    )
    args = parser.parse_args()

    if args.text_file and (args.dataset or args.hf_dataset):
        parser.error("provide either --text-file or a streaming dataset, not both")
    if not args.text_file:
        args.hf_dataset, args.hf_config, args.hf_split, args.text_column = resolve_hf_source(args)
        if not args.hf_dataset:
            parser.error("provide --text-file, --dataset, or --hf-dataset")

    vocab = build_vocab(args)
    start = time.perf_counter()
    docs = 0
    last_flush_tokens = 0
    buffer: list[str] = []

    def flush_buffer(writer) -> None:
        nonlocal buffer, docs
        if not buffer:
            return
        # Batch-encode across all CPU cores (the Rust tokenizer parallelizes); for
        # the byte fallback, encode per text.
        if isinstance(vocab, BPEVocabulary):
            batches = vocab.encode_batch(buffer)
        else:
            batches = [vocab.encode(text).tolist() for text in buffer]
        for ids in batches:
            if args.eos_id >= 0:
                ids = [*ids, args.eos_id]
            writer.write(ids)
        docs += len(buffer)
        buffer = []

    with TokenCorpusWriter(args.output, vocab_size=vocab.size) as writer:
        for text in iter_texts(args):
            buffer.append(text)
            if len(buffer) < args.batch_docs:
                continue
            flush_buffer(writer)
            if docs % max(args.log_every, args.batch_docs) < args.batch_docs:
                rate = writer.n_tokens / max(time.perf_counter() - start, 1e-9)
                print(
                    f"docs={docs:,} tokens={writer.n_tokens:,} ({rate / 1e6:.2f}M tok/s)",
                    flush=True,
                )
            # Periodically persist the sidecar so a streaming crash still leaves a
            # usable corpus (always a valid prefix of the .bin).
            if writer.n_tokens - last_flush_tokens >= args.flush_every:
                writer.flush_sidecar()
                last_flush_tokens = writer.n_tokens
            if args.max_tokens is not None and writer.n_tokens >= args.max_tokens:
                break
        flush_buffer(writer)
        total = writer.n_tokens
    print(
        f"wrote {args.output} — {total:,} tokens, vocab_size={vocab.size}, "
        f"{docs:,} docs, {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    # The corpus + sidecar are fully written and flushed above. Exit hard to skip
    # interpreter finalization, which otherwise crashes the HF streaming dataset's
    # background download thread ("PyGILState_Release") — cosmetic but noisy.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
