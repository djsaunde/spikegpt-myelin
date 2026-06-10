"""Tokenize a corpus to a uint16 ``.bin`` memmap for large-scale pretraining.

Two sources:
  * local text files (``--text-file`` repeatable) — e.g. WikiText train/valid/test;
  * a streaming HuggingFace dataset (``--hf-dataset``) — e.g. OpenWebText2.

Tokens are written with the same vocabulary the model will train under (default
the GPT-NeoX BPE), so the ``.bin`` ids match the checkpoint's tokenizer exactly.
Output is a flat ``uint16`` ``.bin`` + a ``.json`` sidecar (read by
``spikegpt.token_corpus.MemmapTokenCorpus``).

Examples:
  # WikiText-103 splits -> three bins
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --text-file wikitext-103/wiki.train.tokens --output data/wikitext103_train.bin
  # OpenWebText2 (streaming), capped for a validation slice
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --hf-dataset Skylion007/openwebtext --hf-split train --text-column text \\
    --max-tokens 1_000_000_000 --output data/owt_1b.bin
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from spikegpt import BPEVocabulary, ByteVocabulary, TokenCorpusWriter


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
    parser.add_argument("--hf-dataset", help="streaming HuggingFace dataset name")
    parser.add_argument("--hf-config", help="HuggingFace dataset config (e.g. wikitext-103-raw-v1)")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-column", default="text")
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

    if not args.text_file and not args.hf_dataset:
        parser.error("provide --text-file or --hf-dataset")

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
