"""Analyze the spiking nature of a saved SpikeGPT checkpoint.

Runs the model over strided windows of a corpus, captures the real LIF spikes via
forward hooks, and writes a Markdown report: how much of the model is spiking,
spike density (overall / per population / per layer / per token position), and the
fraction of dead (never-firing) and saturated (always-firing) neurons.

Example:
  uv run python examples/analyze_spikegpt_spikes.py \\
    runs/spikegpt_enwik8_12L512d_ctx3072.final139k.pt \\
    --text-file data/enwik8_test --batch 8
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

import torch

from spikegpt import (
    ByteVocabulary,
    collect_spike_statistics,
    compute_spiking_fraction,
    format_spike_report,
    load_spike_language_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-file", type=Path, default=Path("data/enwik8_test"))
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--dead-threshold", type=float, default=0.0)
    parser.add_argument("--saturated-threshold", type=float, default=0.99)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    checkpoint = load_spike_language_checkpoint(args.checkpoint, map_location=args.device)
    model = checkpoint.model.to(device=args.device).eval()
    vocabulary = checkpoint.vocabulary
    config = model.config

    if isinstance(vocabulary, ByteVocabulary):
        tokens = torch.frombuffer(bytearray(args.text_file.read_bytes()), dtype=torch.uint8).to(
            torch.long
        )
    else:
        tokens = vocabulary.encode(args.text_file.read_text(encoding="utf-8"))

    stats = collect_spike_statistics(
        model,
        tokens,
        context_length=args.context_length,
        stride=args.stride,
        batch_size=args.batch,
        max_windows=args.max_windows,
        device=args.device,
        dead_threshold=args.dead_threshold,
        saturated_threshold=args.saturated_threshold,
    )
    fraction = compute_spiking_fraction(model)

    meta_lines = [
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Config: {config.n_layer}L / {config.n_embd}d / ctx {config.context_length} / "
        f"spike_embedding={config.spike_embedding}",
        f"- Corpus: `{args.text_file}` ({tokens.numel():,} tokens)",
        f"- Device: {args.device} · windows: {stats.windows_processed} · "
        f"tokens scored: {stats.tokens_processed:,}",
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
    ]
    report = format_spike_report(
        stats,
        fraction,
        title=f"SpikeGPT spike analysis — {args.checkpoint.stem}",
        meta_lines=meta_lines,
    )
    print(report)

    out_path = args.out or (
        Path("benchmarks/results") / f"spikegpt_spike_analysis_{args.checkpoint.stem}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)

    if args.write_csv:
        neuron_csv = out_path.with_suffix(".neurons.csv")
        with neuron_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["population", "channel", "rate"])
            for name, pop in stats.populations.items():
                for channel, rate in enumerate(pop.per_neuron_rate.tolist()):
                    writer.writerow([name, channel, rate])
        print(f"wrote {neuron_csv}", flush=True)


if __name__ == "__main__":
    main()
