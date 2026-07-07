"""Tests for the memmap token corpus and the generalized batch sampler."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from spikegpt import MemmapTokenCorpus, TokenCorpusWriter
from spikegpt.language import sample_token_batch

# The dataset presets live in the examples/ script (not an installed package), so
# load it by path to test the source-resolution logic without any network access.
# Register it in sys.modules before exec so its @dataclass can resolve field types.
_PREP_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_token_corpus.py"
_spec = importlib.util.spec_from_file_location("prepare_token_corpus", _PREP_PATH)
prepare_token_corpus = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prepare_token_corpus
_spec.loader.exec_module(prepare_token_corpus)


def _source_args(**overrides) -> argparse.Namespace:
    base = {
        "dataset": None,
        "hf_dataset": None,
        "hf_config": None,
        "hf_split": None,
        "text_column": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_fineweb_edu_preset_resolves_to_hf_source() -> None:
    hf_dataset, hf_config, hf_split, text_column = prepare_token_corpus.resolve_hf_source(
        _source_args(dataset="fineweb-edu")
    )
    assert hf_dataset == "HuggingFaceFW/fineweb-edu"
    assert hf_config == "sample-10BT"
    assert hf_split == "train"
    assert text_column == "text"


def test_openwebtext_preset_resolves_with_no_config() -> None:
    hf_dataset, hf_config, hf_split, text_column = prepare_token_corpus.resolve_hf_source(
        _source_args(dataset="openwebtext")
    )
    assert hf_dataset == "Skylion007/openwebtext"
    assert hf_config is None
    assert (hf_split, text_column) == ("train", "text")


def test_explicit_hf_flags_override_preset_fields() -> None:
    # --hf-config picks a larger FineWeb-Edu sample than the preset default.
    _, hf_config, hf_split, _ = prepare_token_corpus.resolve_hf_source(
        _source_args(dataset="fineweb-edu", hf_config="sample-100BT", hf_split="test")
    )
    assert hf_config == "sample-100BT"
    assert hf_split == "test"


def test_raw_hf_dataset_without_preset_falls_back_to_defaults() -> None:
    hf_dataset, hf_config, hf_split, text_column = prepare_token_corpus.resolve_hf_source(
        _source_args(hf_dataset="some/repo")
    )
    assert hf_dataset == "some/repo"
    assert hf_config is None
    assert (hf_split, text_column) == ("train", "text")


def test_token_corpus_write_open_round_trip(tmp_path) -> None:
    ids = list(range(500))
    bin_path = tmp_path / "corpus.bin"
    with TokenCorpusWriter(bin_path, vocab_size=512) as writer:
        writer.write(ids[:200])
        writer.write(ids[200:])
    assert (tmp_path / "corpus.json").exists()

    corpus = MemmapTokenCorpus.open(bin_path)
    assert corpus.numel() == 500
    assert len(corpus) == 500
    assert corpus.vocab_size == 512
    window = corpus[10:20]
    assert window.dtype == torch.long
    assert window.tolist() == list(range(10, 20))


def test_sample_token_batch_over_memmap_matches_tensor_interface(tmp_path) -> None:
    ids = list(range(1000))
    bin_path = tmp_path / "c.bin"
    with TokenCorpusWriter(bin_path, vocab_size=1024) as writer:
        writer.write(ids)
    corpus = MemmapTokenCorpus.open(bin_path)

    torch.manual_seed(0)
    inputs, targets = sample_token_batch(corpus, batch_size=4, context_length=8, device="cpu")
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert inputs.dtype == torch.long
    # Targets are inputs shifted by one (next-token windows over a contiguous corpus).
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert int(inputs.max()) < corpus.vocab_size


def test_split_tail_holds_out_in_domain_validation(tmp_path) -> None:
    ids = list(range(1000))
    bin_path = tmp_path / "c.bin"
    with TokenCorpusWriter(bin_path, vocab_size=1024) as writer:
        writer.write(ids)
    corpus = MemmapTokenCorpus.open(bin_path)

    head, tail = corpus.split_tail(200)
    assert head.numel() == 800
    assert tail.numel() == 200
    assert head.vocab_size == tail.vocab_size == 1024
    assert head[0:5].tolist() == [0, 1, 2, 3, 4]
    assert tail[0:5].tolist() == [800, 801, 802, 803, 804]
    # The held-out tail drives the sampler just like a full corpus.
    inputs, _targets = sample_token_batch(tail, batch_size=2, context_length=8, device="cpu")
    assert inputs.shape == (2, 8)
    import pytest

    with pytest.raises(ValueError, match="tail_tokens"):
        corpus.split_tail(0)


def test_writer_rejects_vocab_exceeding_dtype(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="exceeds dtype"):
        TokenCorpusWriter(tmp_path / "x.bin", vocab_size=70000)  # > uint16 max
