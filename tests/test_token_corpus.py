"""Tests for the memmap token corpus and the generalized batch sampler."""

from __future__ import annotations

import torch

from spikegpt import MemmapTokenCorpus, TokenCorpusWriter
from spikegpt.language import sample_token_batch


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
