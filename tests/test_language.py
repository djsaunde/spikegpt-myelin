from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spikegpt import (
    SPIKEGPT_PRESETS,
    ByteVocabulary,
    CharacterVocabulary,
    SpikeGPTConfig,
    SpikeLanguageModel,
    SpikingSequenceLIF,
    evaluate_language_model,
    evaluate_language_model_strided,
    language_vocabulary_from_dict,
    language_vocabulary_to_dict,
    load_spike_language_checkpoint,
    sample_token_batch,
    save_spike_language_checkpoint,
    spikegpt_config_from_dict,
    spikegpt_config_from_preset,
    spikegpt_config_to_dict,
    split_token_sequence,
    split_train_val_test,
    weighted_key_value,
)


def test_weighted_key_value_returns_expected_first_value_and_backpropagates() -> None:
    key = torch.randn((2, 4, 3), requires_grad=True)
    value = torch.randn((2, 4, 3), requires_grad=True)
    time_decay = torch.zeros(3, requires_grad=True)
    time_first = torch.zeros(3, requires_grad=True)

    output = weighted_key_value(key, value, time_decay, time_first)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == key.shape
    assert torch.allclose(output[:, 0], value[:, 0])
    assert key.grad is not None
    assert value.grad is not None
    assert time_decay.grad is not None
    assert time_first.grad is not None


def test_weighted_key_value_rejects_bad_shapes() -> None:
    key = torch.randn((2, 4, 3))
    value = torch.randn((2, 4, 2))

    with pytest.raises(ValueError, match="same shape"):
        weighted_key_value(key, value, torch.zeros(3), torch.zeros(3))


def test_character_vocabulary_round_trips_and_rejects_unknown_chars() -> None:
    vocab = CharacterVocabulary.from_text("banana")
    encoded = vocab.encode("nab")

    assert vocab.size == 3
    assert vocab.decode(encoded) == "nab"
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        vocab.encode("band")


def test_byte_vocabulary_round_trips_utf8_and_has_fixed_size() -> None:
    text = "spike \u03bb"
    vocab = ByteVocabulary.from_text(text)
    encoded = vocab.encode(text)

    assert vocab.size == 256
    assert encoded.dtype == torch.long
    assert encoded.max() < vocab.size
    assert vocab.decode(encoded) == text


def _tiny_bpe_vocab():
    # Train a tiny in-memory BPE tokenizer (no network) to exercise BPEVocabulary.
    tokenizers = pytest.importorskip("tokenizers")
    from spikegpt import BPEVocabulary

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=200, special_tokens=["<unk>"])
    tokenizer.train_from_iterator(
        ["spiking neural networks", "hello world", "the quick brown fox"] * 20, trainer
    )
    return BPEVocabulary(tokenizer_json=tokenizer.to_str(), vocab_size=tokenizer.get_vocab_size())


def test_bpe_vocabulary_encodes_and_serializes_round_trip() -> None:
    from spikegpt.language import language_vocabulary_from_dict, language_vocabulary_to_dict

    vocab = _tiny_bpe_vocab()
    encoded = vocab.encode("hello spiking world")
    assert encoded.dtype == torch.long
    assert encoded.ndim == 1
    assert int(encoded.max()) < vocab.size
    assert isinstance(vocab.decode(encoded), str)

    # Serializing and reloading reproduces an identical tokenizer (same encode).
    restored = language_vocabulary_from_dict(language_vocabulary_to_dict(vocab))
    assert restored.size == vocab.size
    assert torch.equal(restored.encode("the quick brown fox"), vocab.encode("the quick brown fox"))


def test_spike_language_checkpoint_round_trips_bpe_vocabulary(tmp_path: Path) -> None:
    vocab = _tiny_bpe_vocab()
    config = SpikeGPTConfig(
        vocab_size=vocab.size, context_length=16, n_layer=1, n_embd=32, dropout=0.0
    )
    model = SpikeLanguageModel(config).eval()
    path = tmp_path / "bpe_ckpt.pt"
    save_spike_language_checkpoint(path, model, vocab)
    loaded = load_spike_language_checkpoint(path)

    from spikegpt import BPEVocabulary

    assert isinstance(loaded.vocabulary, BPEVocabulary)
    assert loaded.vocabulary.size == vocab.size
    assert torch.equal(loaded.vocabulary.encode("hello world"), vocab.encode("hello world"))


def test_language_vocabulary_serialization_round_trips() -> None:
    char_vocab = CharacterVocabulary.from_text("banana")
    byte_vocab = ByteVocabulary()

    restored_char = language_vocabulary_from_dict(language_vocabulary_to_dict(char_vocab))
    restored_byte = language_vocabulary_from_dict(language_vocabulary_to_dict(byte_vocab))

    assert isinstance(restored_char, CharacterVocabulary)
    assert restored_char.tokens == char_vocab.tokens
    assert isinstance(restored_byte, ByteVocabulary)


def test_language_vocabulary_serialization_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unsupported vocabulary type"):
        language_vocabulary_from_dict({"type": "wordpiece"})


def test_spikegpt_config_from_preset_uses_named_dimensions() -> None:
    config = spikegpt_config_from_preset(
        "tiny",
        vocab_size=257,
        dropout=0.1,
        lif_threshold=0.0,
        spike_embedding=False,
        gradient_checkpointing=True,
    )

    assert set(SPIKEGPT_PRESETS) == {"micro", "tiny", "small", "base", "gpt2-216m"}
    assert config.vocab_size == 257
    assert config.context_length == SPIKEGPT_PRESETS["tiny"].context_length
    assert config.n_layer == SPIKEGPT_PRESETS["tiny"].n_layer
    assert config.n_embd == SPIKEGPT_PRESETS["tiny"].n_embd
    assert config.dropout == 0.1
    assert config.lif_threshold == 0.0
    assert not config.spike_embedding
    assert config.gradient_checkpointing


def test_spikegpt_config_serialization_round_trips() -> None:
    config = SpikeGPTConfig(
        vocab_size=257,
        context_length=16,
        n_layer=2,
        n_embd=32,
        dropout=0.1,
        model_type="rwkv-ffn-pre",
        lif_threshold=0.0,
        spike_embedding=False,
        gradient_checkpointing=True,
    )

    restored = spikegpt_config_from_dict(spikegpt_config_to_dict(config))

    assert restored == config


def test_spikegpt_config_deserialization_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        spikegpt_config_from_dict(
            {
                "vocab_size": 8,
                "context_length": 4,
                "model_type": "attention",
            }
        )


def test_split_token_sequence_and_sample_batch() -> None:
    tokens = torch.arange(20)

    train_tokens, val_tokens = split_token_sequence(
        tokens,
        validation_fraction=0.25,
        min_validation_tokens=3,
    )
    inputs, targets = sample_token_batch(
        train_tokens,
        batch_size=4,
        context_length=3,
        device="cpu",
    )

    assert train_tokens.tolist() == list(range(15))
    assert val_tokens.tolist() == list(range(15, 20))
    assert inputs.shape == (4, 3)
    assert targets.shape == (4, 3)


def test_split_train_val_test_holds_out_test_tail_first() -> None:
    # 100 tokens -> test = last 10 (held out first), then val = last 5 of the
    # remaining 90, train = first 85. The three splits are contiguous, disjoint,
    # and cover the whole sequence with the test tail never overlapping val/train.
    tokens = torch.arange(100)

    train_tokens, val_tokens, test_tokens = split_train_val_test(
        tokens,
        validation_fraction=0.0,
        min_validation_tokens=5,
        test_tokens=10,
    )

    assert train_tokens.tolist() == list(range(85))
    assert val_tokens.tolist() == list(range(85, 90))
    assert test_tokens.tolist() == list(range(90, 100))
    # The held-out test tail is disjoint from both train and val.
    assert set(test_tokens.tolist()).isdisjoint(train_tokens.tolist())
    assert set(test_tokens.tolist()).isdisjoint(val_tokens.tolist())


def test_split_train_val_test_without_test_matches_two_way_split() -> None:
    # test_fraction=0 reproduces split_token_sequence (val IS the tail) and
    # returns an empty test tensor, so the new path is backward compatible.
    tokens = torch.arange(20)

    train_tokens, val_tokens, test_tokens = split_train_val_test(
        tokens,
        validation_fraction=0.25,
        min_validation_tokens=3,
    )
    ref_train, ref_val = split_token_sequence(
        tokens, validation_fraction=0.25, min_validation_tokens=3
    )

    assert train_tokens.tolist() == ref_train.tolist()
    assert val_tokens.tolist() == ref_val.tolist()
    assert test_tokens.numel() == 0


def test_spiking_sequence_lif_is_binary_in_hard_forward_and_has_gradients() -> None:
    inputs = torch.full((2, 3, 4), 1.2, requires_grad=True)
    lif = SpikingSequenceLIF(tau=2.0, threshold=1.0, surrogate_slope=2.0)

    spikes = lif(inputs)
    spikes.sum().backward()

    assert spikes.shape == inputs.shape
    assert set(spikes.detach().flatten().tolist()) <= {0.0, 1.0}
    assert inputs.grad is not None


def test_spike_language_model_returns_logits_loss_and_spike_rates() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=11,
        context_length=8,
        n_layer=2,
        n_embd=16,
        dropout=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))

    logits = model(input_ids)
    loss, logits_with_loss = model(input_ids, targets)
    loss.backward()
    rates = model.spike_rates(input_ids)

    assert logits.shape == (3, config.context_length, config.vocab_size)
    assert logits_with_loss.shape == logits.shape
    assert loss.ndim == 0
    assert model.embedding.weight.grad is not None
    assert "embedding" in rates
    assert "blocks.0.time" in rates
    assert "blocks.0.channel" in rates


def test_spike_language_model_ffn_pre_variant_runs_and_backpropagates() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=11,
        context_length=8,
        n_layer=2,
        n_embd=16,
        dropout=0.0,
        model_type="rwkv-ffn-pre",
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))

    loss, logits = model(input_ids, targets)
    loss.backward()
    first_state = model.initial_state(batch_size=input_ids.shape[0]).blocks[0]

    assert logits.shape == (3, config.context_length, config.vocab_size)
    assert loss.ndim == 0
    assert model.embedding.weight.grad is not None
    assert first_state.time_mix is None
    assert first_state.ffn_pre is not None


def test_spike_language_checkpoint_round_trips_model_vocab_and_metadata(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    vocabulary = CharacterVocabulary.from_text("banana")
    config = SpikeGPTConfig(
        vocab_size=vocabulary.size,
        context_length=4,
        n_layer=1,
        n_embd=8,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = vocabulary.encode("bana").unsqueeze(0)
    targets = vocabulary.encode("anan").unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss, _logits = model(input_ids, targets)
    loss.backward()
    optimizer.step()
    expected_logits = model(input_ids)
    path = tmp_path / "spikegpt.pt"

    save_spike_language_checkpoint(
        path,
        model,
        vocabulary,
        metadata={"steps": 3, "note": "unit"},
        optimizer=optimizer,
    )
    checkpoint = load_spike_language_checkpoint(path, map_location="cpu")
    actual_logits = checkpoint.model(input_ids)

    assert checkpoint.model.config == config
    assert isinstance(checkpoint.vocabulary, CharacterVocabulary)
    assert checkpoint.vocabulary.tokens == vocabulary.tokens
    assert checkpoint.metadata == {"steps": 3, "note": "unit"}
    assert checkpoint.optimizer_state_dict is not None
    assert checkpoint.optimizer_state_dict["state"]
    assert torch.allclose(actual_logits, expected_logits)


def test_spike_language_checkpoint_without_optimizer_loads_as_model_only(
    tmp_path: Path,
) -> None:
    vocabulary = CharacterVocabulary.from_text("banana")
    model = SpikeLanguageModel(
        SpikeGPTConfig(
            vocab_size=vocabulary.size,
            context_length=4,
            n_layer=1,
            n_embd=8,
            dropout=0.0,
        )
    )
    path = tmp_path / "model_only.pt"

    save_spike_language_checkpoint(path, model, vocabulary)
    checkpoint = load_spike_language_checkpoint(path, map_location="cpu")

    assert checkpoint.optimizer_state_dict is None
    assert checkpoint.metadata == {}


def test_spike_language_model_gradient_checkpointing_preserves_loss_and_gradients() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=11,
        context_length=8,
        n_layer=2,
        n_embd=16,
        dropout=0.0,
        lif_threshold=0.0,
    )
    reference = SpikeLanguageModel(config)
    checkpointed = SpikeLanguageModel(config)
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.set_gradient_checkpointing(True)
    input_ids = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))

    reference_loss, _reference_logits = reference(input_ids, targets)
    checkpointed_loss, _checkpointed_logits = checkpointed(input_ids, targets)
    reference_loss.backward()
    checkpointed_loss.backward()

    assert torch.allclose(reference_loss, checkpointed_loss)
    for reference_parameter, checkpointed_parameter in zip(
        reference.parameters(),
        checkpointed.parameters(),
        strict=True,
    ):
        assert reference_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.allclose(reference_parameter.grad, checkpointed_parameter.grad)


def test_spike_language_model_generate_extends_context_and_restores_training() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7,
        context_length=4,
        n_layer=1,
        n_embd=8,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    model.train()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])

    generated = model.generate(input_ids, max_new_tokens=3, sampling="greedy")

    assert generated.shape == (1, 9)
    assert torch.equal(generated[:, : input_ids.shape[1]], input_ids)
    assert model.training


def test_cached_forward_step_matches_full_sequence_logits() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=9,
        context_length=6,
        n_layer=2,
        n_embd=12,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))

    full_logits = model(input_ids)
    state = model.initial_state(batch_size=input_ids.shape[0])
    step_logits = []
    for step in range(input_ids.shape[1]):
        logits, state = model.forward_step(input_ids[:, step], state)
        step_logits.append(logits)

    assert isinstance(full_logits, torch.Tensor)
    assert torch.allclose(torch.stack(step_logits, dim=1), full_logits, atol=1e-6)


def test_cached_and_uncached_greedy_generation_match_within_context_window() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7,
        context_length=8,
        n_layer=1,
        n_embd=8,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.tensor([[0, 1, 2]])

    cached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=False)

    assert torch.equal(cached, uncached)


def test_generate_stream_matches_cached_generate_greedy() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7, context_length=8, n_layer=1, n_embd=8, dropout=0.0, lif_threshold=0.0
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.tensor([[0, 1, 2]])

    full = model.generate(input_ids, max_new_tokens=4, sampling="greedy", use_cache=True)
    model.eval()
    streamed = list(model.generate_stream(input_ids, max_new_tokens=4, sampling="greedy"))

    assert len(streamed) == 4
    assert all(tok.shape == (1, 1) for tok in streamed)
    assert torch.equal(torch.cat([input_ids, *streamed], dim=1), full)
    # generate_stream restores the prior mode (eval here) after iterating.
    assert model.training is False


def test_ffn_pre_cached_and_uncached_greedy_generation_match() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7,
        context_length=8,
        n_layer=2,
        n_embd=8,
        dropout=0.0,
        model_type="rwkv-ffn-pre",
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.tensor([[0, 1, 2]])

    cached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=False)

    assert torch.equal(cached, uncached)


def test_evaluate_language_model_reports_loss_bpc_and_restores_training() -> None:
    torch.manual_seed(0)
    model = SpikeLanguageModel(
        SpikeGPTConfig(
            vocab_size=8,
            context_length=4,
            n_layer=1,
            n_embd=8,
            dropout=0.0,
            lif_threshold=0.0,
        )
    )
    model.train()
    tokens = torch.randint(0, 8, (32,))

    metrics = evaluate_language_model(
        model,
        tokens,
        batch_size=2,
        context_length=4,
        device="cpu",
        batches=2,
    )

    assert metrics.loss > 0
    assert metrics.bits_per_character > 0
    assert metrics.perplexity > 1
    assert model.training


def test_evaluate_language_model_restores_training_after_error() -> None:
    model = SpikeLanguageModel(
        SpikeGPTConfig(
            vocab_size=8,
            context_length=4,
            n_layer=1,
            n_embd=8,
            dropout=0.0,
        )
    )
    model.train()

    with pytest.raises(ValueError, match="too short"):
        evaluate_language_model(
            model,
            torch.arange(4),
            batch_size=2,
            context_length=4,
            device="cpu",
            batches=1,
        )

    assert model.training


def test_evaluate_language_model_strided_is_deterministic_and_restores_training() -> None:
    torch.manual_seed(0)
    model = SpikeLanguageModel(
        SpikeGPTConfig(
            vocab_size=8,
            context_length=4,
            n_layer=1,
            n_embd=8,
            dropout=0.0,
            lif_threshold=0.0,
        )
    )
    model.train()
    tokens = torch.arange(20) % 8

    first = evaluate_language_model_strided(
        model,
        tokens,
        batch_size=2,
        context_length=4,
        device="cpu",
    )
    second = evaluate_language_model_strided(
        model,
        tokens,
        batch_size=3,
        context_length=4,
        device="cpu",
    )

    assert first.loss == pytest.approx(second.loss)
    assert first.bits_per_character == pytest.approx(second.bits_per_character)
    assert first.perplexity == pytest.approx(second.perplexity)
    assert first.loss > 0
    assert model.training


def test_evaluate_language_model_strided_count_last_matches_manual() -> None:
    """count_last must score exactly the intended (token, position) pairs and be
    invariant to batch size; it should differ from all-position scoring."""
    torch.manual_seed(0)
    model = SpikeLanguageModel(
        SpikeGPTConfig(vocab_size=8, context_length=4, n_layer=1, n_embd=8, dropout=0.0)
    ).eval()
    tokens = torch.arange(20) % 8

    full = evaluate_language_model_strided(
        model, tokens, batch_size=2, context_length=4, device="cpu"
    )
    last2_a = evaluate_language_model_strided(
        model, tokens, batch_size=2, context_length=4, device="cpu", stride=2, count_last=2
    )
    last2_b = evaluate_language_model_strided(
        model, tokens, batch_size=5, context_length=4, device="cpu", stride=2, count_last=2
    )

    # batch-size invariant
    assert last2_a.loss == pytest.approx(last2_b.loss)
    # full-context scoring differs from all-position scoring (it drops low-context heads)
    assert last2_a.loss != pytest.approx(full.loss)

    # manual: for stride==count_last==2, counted targets tile the corpus; first
    # window counts all 4 positions, later windows only their last 2.
    with torch.no_grad():
        starts = list(range(0, tokens.numel() - 4, 2))
        total, n = 0.0, 0
        for s in starts:
            inp = tokens[s : s + 4].unsqueeze(0)
            tgt = tokens[s + 1 : s + 5]
            logp = torch.log_softmax(model(inp)[0], dim=-1)
            lo = 0 if s == 0 else 4 - 2
            for pos in range(lo, 4):
                total += -float(logp[pos, int(tgt[pos])])
                n += 1
    assert last2_a.loss == pytest.approx(total / n, rel=1e-5)


def test_evaluate_language_model_strided_rejects_bad_count_last() -> None:
    model = SpikeLanguageModel(SpikeGPTConfig(vocab_size=8, context_length=4, n_layer=1, n_embd=8))
    tokens = torch.arange(20) % 8
    with pytest.raises(ValueError, match="count_last"):
        evaluate_language_model_strided(
            model, tokens, batch_size=2, context_length=4, device="cpu", count_last=5
        )


def test_spike_language_model_rejects_context_overflow() -> None:
    model = SpikeLanguageModel(SpikeGPTConfig(vocab_size=5, context_length=4))

    with pytest.raises(ValueError, match="exceeds context_length"):
        model(torch.zeros((1, 5), dtype=torch.long))
