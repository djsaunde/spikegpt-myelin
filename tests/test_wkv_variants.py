"""Correctness of the loop-free WKV variants vs the reference recurrence."""

from __future__ import annotations

import pytest
import torch

from spikegpt.benchmarks.spikegpt_wkv_compare import wkv_chunked, wkv_parallel
from spikegpt.language import weighted_key_value, weighted_key_value_loop

# The sequential loop is the correctness oracle; every loop-free variant
# (including the default associative-scan `weighted_key_value`) is validated
# against it.
wkv_reference = weighted_key_value_loop

VARIANTS = {
    "associative": weighted_key_value,
    "parallel": wkv_parallel,
    "chunked8": lambda *a: wkv_chunked(*a, chunk_size=8),
    "chunked16": lambda *a: wkv_chunked(*a, chunk_size=16),
}


def _inputs(batch, t, channels, *, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(batch, t, channels, generator=g, dtype=dtype)
    value = torch.randn(batch, t, channels, generator=g, dtype=dtype)
    time_decay = torch.randn(channels, generator=g, dtype=dtype)
    time_first = torch.randn(channels, generator=g, dtype=dtype)
    return key, value, time_decay, time_first


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("t", [1, 4, 16, 17, 40])
def test_wkv_variant_forward_matches_reference(name: str, t: int) -> None:
    key, value, time_decay, time_first = _inputs(3, t, 8)
    ref = wkv_reference(key, value, time_decay, time_first)
    got = VARIANTS[name](key, value, time_decay, time_first)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_wkv_variant_backward_matches_reference(name: str) -> None:
    def grads(fn) -> list[torch.Tensor]:
        key, value, time_decay, time_first = _inputs(2, 24, 8, seed=1)
        tensors = [x.clone().requires_grad_(True) for x in (key, value, time_decay, time_first)]
        fn(*tensors).square().sum().backward()
        out = [x.grad for x in tensors]
        assert all(g is not None for g in out)
        return [g for g in out if g is not None]

    ref_grads = grads(wkv_reference)
    got_grads = grads(VARIANTS[name])
    for got, ref in zip(got_grads, ref_grads, strict=True):
        assert torch.allclose(got, ref, atol=1e-9, rtol=1e-6)


def test_wkv_parallel_handles_long_context_float32_without_nan() -> None:
    key, value, time_decay, time_first = _inputs(2, 256, 16, dtype=torch.float32)
    out = wkv_parallel(key, value, time_decay, time_first)
    assert torch.isfinite(out).all()
