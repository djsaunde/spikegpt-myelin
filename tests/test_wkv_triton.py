"""Parity tests for the fused Triton WKV kernel (CUDA only).

The Triton forward/backward must match the eager loop oracle
``weighted_key_value_loop`` (whose autograd is exact) to float32 precision.
"""

from __future__ import annotations

import pytest
import torch
from myelin._optional import has_triton

from spikegpt.language import weighted_key_value_loop

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="Triton WKV tests require CUDA and Triton",
)


def _grads(fn, k0, v0, td0, tf0, gy):
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    td = td0.clone().requires_grad_(True)
    tf = tf0.clone().requires_grad_(True)
    y = fn(k, v, td, tf)
    y.backward(gy)

    def z(grad, ref):  # an unused input (e.g. time_decay at T=1) yields None
        return torch.zeros_like(ref) if grad is None else grad

    return y, z(k.grad, k), z(v.grad, v), z(td.grad, td), z(tf.grad, tf)


@pytest.mark.parametrize("t", [1, 7, 64])
def test_triton_wkv_forward_and_backward_match_loop(t: int) -> None:
    from spikegpt.wkv_triton import weighted_key_value_triton

    torch.manual_seed(t)
    b, c = 3, 96
    k0 = torch.randn(b, t, c, device="cuda")
    v0 = torch.randn(b, t, c, device="cuda")
    td0 = torch.randn(c, device="cuda")
    tf0 = torch.randn(c, device="cuda")
    gy = torch.randn(b, t, c, device="cuda")

    yr, gkr, gvr, gtdr, gtfr = _grads(weighted_key_value_loop, k0, v0, td0, tf0, gy)
    yt, gkt, gvt, gtdt, gtft = _grads(weighted_key_value_triton, k0, v0, td0, tf0, gy)

    assert torch.allclose(yt, yr, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gkt, gkr, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gvt, gvr, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gtdt, gtdr, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gtft, gtfr, atol=1e-4, rtol=1e-4)


def test_triton_wkv_handles_channels_not_multiple_of_block() -> None:
    from spikegpt.wkv_triton import weighted_key_value_triton

    torch.manual_seed(0)
    b, t, c = 2, 16, 70  # 70 not a multiple of the 64-wide block
    k0 = torch.randn(b, t, c, device="cuda")
    v0 = torch.randn(b, t, c, device="cuda")
    td0 = torch.randn(c, device="cuda")
    tf0 = torch.randn(c, device="cuda")
    ref = weighted_key_value_loop(k0, v0, td0, tf0)
    got = weighted_key_value_triton(k0, v0, td0, tf0)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
