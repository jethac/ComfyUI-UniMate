"""Constrained flow integration must preserve reference tokens during sampling."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from unimate_pack.constrained import sample_replacement
from unimate_pack.upstream import sample_constrained_flow


@pytest.mark.parametrize("axis", ["frames", "joints"])
def test_replacement_matches_pinned_upstream_euler(axis):
    root = os.environ.get("UNIMATE_REFERENCE")
    if not root:
        pytest.skip("Set UNIMATE_REFERENCE to the pinned upstream checkout")
    spec = importlib.util.spec_from_file_location(
        "reference_inbetween", Path(root) / "unimate/inference/motion_inbetweening.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    shape = (1, 5, 12, 7)
    known = torch.arange(420, dtype=torch.float32).reshape(shape) / 100
    mask = torch.zeros((1, 1, 1, 7) if axis == "frames" else (1, 5, 1, 1), dtype=torch.bool)
    if axis == "frames":
        mask[..., 0] = mask[..., -1] = True
    else:
        mask[:, 1] = True

    class Transport:
        train_eps = sample_eps = 0

        def check_interval(self, *args, **kwargs):
            return 0, 1

    def velocity(x, t, cond=None):
        return x * 0.03 + t[:, None, None, None] + cond["offset"]

    torch.manual_seed(73)
    expected = reference.inbetween_sample_ode(
        velocity, Transport(), {"offset": 0.2}, known, mask, shape,
        device=torch.device("cpu"),
    )
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(73))
    actual = sample_replacement(velocity, {"offset": 0.2}, known, mask, noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expanded = mask.expand(shape)
    torch.testing.assert_close(actual[expanded], known[expanded], rtol=0, atol=0)


def test_cancellation_stops_before_next_model_call():
    calls = []

    def check():
        if len(calls) == 2:
            raise RuntimeError("cancelled")

    def model(x, t, cond=None):
        calls.append(float(t[0]))
        return torch.ones_like(x)

    values = torch.zeros((1, 5, 12, 3))
    with pytest.raises(RuntimeError, match="cancelled"):
        sample_replacement(model, {}, values, torch.zeros_like(values, dtype=torch.bool),
                           values, check_cancel=check)
    assert len(calls) == 2


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_step_counts_are_rejected(steps):
    values = torch.zeros((1, 5, 12, 3))
    with pytest.raises(ValueError):
        sample_replacement(lambda x, t, cond=None: x, {}, values,
                           torch.zeros_like(values, dtype=torch.bool), values, steps=steps)


def test_guided_constrained_flow_uses_local_seed_and_preserves_known_values():
    def model(x, t, cond=None, force_mask=False):
        return torch.ones_like(x) * (0.1 if force_mask else 0.2)

    shape = (1, 5, 12, 7)
    known = torch.zeros(shape)
    mask = torch.zeros((1, 1, 1, 7), dtype=torch.bool)
    mask[..., 0] = True
    rng = torch.random.get_rng_state().clone()
    actual = sample_constrained_flow(model, {}, known, mask, 73, 3, lambda: None)
    assert torch.equal(torch.random.get_rng_state(), rng)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(73))
    expected = sample_replacement(
        lambda x, t, cond=None: torch.ones_like(x) * (0.1 + 3 * (0.2 - 0.1)),
        {}, known, mask, noise,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
