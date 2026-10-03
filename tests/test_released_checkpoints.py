"""Opt-in strict loading checks for the remaining official checkpoint families."""

import json
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("family", ["unimate_mixamo_f60", "unimate_uniml3d_f60_preview",
                                    "unimate_uniml3d_f60_v2_full_cross_attn"])
def test_released_checkpoint_matches_upstream_parameter_inventory(family):
    root = os.environ.get("UNIMATE_RELEASED_CHECKPOINTS")
    reference = os.environ.get("UNIMATE_REFERENCE")
    if not root or not reference:
        pytest.skip("Requires explicit installed checkpoints and pinned upstream")
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, reference)
    from unimate.configs.schema import MainConfig
    from unimate.models.factory import create_model
    from unimate_pack.upstream import create_denoiser

    config_path = Path(__file__).parent / "fixtures/model_configs" / f"{family}.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    actual = create_denoiser(config)
    upstream = MainConfig.from_json(config_path)
    expected = create_model(upstream.dataset, upstream.model)
    assert list(actual.state_dict()) == list(expected.state_dict())
    assert [p.shape for p in actual.parameters()] == [p.shape for p in expected.parameters()]
    checkpoint = torch.load(Path(root) / family / "checkpoints/checkpoint_step_100000.pt",
                            map_location="cpu", weights_only=True)
    actual.load_state_dict(checkpoint["model_state_dict"], strict=True)
    shadows = checkpoint["ema_state_dict"]["shadow_params"]
    assert len(shadows) == len(list(actual.parameters()))
    for parameter, shadow in zip(actual.parameters(), shadows):
        assert parameter.shape == shadow.shape
        assert torch.isfinite(shadow).all()
