"""Released configs from Linzhan/UniMate revision 971da7c (MIT)."""

import json
from pathlib import Path

import pytest
import numpy as np

from unimate_pack.upstream import validate_config, create_denoiser
from unimate_pack.upstream import build_condition


@pytest.mark.parametrize("path", sorted((Path(__file__).parent / "fixtures/model_configs").glob("*.json")))
def test_released_config_reconstructs_correct_backbone(monkeypatch, path):
    from unimate_pack._vendor import denoiser
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    axes = (config["model"]["attention"], config["model"]["text_cond"])
    name = {("graph", "adaln"): "UniMateGraphAdaLN",
            ("full", "cross_attn"): "UniMateFullCrossAttn"}[axes]
    monkeypatch.setattr(denoiser, name, lambda **kwargs: kwargs)
    kwargs = create_denoiser(config)
    assert kwargs["max_joints"] == config["dataset"]["max_joints"]
    assert kwargs["max_depth"] == config["dataset"]["max_depth"]
    assert kwargs["num_layers"] == config["model"]["num_layers"]
    assert ("use_graph_attn_bias" in kwargs) == (axes[0] == "graph")


def test_unreleased_capacity_combination_is_rejected():
    path = Path(__file__).parent / "fixtures/model_configs/unimate_uniml3d_f60_v2.json"
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    config["dataset"]["max_joints"] = 22
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("path", sorted((Path(__file__).parent / "fixtures/model_configs").glob("*.json")))
def test_conditioning_uses_released_capacity_and_retains_text_tokens(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    arrays = dict(parents=np.array([-1, 0, 0, 1, 2]),
                  tpos_first_frame=np.arange(15).reshape(5, 3) / 10)
    stats = {f"{family}_{kind}_{part}": np.full(12, 0.3 if kind == "mean" else 1.7)
             for family in ("mixamo", "objaverse", "truebones")
             for kind in ("mean", "std") for part in ("root", "local")}
    tokens = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
    cond = build_condition(arrays, config, stats, "mixamo", tokens, np.ones((5, 768), np.float32))
    capacity = config["dataset"]["max_joints"]
    assert cond["mean"].shape == (1, capacity, 12)
    assert cond["joint_mask"].shape == (1, 1, 1, capacity)
    np.testing.assert_array_equal(cond["caption_tokens"][0].numpy(), tokens)
    assert cond["caption_mask"].all()
