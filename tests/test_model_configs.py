"""Released configs from Linzhan/UniMate revision 971da7c (MIT)."""

import json
from pathlib import Path

import pytest

from unimate_pack.upstream import validate_config, create_denoiser


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
