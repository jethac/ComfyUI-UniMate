from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from unimate_pack import contracts, inference, upstream
from unimate_pack.contracts import encode_arrays, make_motion, decode_arrays


@pytest.mark.parametrize("mode,selection", [("inbetween", "0,-1"), ("edit", "joint1")])
def test_constrained_inference_normalizes_reference_and_preserves_selected_features(monkeypatch, mode, selection):
    from comfy import model_management as mm
    joints, frames = 5, 7
    names = np.asarray([f"joint{i}" for i in range(joints)])
    rig = {"rig_id": "a" * 64, "conditioning": encode_arrays(
        parents=np.array([-1, 0, 1, 2, 3]), joint_names=names,
    )}
    raw = np.arange(frames * joints * 12, dtype=np.float32).reshape(frames, joints, 12) / 17
    reference = make_motion(rig["rig_id"], encode_arrays(features=raw), {})
    monkeypatch.setattr(contracts, "validate_model", lambda value: None)
    monkeypatch.setattr(contracts, "validate_rig", lambda value: None)
    monkeypatch.setattr(mm, "load_models_gpu", lambda *a, **kw: None)
    cond = {"mean": torch.ones((1, 71, 12)) * 0.3, "std": torch.ones((1, 71, 12)) * 1.7,
            "motion_length": torch.tensor([60]), "lengths_mask": torch.ones((1, 1, 1, 60), dtype=torch.bool)}
    monkeypatch.setattr(upstream, "build_condition", lambda *a: cond)
    runtime = SimpleNamespace(
        encode=lambda texts: ([np.ones((2, 768), np.float32) for _ in texts], np.ones((len(texts), 768))),
        config={}, stats={}, denoiser=SimpleNamespace(load_device=torch.device("cpu"), model=lambda x,t,cond=None,force_mask=False: x*0.01),
        manifest={"solver": {}, "model_revision": "fixture", "text_encoder": {}, "upstream_revision": "fixture"},
    )
    monkeypatch.setattr(inference, "_get_runtime", lambda model: runtime)
    result = inference.generate_motion({"sha256": "b" * 64}, rig, "walk", 0, 3,
        reference=reference, constraint_mode=mode, selection=selection)
    actual = decode_arrays(result["features"])["features"]
    assert actual.shape == raw.shape
    if mode == "inbetween":
        np.testing.assert_array_equal(actual[[0, -1]], raw[[0, -1]])
    else:
        np.testing.assert_array_equal(actual[:, 1], raw[:, 1])
    assert result["metadata"]["solver"]["method"] == "euler"
    assert cond["motion_length"].tolist() == [frames]
    assert cond["lengths_mask"].shape == (1, 1, 1, 60)
    assert cond["lengths_mask"].reshape(-1).tolist() == [True] * frames + [False] * (60 - frames)
