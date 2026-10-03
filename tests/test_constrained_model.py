"""Opt-in real offline constrained inference and Blender output checks."""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("UNIMATE_TEST_BUNDLE") or not os.environ.get("UNIMATE_BLENDER"),
    reason="Requires installed UniMate bundle and Blender",
)


@pytest.mark.parametrize("guidance", [1, 3])
def test_real_model_free_generation_exports_playable_motion(tmp_path, guidance):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from rig_generator import synthetic_glb
    from test_blender_math import evaluated_vertices
    import torch
    from comfy.cli_args import args
    if not torch.cuda.is_available():
        args.cpu = True
    torch.set_num_threads(4)
    from unimate_pack.blender import prepare_rig, export_glb
    from unimate_pack.contracts import make_asset, decode_arrays
    from unimate_pack.inference import load_model_bundle, generate_motion
    rig = prepare_rig(make_asset(synthetic_glb(True), "branching.glb"), "+Z")
    model = load_model_bundle(os.environ["UNIMATE_TEST_BUNDLE"])
    motion = generate_motion(model, rig, "A character walks forward.", 42, guidance,
        normalization=os.environ.get("UNIMATE_TEST_NORMALIZATION", "objaverse"))
    features = decode_arrays(motion["features"])["features"]
    assert features.shape == (60, 7, 12)
    assert np.isfinite(features).all()
    output = export_glb(rig, motion)
    vertices = np.stack([evaluated_vertices(output, frame) for frame in range(60)])
    assert np.isfinite(vertices).all()
    assert np.any(vertices[1:] != vertices[0])
    (tmp_path / f"guidance-{guidance}.glb").write_bytes(output)


@pytest.mark.parametrize("frames", [7, 60])
def test_real_model_edit_and_inbetween_preserve_constraints_and_export(tmp_path, frames):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from rig_generator import synthetic_glb
    from test_blender_math import evaluated_vertices
    import torch
    from comfy.cli_args import args
    if not torch.cuda.is_available():
        args.cpu = True
    torch.set_num_threads(4)
    from unimate_pack.blender import prepare_rig, export_glb
    from unimate_pack.contracts import make_asset, make_motion, encode_arrays, decode_arrays
    from unimate_pack.inference import load_model_bundle, generate_motion
    from unimate_pack.rig_math import parse_glb
    model = load_model_bundle(os.environ["UNIMATE_TEST_BUNDLE"])
    rig = prepare_rig(make_asset(synthetic_glb(True), "branching.glb"), "+Z")
    conditioning = decode_arrays(rig["conditioning"])
    joints = len(conditioning["parents"])
    features = np.zeros((frames, joints, 12), np.float32)
    features[:, :, 3] = features[:, :, 7] = 1
    features[:, 0, 1] = conditioning["tpos_first_frame"][0, 1]
    reference = make_motion(rig["rig_id"], encode_arrays(features=features), {})
    joint_name = str(conditioning["joint_names"][1])
    for mode, selection in (("inbetween", "0,-1"), ("edit", joint_name)):
        motion = generate_motion(model, rig, "A character walks forward.", 0, 3,
            normalization=os.environ.get("UNIMATE_TEST_NORMALIZATION", "objaverse"),
            reference=reference, constraint_mode=mode, selection=selection)
        actual = decode_arrays(motion["features"])["features"]
        assert actual.shape == features.shape
        assert np.isfinite(actual).all()
        assert np.any(actual != features)
        if mode == "inbetween":
            np.testing.assert_array_equal(actual[[0, -1]], features[[0, -1]])
        else:
            np.testing.assert_array_equal(actual[:, 1], features[:, 1])
        output = export_glb(rig, motion)
        doc, _ = parse_glb(output)
        for sampler in doc["animations"][0]["samplers"]:
            assert doc["accessors"][sampler["output"]]["count"] == frames
        vertices = np.stack([evaluated_vertices(output, frame) for frame in range(frames)])
        assert np.isfinite(vertices).all()
        assert np.any(vertices[1:] != vertices[0])
        (tmp_path / f"{mode}.glb").write_bytes(output)


def test_real_model_expansion_exports_all_frames_and_preserves_source(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from rig_generator import synthetic_glb
    from test_blender_math import evaluated_vertices
    import torch
    from comfy.cli_args import args
    if not torch.cuda.is_available():
        args.cpu = True
    torch.set_num_threads(4)
    from unimate_pack.blender import prepare_rig, export_glb
    from unimate_pack.contracts import make_asset, decode_arrays
    from unimate_pack.inference import load_model_bundle
    from unimate_pack.expansion import expand_motion
    from unimate_pack.rig_math import parse_glb
    source = synthetic_glb(True)
    rig = prepare_rig(make_asset(source, "branching.glb"), "+Z")
    model = load_model_bundle(os.environ["UNIMATE_TEST_BUNDLE"])
    motion = expand_motion(model, rig,
        ["A character stands still.", "A character walks forward."], 0, 3, overlap=10)
    features = decode_arrays(motion["features"])["features"]
    assert features.shape == (110, 7, 12)
    assert np.isfinite(features).all()
    output = export_glb(rig, motion)
    doc, binary = parse_glb(output)
    original, body = parse_glb(source)
    assert binary[:len(body)] == body
    for name in ("meshes", "skins", "materials", "images", "textures"):
        assert doc[name] == original[name]
    for sampler in doc["animations"][0]["samplers"]:
        assert doc["accessors"][sampler["output"]]["count"] == 110
    vertices = np.stack([evaluated_vertices(output, frame) for frame in range(110)])
    assert np.isfinite(vertices).all()
    assert np.any(vertices[1:] != vertices[0])
    (tmp_path / "expansion.glb").write_bytes(output)
