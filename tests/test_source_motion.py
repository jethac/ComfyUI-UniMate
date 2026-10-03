from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import parse_glb, prepare_document, animate_document
from unimate_pack.contracts import make_asset, make_rig, encode_arrays, decode_arrays
from unimate_pack.source_motion import extract_motion


@pytest.mark.parametrize("body_axis", [False, True])
def test_prepare_four_joint_facing_retains_extraction_mode(body_axis):
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    cond, mapping = prepare_document(document, "joint_pair", "Joint_2", "Joint_1",
        left_shoulder="Joint_4", right_shoulder="Joint_3", body_axis=body_axis)
    assert cond["face_joint_idxs"].shape == (4,)
    rig = make_rig(make_asset(source, "rig.glb"), encode_arrays(**cond), mapping)
    assert rig["mapping"]["body_axis"] == body_axis


def test_joint_pair_extraction_facing_matches_upstream():
    import importlib.util
    import os
    from scipy.spatial.transform import Rotation
    from unimate_pack.clip_sampling import sample_clip
    root = os.environ.get("UNIMATE_REFERENCE")
    motion_reference = os.environ.get("UNIMATE_MOTION_REFERENCE")
    if not root or not motion_reference:
        pytest.skip("Requires pinned upstream and local Motion reference")
    sys.path.insert(0, motion_reference)
    spec = importlib.util.spec_from_file_location("source_facing_reference",
        Path(root) / "data_process/utils/skeleton.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    cond, mapping = prepare_document(document, "joint_pair", "Joint_2", "Joint_1")
    features = np.zeros((17, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    heading = Rotation.from_euler("y", np.linspace(0, 0.8, 17)[:, None]).as_matrix()
    features[:, 0, 3:9] = heading[..., :2].swapaxes(-1, -2).reshape(17, 6)
    animated = animate_document(source, cond, mapping, features)
    rig = make_rig(make_asset(animated, "turn.glb"), encode_arrays(**cond), mapping)
    extracted = decode_arrays(extract_motion(rig)["features"])["features"]
    _, worlds, _ = sample_clip(animated)
    transform = np.asarray(mapping["source_to_canonical"])
    positions = worlds[:, mapping["joint_indices"]][..., :3, 3] @ transform[:3, :3].T + transform[:3, 3]
    expected = upstream.get_root_facing_quat(positions, cond["face_joint_idxs"])
    np.testing.assert_allclose(extracted[:, 0, 3:9], expected.rotation_matrix(cont6d=True)[:-1], atol=2e-6)


@pytest.mark.parametrize("facing", ["+Z", "+X"])
def test_extract_source_animation_preserves_reference_root_trajectory(facing):
    source = synthetic_glb(True)
    doc, _ = parse_glb(source)
    cond, mapping = prepare_document(doc, facing)
    values = np.zeros((61, 7, 12), np.float32)
    values[..., 3] = values[..., 7] = 1
    values[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    values[:, 0, 9] = 0.01
    animated = animate_document(source, cond, mapping, values)
    rig = make_rig(make_asset(animated, "animated.glb"), encode_arrays(**cond), mapping)
    motion = extract_motion(rig, 0)
    features = decode_arrays(motion["features"])["features"]
    assert features.shape == (60, 7, 12)
    np.testing.assert_allclose(features[:, 0, 9], 0.01, atol=2e-6)
    np.testing.assert_allclose(features[:, 0, 1], values[:-1, 0, 1], atol=2e-6)
    assert motion["rig_id"] == rig["rig_id"]


def test_source_root_origin_survives_extract_and_export():
    from test_blender_math import evaluated_vertices
    source = synthetic_glb(True)
    doc, _ = parse_glb(source)
    cond, mapping = prepare_document(doc, "+Z")
    values = np.zeros((8, 7, 12), np.float32)
    values[..., 3] = values[..., 7] = 1
    values[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    values[:, 0, 9] = 0.01
    origin = [0.4, 0, -0.2]
    animated = animate_document(source, cond, mapping, values, root_origin=origin)
    rig = make_rig(make_asset(animated, "clip.glb"), encode_arrays(**cond), mapping)
    motion = extract_motion(rig)
    restored = animate_document(animated, cond, mapping,
        decode_arrays(motion["features"])["features"], root_origin=motion["metadata"]["canonical_root_origin"])
    for frame in range(7):
        np.testing.assert_allclose(evaluated_vertices(restored, frame), evaluated_vertices(animated, frame), atol=2e-6)
