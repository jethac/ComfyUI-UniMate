from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import parse_glb, prepare_document, animate_document
from unimate_pack.contracts import make_asset, make_rig, encode_arrays, decode_arrays
from unimate_pack.source_motion import extract_motion


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
