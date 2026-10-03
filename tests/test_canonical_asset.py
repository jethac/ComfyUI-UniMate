import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from test_blender_math import evaluated_vertices
from unimate_pack.rig_math import parse_glb, prepare_document, world_matrices
from unimate_pack.contracts import make_asset, make_rig, encode_arrays
from unimate_pack.canonical_asset import canonical_asset


def test_canonical_asset_preserves_binary_and_skinning_in_prepared_coordinates():
    source = synthetic_glb(True)
    document, binary = parse_glb(source)
    cond, mapping = prepare_document(document, "+X")
    rig = make_rig(make_asset(source, "rig.glb"), encode_arrays(**cond), mapping)
    asset = canonical_asset(rig)
    actual, body = parse_glb(asset["glb"])
    assert body == binary
    for key in ("meshes", "skins", "materials", "images", "textures", "accessors", "bufferViews"):
        assert actual[key] == document[key]
    worlds, _, _ = world_matrices(actual)
    np.testing.assert_allclose(worlds[mapping["joint_indices"], :3, 3], cond["tpos_first_frame"], atol=2e-6)
    transform = np.asarray(mapping["source_to_canonical"])
    expected = evaluated_vertices(source) @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(evaluated_vertices(asset["glb"]), expected, atol=2e-6)
