import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import (
    prepare_document,
    decode_features,
    parse_glb,
    animate_document,
)


@pytest.mark.parametrize("branching", [False, True])
def test_identity_preserves_source(branching):
    source = synthetic_glb(branching)
    doc, binary = parse_glb(source)
    cond, mapping = prepare_document(doc, "+Z")
    j = len(cond["parents"])
    features = np.zeros((60, j, 12), dtype=np.float32)
    features[:, :, 3] = 1
    features[:, :, 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    out = animate_document(source, cond, mapping, features)
    got, body = parse_glb(out)
    assert body[: len(binary)] == binary
    for field in ("skins", "meshes", "materials", "nodes", "images", "textures"):
        assert got[field] == doc[field]
    assert len(got["animations"]) == 1
    assert len(got["animations"][0]["channels"]) == j + 1
    rotations, root = decode_features(features, cond["parents"])
    np.testing.assert_allclose(rotations, np.broadcast_to(np.eye(3), rotations.shape))
    np.testing.assert_allclose(
        root, np.broadcast_to(cond["tpos_first_frame"][0], root.shape), atol=1e-6
    )


def test_destination_facing_velocity_integration():
    features = np.zeros((60, 5, 12))
    features[:, :, 3] = 1
    features[:, :, 7] = 1
    features[1:, 0, 3:9] = [0, 0, -1, 0, 1, 0]
    features[:, 0, 9] = 0.1
    _, root = decode_features(features, [-1, 0, 1, 2, 3])
    np.testing.assert_allclose(root[-1], [0, 0, 5.9], atol=1e-6)


def read_accessor(doc, binary, index):
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    dtype = {5126: "<f4", 5123: "<u2"}[accessor["componentType"]]
    return np.frombuffer(
        binary,
        dtype=dtype,
        count=accessor["count"] * width,
        offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
    ).reshape(accessor["count"], width)


def evaluated_vertices(glb, frame=None):
    import copy
    from unimate_pack.rig_math import world_matrices

    doc, binary = parse_glb(glb)
    doc = copy.deepcopy(doc)
    if frame is not None:
        anim = doc["animations"][0]
        for channel in anim["channels"]:
            sampler = anim["samplers"][channel["sampler"]]
            values = read_accessor(doc, binary, sampler["output"])
            doc["nodes"][channel["target"]["node"]][channel["target"]["path"]] = values[
                frame
            ].tolist()
    world, _, _ = world_matrices(doc)
    skin = doc["skins"][0]
    inverse = (
        read_accessor(doc, binary, skin["inverseBindMatrices"])
        .reshape(-1, 4, 4)
        .transpose(0, 2, 1)
    )
    primitive = doc["meshes"][0]["primitives"][0]
    positions = read_accessor(doc, binary, primitive["attributes"]["POSITION"])
    joints = read_accessor(doc, binary, primitive["attributes"]["JOINTS_0"]).astype(int)
    weights = read_accessor(doc, binary, primitive["attributes"]["WEIGHTS_0"])
    vertices = []
    for p, js, ws in zip(positions, joints, weights):
        vertices.append(
            sum(
                w * (world[skin["joints"][j]] @ inverse[j] @ np.array([*p, 1]))[:3]
                for j, w in zip(js, ws)
            )
        )
    return np.asarray(vertices)


@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize("facing", ["+Z", "-Z", "+X", "-X"])
def test_independent_skinning_identity_and_root_displacement(branching, facing):
    source = synthetic_glb(branching)
    doc, _ = parse_glb(source)
    cond, mapping = prepare_document(doc, facing)
    features = np.zeros((60, len(cond["parents"]), 12))
    features[:, :, 3] = 1
    features[:, :, 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    output = animate_document(source, cond, mapping, features)
    before = evaluated_vertices(source)
    for frame in range(60):
        np.testing.assert_allclose(evaluated_vertices(output, frame), before, atol=2e-6)
    features[:, 0, 9] = 0.01
    output = animate_document(source, cond, mapping, features)
    shift = np.linalg.inv(np.asarray(mapping["source_to_canonical"]))[
        :3, :3
    ] @ np.array([0.59, 0, 0])
    np.testing.assert_allclose(
        evaluated_vertices(output, 59) - before,
        np.broadcast_to(shift, before.shape),
        atol=2e-6,
    )
    document, binary = parse_glb(output)
    animation = document["animations"][0]
    for sampler in animation["samplers"]:
        np.testing.assert_allclose(
            read_accessor(document, binary, sampler["input"])[:, 0],
            np.arange(60) / 30,
            atol=1e-7,
        )
        assert sampler["interpolation"] == "LINEAR"
