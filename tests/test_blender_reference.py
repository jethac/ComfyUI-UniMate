"""Optional comparisons to the pinned local research checkout (not distributed)."""

import os
from pathlib import Path
import sys
import numpy as np
import pytest
from unimate_pack.rig_math import (
    prepare_document,
    parse_glb,
    world_matrices,
    rotation_part,
    matrix_quaternion,
    decode_features,
)

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb


@pytest.fixture(scope="module")
def reference():
    location = os.environ.get("UNIMATE_REFERENCE")
    motion = os.environ.get("UNIMATE_MOTION_REFERENCE")
    if not location or not motion:
        pytest.skip(
            "Reference comparisons require explicit local upstream + Motion checkout"
        )
    sys.path[:0] = [location, motion]
    from data_process.utils.motion_features import process_tpose, build_topology_cond
    from unimate.utils.motion_utils import recover_unimate_anim_from_rot
    from Animation import positions_global, rotations_global

    return (
        process_tpose,
        build_topology_cond,
        recover_unimate_anim_from_rot,
        positions_global,
        rotations_global,
    )


@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize("pair", [False, True])
def test_conditioning_matches_pinned_reference(reference, branching, pair):
    process, build, _, positions_global, rotations_global = reference
    doc, _ = parse_glb(synthetic_glb(branching))
    world, local, parents = world_matrices(doc)
    joints = doc["skins"][0]["joints"]
    jp = [joints.index(parents[j]) if parents[j] in joints else -1 for j in joints]
    rotations = rotation_part(world[joints])
    pos = world[joints, :3, 3]
    offsets = pos.copy()
    local_rot = rotations.copy()
    for i, p in enumerate(jp):
        if p >= 0:
            offsets[i] = rotations[p].T @ (pos[i] - pos[p])
            local_rot[i] = rotations[p].T @ rotations[i]
    names = [doc["nodes"][j]["name"] for j in joints]
    tpos = dict(
        names=np.array(names),
        parents=np.array(jp),
        rest_local_pos=offsets,
        rest_local_rot=np.array(
            [matrix_quaternion(r)[[3, 0, 1, 2]] for r in local_rot]
        ),
        fps=np.array(30),
    )
    face_joints = (
        {"r_hip": {"raw": names[1]}, "l_hip": {"raw": names[2]}} if pair else None
    )
    anim, off, scale, ground, par, names_bfs, fps, bfs, face, axis = process(
        tpos, face_joints=face_joints
    )
    actual, mapping = prepare_document(
        doc, "joint_pair" if pair else "+Z", left_joint=names[2], right_joint=names[1]
    )
    expected = build(
        "character",
        par,
        off,
        names_bfs,
        actual["clean_joint_names"],
        positions_global(anim)[0],
        anim.rotations.qs[0],
        rotations_global(anim).qs[0],
        face_joint_idxs=face,
        scale_factor=scale,
    )
    for key in [
        "parents",
        "offsets",
        "tpos_offsets",
        "tpos_first_frame",
        "joint_relations",
        "joint_graph_dists",
        "joint_depths",
        "edge_indexs",
        "spectral_feats",
        "scale_factor",
    ]:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-7, err_msg=key)
    for key in ["tpos_local_rotations", "tpos_global_rotations"]:
        a = actual[key]
        b = expected[key]
        np.testing.assert_allclose(
            np.abs(np.sum(a * b, axis=-1)), 1, atol=2e-7, err_msg=key
        )


@pytest.mark.parametrize("branching", [False, True])
def test_motion_recovery_matches_pinned_reference(reference, branching):
    _, _, recover, positions_global, _ = reference
    doc, _ = parse_glb(synthetic_glb(branching))
    cond, _ = prepare_document(doc, "+Z")
    rng = np.random.default_rng(814)
    features = rng.normal(size=(60, len(cond["parents"]), 12))
    features[:, 0, 1] = 1.0
    features[:, :, 9:] *= 0.03
    rotations, root = decode_features(features, cond["parents"])
    expected = recover(features, cond["parents"], cond["tpos_offsets"])
    np.testing.assert_allclose(
        rotations, expected.rotations.rotation_matrix(cont6d=False), atol=2e-7
    )
    np.testing.assert_allclose(root, expected.positions[:, 0], atol=2e-7)
    pos = np.empty((60, len(cond["parents"]), 3))
    glob = rotations.copy()
    pos[:, 0] = root
    for j, p in enumerate(cond["parents"][1:], 1):
        pos[:, j] = pos[:, p] + np.einsum(
            "tij,j->ti", glob[:, p], cond["tpos_offsets"][j]
        )
        glob[:, j] = glob[:, p] @ rotations[:, j]
    np.testing.assert_allclose(pos, positions_global(expected), atol=2e-7)


@pytest.mark.skipif(not os.environ.get("UNIMATE_BLENDER"), reason="Requires Blender")
@pytest.mark.parametrize("branching", [False, True])
def test_blender_rest_matches_pinned_reference(reference, branching, tmp_path):
    from unimate_pack.blender import prepare_rig, blender_executable, _run_process
    from unimate_pack.contracts import make_asset, decode_arrays
    from Animation import positions_global, rotations_global
    from Quaternions import Quaternions

    process, build, *_ = reference
    source = synthetic_glb(branching)
    (tmp_path / "input.glb").write_bytes(source)
    script = Path(__file__).parent / "fixtures" / "blender_reference.py"
    _run_process(
        [
            blender_executable(),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(script.resolve()),
            "--",
            str(tmp_path / "input.glb"),
            os.environ["UNIMATE_REFERENCE"],
            str(tmp_path / "reference.npz"),
        ],
        tmp_path,
    )
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as z:
        data = dict(z)
    # Upstream apply_zup_to_yup rotates only the root; other locals stay unchanged.
    q = Quaternions.from_euler(np.array([[-np.pi / 2, 0, 0]]))
    data["rest_local_pos"][0] = (q * data["rest_local_pos"][0:1])[0]
    data["rest_local_rot"][0] = (q * Quaternions(data["rest_local_rot"][0:1])).qs[0]
    anim, off, scale, ground, par, names, fps, bfs, face, axis = process(data)
    rig = prepare_rig(make_asset(source, "character.glb"), "+Z")
    actual = decode_arrays(rig["conditioning"])
    expected = build(
        "character",
        par,
        off,
        names,
        actual["clean_joint_names"],
        positions_global(anim)[0],
        anim.rotations.qs[0],
        rotations_global(anim).qs[0],
        face_joint_idxs=face,
        scale_factor=scale,
    )
    for key in [
        "parents",
        "offsets",
        "tpos_offsets",
        "tpos_first_frame",
        "joint_relations",
        "joint_graph_dists",
        "joint_depths",
        "edge_indexs",
        "spectral_feats",
        "scale_factor",
    ]:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, err_msg=key)
    for key in ["tpos_local_rotations", "tpos_global_rotations"]:
        np.testing.assert_allclose(
            np.abs(np.sum(actual[key] * expected[key], axis=-1)),
            1,
            atol=2e-6,
            err_msg=key,
        )


@pytest.mark.parametrize("branching", [False, True])
def test_export_fk_matches_pinned_reference(reference, branching):
    import copy
    from unimate_pack.rig_math import animate_document

    sys.path.insert(0, str(Path(__file__).parent))
    from test_blender_math import read_accessor

    _, _, recover, positions_global, _ = reference
    source = synthetic_glb(branching)
    doc, _ = parse_glb(source)
    cond, mapping = prepare_document(doc, "-X")
    rng = np.random.default_rng(933)
    features = rng.normal(size=(60, len(cond["parents"]), 12))
    features[:, :, 9:] *= 0.01
    expected = positions_global(
        recover(features.astype(np.float64), cond["parents"], cond["tpos_offsets"])
    )
    sim = np.asarray(mapping["source_to_canonical"])
    expected = (expected - sim[:3, 3]) @ np.linalg.inv(sim[:3, :3]).T
    out, body = parse_glb(animate_document(source, cond, mapping, features))
    animation = out["animations"][0]
    for frame in range(60):
        posed = copy.deepcopy(out)
        for channel in animation["channels"]:
            values = read_accessor(
                out, body, animation["samplers"][channel["sampler"]]["output"]
            )
            posed["nodes"][channel["target"]["node"]][channel["target"]["path"]] = (
                values[frame].tolist()
            )
        world, _, _ = world_matrices(posed)
        np.testing.assert_allclose(
            world[mapping["joint_indices"], :3, 3], expected[frame], atol=2e-6
        )


@pytest.mark.skipif(not os.environ.get("UNIMATE_BLENDER"), reason="Requires Blender")
@pytest.mark.parametrize(
    "variant", ["rotated_ancestor", "scaled_root", "nonjoint_intermediary"]
)
def test_blender_scene_transform_export_fk(reference, variant):
    import copy
    from unimate_pack.blender import prepare_rig, export_glb
    from unimate_pack.contracts import (
        make_asset,
        make_motion,
        encode_arrays,
        decode_arrays,
    )
    from unimate_pack.rig_math import pack_glb

    sys.path.insert(0, str(Path(__file__).parent))
    from test_blender_math import read_accessor

    _, _, recover, positions_global, _ = reference
    doc, binary = parse_glb(synthetic_glb(False))
    if variant == "rotated_ancestor":
        doc["nodes"][0]["rotation"] = [0, 0, np.sin(0.3), np.cos(0.3)]
    elif variant == "scaled_root":
        doc["nodes"][1]["scale"] = [2, 2, 2]
    else:
        parent = doc["nodes"][1]
        child = parent["children"][0]
        parent["children"] = [len(doc["nodes"])]
        doc["nodes"].append(
            {
                "name": "StaticHelper",
                "translation": [0.1, 0.2, 0.05],
                "rotation": [0, 0, np.sin(0.15), np.cos(0.15)],
                "children": [child],
            }
        )
    source = pack_glb(doc, binary)
    rig = prepare_rig(make_asset(source, "transformed.glb"), "-X")
    cond = decode_arrays(rig["conditioning"])
    mapping = rig["mapping"]
    rng = np.random.default_rng(44)
    features = rng.normal(size=(60, len(cond["parents"]), 12)).astype(np.float32)
    features[:, :, 9:] *= 0.01
    expected = positions_global(
        recover(features.astype(np.float64), cond["parents"], cond["tpos_offsets"])
    )
    output = export_glb(
        rig, make_motion(rig["rig_id"], encode_arrays(features=features), {})
    )
    out, body = parse_glb(output)
    animation = out["animations"][0]
    sim = np.asarray(mapping["source_to_canonical"])
    for frame in range(60):
        posed = copy.deepcopy(out)
        for channel in animation["channels"]:
            values = read_accessor(
                out, body, animation["samplers"][channel["sampler"]]["output"]
            )
            posed["nodes"][channel["target"]["node"]][channel["target"]["path"]] = (
                values[frame].tolist()
            )
        world, _, _ = world_matrices(posed)
        canonical = world[mapping["joint_indices"], :3, 3] @ sim[:3, :3].T + sim[:3, 3]
        np.testing.assert_allclose(canonical, expected[frame], atol=3e-6)
