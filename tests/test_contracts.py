import copy
import hashlib
import io
import json
import zipfile
import warnings

import numpy as np
import pytest

from unimate_pack.contracts import (
    decode_arrays,
    encode_arrays,
    make_asset,
    make_model,
    make_motion,
    make_rig,
    validate_asset,
    validate_model,
    validate_motion,
    validate_rig,
)
from test_assets import synthetic_glb


def npz(**arrays):
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def model_bundle(entries=None):
    entries = entries or {"config.json": b"{}"}
    manifest = {
        "schema": "unimate.bundle.v1",
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in entries.items()
        },
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, value in entries.items():
            archive.writestr(name, value)
    return stream.getvalue()


def rig_mapping(**changes):
    """A complete portable mapping for the independent five-joint chain fixture."""
    value = {
        "joint_indices": [0, 1, 2, 3, 4],
        "source_to_canonical": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "adapter_revision": "gltf-preserving-v1",
        "upstream_revision": "5d6aabedd947297b5ba6706d8e9113e68c0c3e4f",
        "facing": "+Z",
        "quaternion_order": "wxyz conditioning; xyzw glTF",
    }
    value.update(changes)
    return value


def test_array_roundtrip_is_plain_npz_and_preserves_unicode_and_numeric_dtype():
    payload = encode_arrays(
        parents=np.array([-1, 0, 1], dtype=np.int64),
        names=np.array(["root", "左", "右"]),
        rest=np.zeros((3, 3), np.float32),
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as actual:
        assert actual["parents"].tolist() == [-1, 0, 1]
        assert actual["names"].tolist() == ["root", "左", "右"]
    assert decode_arrays(payload)["rest"].dtype == np.float32


@pytest.mark.parametrize(
    "array",
    [
        np.array([{}], object),
        np.array([float("inf")]),
        np.array([float("nan")]),
        np.array([1 + 2j]),
    ],
)
def test_unsafe_arrays_are_rejected_at_both_boundaries(array):
    with pytest.raises(ValueError):
        encode_arrays(unsafe=array)
    with pytest.raises(ValueError):
        decode_arrays(npz(unsafe=array))


def test_npz_traversal_and_duplicate_names_are_rejected():
    member = io.BytesIO()
    np.save(member, np.zeros(1), allow_pickle=False)
    for names in (["../a.npy"], ["a.npy", "a.npy"]):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name in names:
                archive.writestr(name, member.getvalue())
        with pytest.raises(ValueError):
            decode_arrays(stream.getvalue())


def test_digest_identifies_exact_prepared_rig_and_survives_portable_copy():
    asset = make_asset(synthetic_glb(), "character.glb")
    data = encode_arrays(parents=np.array([-1, 0, 1, 2, 3], np.int64))
    first = make_rig(asset, data, rig_mapping())
    validate_rig(copy.deepcopy(first))
    reordered = make_rig(asset, data, dict(reversed(list(rig_mapping().items()))))
    assert first["rig_id"] == reordered["rig_id"]
    changed = make_rig(asset, data, rig_mapping(facing="-Z"))
    assert changed["rig_id"] != first["rig_id"]
    corrupted = copy.deepcopy(first)
    corrupted["mapping"]["facing"] = "-Z"
    with pytest.raises(ValueError):
        validate_rig(corrupted)


def test_asset_rejects_tampering_and_local_paths():
    asset = make_asset(synthetic_glb(), "character.glb")
    for change in (
        {"sha256": "0" * 64},
        {"name": "C:\\secret\\character.glb"},
        {"path": "C:\\local.glb"},
        {"glb": b"broken"},
    ):
        with pytest.raises(ValueError):
            validate_asset(dict(asset, **change))


def test_motion_shape_fps_dtype_and_identity_are_enforced_before_export():
    rig_id = "a" * 64
    payload = npz(features=np.zeros((60, 5, 12), np.float32))
    motion = make_motion(rig_id, payload, {"seed": 7})
    validate_motion(motion, rig_id)
    with pytest.raises(ValueError, match="rig"):
        validate_motion(motion, "b" * 64)
    for change in (
        {"fps": 24},
        {"features": npz(features=np.zeros((0, 5, 12), np.float32))},
        {"features": npz(features=np.zeros((60, 5, 12), np.float64))},
        {"features": npz(features=np.zeros((60, 71, 12), np.float32))},
        {"metadata": {"model": "C:\\models\\local.safetensors"}},
    ):
        with pytest.raises(ValueError):
            validate_motion(dict(motion, **change))


@pytest.mark.parametrize("frames", [1, 59, 60, 110, 600])
def test_motion_contract_supports_reference_and_expanded_clip_lengths(frames):
    motion = make_motion(
        "a" * 64,
        npz(features=np.zeros((frames, 5, 12), np.float32)),
        {"frames": frames},
    )
    validate_motion(motion, "a" * 64)


def test_model_bundle_has_verified_hashes_and_safe_entries():
    value = make_model(model_bundle(), "tiny.unimate")
    validate_model(copy.deepcopy(value))
    with pytest.raises(ValueError):
        validate_model(dict(value, sha256="0" * 64))
    for entries in (
        {"../escape": b"x"},
        {"C:/absolute": b"x"},
        {"config.json": b"{}", "unsafe.pkl": b"x"},
    ):
        with pytest.raises(ValueError):
            make_model(model_bundle(entries), "tiny.unimate")


def test_nonfinite_json_and_live_values_cannot_cross_boundary():
    asset = make_asset(synthetic_glb(), "rig.glb")
    for mapping in (
        {"number": float("nan")},
        {"array": np.zeros(1)},
        {"local_path": "tmp.glb"},
    ):
        with pytest.raises(ValueError):
            make_rig(
                asset, npz(parents=np.array([-1, 0, 1, 2, 3])), rig_mapping(**mapping)
            )


def test_npy_header_cannot_request_allocation_beyond_member_data():
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        member, {"descr": "<f4", "fortran_order": False, "shape": (1_000_000_000,)}
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("huge.npy", member.getvalue())
    with pytest.raises(ValueError):
        decode_arrays(stream.getvalue())


def test_archive_size_limits_are_checked_before_decompression(monkeypatch):
    from unimate_pack import contracts

    payload = npz(values=np.zeros(1000))
    monkeypatch.setattr(contracts, "MAX_ARRAY_BYTES", 100)
    with pytest.raises(ValueError):
        decode_arrays(payload)


@pytest.mark.parametrize("creator", [make_asset, make_model])
def test_invalid_binary_input_has_actionable_validation_error(creator):
    with pytest.raises(ValueError):
        creator("local file", "rig.glb")


def test_rig_mapping_indices_must_identify_source_skin():
    asset = make_asset(synthetic_glb(), "rig.glb")
    parents = npz(parents=np.array([-1, 0, 1, 2, 3]))
    with pytest.raises(ValueError):
        make_rig(asset, parents, rig_mapping(joint_indices=[0, 1, 2, 3, 99]))


def test_model_zip_corrupted_member_hash_cannot_be_accepted():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "unimate.bundle.v1",
                    "files": {"config.json": {"sha256": "0" * 64, "size": 2}},
                }
            ),
        )
        archive.writestr("config.json", b"{}")
    with pytest.raises(ValueError):
        make_model(stream.getvalue(), "rig.unimate")


def test_model_zip_executable_and_object_stats_are_rejected():
    for entries in (
        {"run.sh": b"echo unsafe"},
        {"stats.npz": npz(mean=np.array([{}], object))},
    ):
        with pytest.raises(ValueError):
            make_model(model_bundle(entries), "rig.unimate")


def test_prompt_is_plain_text_even_when_it_begins_with_slash_or_a_path_example():
    for prompt in (
        "/walk like a robot",
        "C:\\temp\\character.glb is written on the sign",
    ):
        motion = make_motion(
            "a" * 64,
            npz(features=np.zeros((60, 5, 12), np.float32)),
            {"prompt": prompt},
        )
        assert motion["metadata"]["prompt"] == prompt


def test_model_manifest_duplicate_keys_and_nonobject_json_are_rejected():
    for manifest in (
        b"[]",
        b'{"schema":"wrong","schema":"unimate.bundle.v1","files":{}}',
    ):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("manifest.json", manifest)
        with pytest.raises(ValueError):
            make_model(stream.getvalue(), "rig.unimate")


def test_prepared_joint_count_cannot_differ_from_source_skin():
    asset = make_asset(synthetic_glb(), "rig.glb")
    with pytest.raises(ValueError):
        make_rig(asset, npz(parents=np.array([-1, 0, 1, 2, 3, 4])), rig_mapping())


@pytest.mark.parametrize(
    "field, array",
    [
        ("offsets", np.full((5, 3), "invalid")),
        ("joint_names", np.arange(5)),
        ("edge_indexs", np.array([[0, 5], [1, 2]])),
        ("kinematic_chains", np.array([[0, 1, 99, -1]])),
        ("face_joint_idxs", np.array([-1, 2])),
        ("scale_factor", np.array(0.0)),
        ("spectral_feats", np.zeros((4, 8))),
        ("joint_graph_dists", -np.ones((5, 5))),
        ("tpos_local_rotations", np.zeros((5, 4))),
    ],
)
def test_conditioning_semantic_shapes_and_indices_are_rejected(field, array):
    asset = make_asset(synthetic_glb(), "rig.glb")
    values = {"parents": np.array([-1, 0, 1, 2, 3]), field: array}
    with pytest.raises(ValueError):
        make_rig(asset, npz(**values), rig_mapping())


def test_prepared_parent_mapping_must_preserve_source_joint_hierarchy():
    asset = make_asset(synthetic_glb(), "rig.glb")
    with pytest.raises(ValueError):
        make_rig(
            asset,
            npz(parents=np.array([-1, 0, 0, 0, 0])),
            rig_mapping(),
        )


@pytest.mark.parametrize("field", list(rig_mapping()))
def test_required_rig_mapping_fields_cannot_be_omitted(field):
    mapping = rig_mapping()
    del mapping[field]
    with pytest.raises(ValueError):
        make_rig(
            make_asset(synthetic_glb(), "rig.glb"),
            npz(parents=np.array([-1, 0, 1, 2, 3])),
            mapping,
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, 0, 0, 0]] * 4,
        [[1, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0], [0, 1]],
    ],
)
def test_source_to_canonical_requires_invertible_positive_uniform_similarity(matrix):
    with pytest.raises(ValueError):
        make_rig(
            make_asset(synthetic_glb(), "rig.glb"),
            npz(parents=np.array([-1, 0, 1, 2, 3])),
            rig_mapping(source_to_canonical=matrix),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"adapter_revision": "unknown"},
        {"upstream_revision": "0" * 40},
        {"quaternion_order": "xyzw conditioning"},
        {"facing": "unknown"},
        {"facing": "joint_pair"},
        {"facing": "joint_pair", "left_joint": "joint_0", "right_joint": "joint_0"},
        {"facing": "joint_pair", "left_joint": "joint_0", "right_joint": "missing"},
    ],
)
def test_rig_revisions_conventions_and_facing_pair_are_explicit(change):
    with pytest.raises(ValueError):
        make_rig(
            make_asset(synthetic_glb(), "rig.glb"),
            npz(parents=np.array([-1, 0, 1, 2, 3])),
            rig_mapping(**change),
        )


def test_positive_similarity_can_be_reversed_after_portable_rig_roundtrip():
    matrix = [[0, 0, 2, 4], [0, 2, 0, -3], [-2, 0, 0, 5], [0, 0, 0, 1]]
    rig = make_rig(
        make_asset(synthetic_glb(), "rig.glb"),
        npz(parents=np.array([-1, 0, 1, 2, 3])),
        rig_mapping(source_to_canonical=matrix),
    )
    validate_rig(copy.deepcopy(rig))
    transform = np.asarray(rig["mapping"]["source_to_canonical"])
    source = np.array([1, 2, 3, 1])
    assert (transform @ source).tolist() == [10, 1, 3, 1]
    np.testing.assert_allclose(np.linalg.inv(transform) @ [10, 1, 3, 1], source)


@pytest.mark.parametrize("branching", [False, True])
def test_complete_real_adapter_mapping_passes_contract_for_both_topologies(branching):
    from fixtures.rig_generator import synthetic_glb as adapter_fixture
    from unimate_pack.rig_math import parse_glb, prepare_document

    source = adapter_fixture(branching)
    conditioning, mapping = prepare_document(parse_glb(source)[0], "+Z")
    rig = make_rig(
        make_asset(source, "fixture.glb"), encode_arrays(**conditioning), mapping
    )
    validate_rig(copy.deepcopy(rig))
    assert len(decode_arrays(rig["conditioning"])["parents"]) == (7 if branching else 5)


def test_transform_integer_overflow_fails_as_contract_error():
    mapping = rig_mapping()
    mapping["source_to_canonical"][0][0] = 10**1000
    with pytest.raises(ValueError):
        make_rig(
            make_asset(synthetic_glb(), "rig.glb"),
            npz(parents=np.array([-1, 0, 1, 2, 3])),
            mapping,
        )
