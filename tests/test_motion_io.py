import numpy as np
import pytest

from unimate_pack.contracts import make_motion, encode_arrays
from unimate_pack.motion_io import dump_motion, load_motion


def test_reference_motion_round_trip_preserves_basis_identity_and_metadata():
    features = np.arange(110 * 5 * 12, dtype=np.float32).reshape(110, 5, 12) / 100
    original = make_motion("a" * 64, encode_arrays(features=features), {"prompt": "歩く"})
    restored = load_motion(dump_motion(original), rig_id="a" * 64)
    assert restored == original


def test_reference_motion_cannot_be_loaded_for_another_rig():
    original = make_motion("a" * 64, encode_arrays(features=np.zeros((3, 5, 12), np.float32)), {})
    with pytest.raises(ValueError, match="identities"):
        load_motion(dump_motion(original), rig_id="b" * 64)


def test_plain_features_cannot_silently_acquire_a_rig_identity():
    with pytest.raises(ValueError):
        load_motion(encode_arrays(features=np.zeros((60, 5, 12), np.float32)))


def test_long_expansion_provenance_round_trips_without_unicode_dtype_overflow():
    original = make_motion("a" * 64, encode_arrays(features=np.zeros((110, 5, 12), np.float32)),
                           {"segments": [{"prompt": "歩く" * 2000} for _ in range(4)]})
    assert load_motion(dump_motion(original)) == original


def test_existing_v1_unicode_archive_remains_readable():
    features = np.zeros((7, 5, 12), np.float32)
    old = encode_arrays(features=features, schema=np.asarray("unimate.motion.file.v1"),
        rig_id=np.asarray("a" * 64), fps=np.asarray(30, dtype=np.int32),
        metadata=np.asarray('{"prompt":"walk"}'))
    expected = make_motion("a" * 64, encode_arrays(features=features), {"prompt": "walk"})
    assert load_motion(old) == expected


def test_v2_metadata_requires_uint8_utf8_vector():
    bad = encode_arrays(features=np.zeros((7, 5, 12), np.float32),
        schema=np.asarray("unimate.motion.file.v2"), rig_id=np.asarray("a" * 64),
        fps=np.asarray(30, dtype=np.int32), metadata=np.asarray("{}"))
    with pytest.raises(ValueError, match="uint8"):
        load_motion(bad)
