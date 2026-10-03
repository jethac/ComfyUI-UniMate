import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from unimate_pack.feature_encoding import encode_motion_features, rebase_rotations


def test_feature_encoding_uses_destination_facing_for_velocity():
    positions = np.zeros((3, 5, 3))
    positions[:, :, 0] = np.arange(3)[:, None]
    positions[:, :, 1] = 2
    rotations = np.broadcast_to(np.eye(3), (3, 5, 3, 3)).copy()
    facing = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
    facing[1:] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    features = encode_motion_features(positions, rotations, [-1, 0, 1, 1, 2], facing)
    assert features.shape == (2, 5, 12)
    np.testing.assert_array_equal(features[:, :, 9:12], np.broadcast_to([0, 0, -1], (2, 5, 3)))
    np.testing.assert_array_equal(features[:, 0, :3], [[0, 2, 0], [0, 2, 0]])


def test_feature_encoding_matches_upstream_for_rotating_branching_motion():
    root = os.environ.get("UNIMATE_REFERENCE")
    motion = os.environ.get("UNIMATE_MOTION_REFERENCE")
    if not root or not motion:
        pytest.skip("Requires pinned upstream and local Motion reference")
    sys.path[:0] = [root, motion]
    from Quaternions import Quaternions
    spec = importlib.util.spec_from_file_location("reference_motion_utils", Path(root) / "unimate/utils/motion_utils.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(8, 5, 3))
    rotations = Quaternions.from_euler(rng.normal(size=(8, 5, 3)) * 0.2)
    facing = Quaternions.from_euler(rng.normal(size=(8, 3)) * 0.1)
    parents = np.array([-1, 0, 0, 1, 2])
    expected = reference.compute_unimate_motion_feats(positions, rotations, parents, facing)
    actual = encode_motion_features(positions, rotations.rotation_matrix(cont6d=False), parents,
                                   facing.rotation_matrix(cont6d=False))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_feature_encoding_rejects_single_pose_without_velocity_destination():
    with pytest.raises(ValueError):
        encode_motion_features(np.zeros((1, 5, 3)), np.zeros((1, 5, 3, 3)), [-1, 0, 1, 2, 3], np.zeros((1, 3, 3)))


def test_rotated_rest_pose_rebase_matches_pinned_upstream():
    root = os.environ.get("UNIMATE_REFERENCE")
    motion = os.environ.get("UNIMATE_MOTION_REFERENCE")
    if not root or not motion:
        pytest.skip("Requires local references")
    sys.path[:0] = [root, motion]
    from Quaternions import Quaternions
    spec = importlib.util.spec_from_file_location("reference_rotation_rebase", Path(root) / "unimate/utils/motion_utils.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(7)
    parents = np.array([-1, 0, 0, 1, 2])
    rest = Quaternions.from_euler(rng.normal(size=(1, 5, 3)) * 0.5)
    animated = Quaternions.from_euler(rng.normal(size=(8, 5, 3)) * 0.3)
    expected = reference.compute_rots_from_tpos(rest, animated, parents).rotation_matrix(cont6d=False)
    actual = rebase_rotations(rest.rotation_matrix(cont6d=False)[0],
                              animated.rotation_matrix(cont6d=False), parents)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
