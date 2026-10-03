import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from unimate_pack.facing import facing_rotations


@pytest.mark.parametrize("indices,body_axis", [([0, 1], False), ([0, 1, 2, 3], False),
                                              ([0, 1], True), ([-1, -1], False)])
def test_facing_matches_pinned_upstream(indices, body_axis):
    root = os.environ.get("UNIMATE_REFERENCE")
    motion = os.environ.get("UNIMATE_MOTION_REFERENCE")
    if not root or not motion:
        pytest.skip("Requires local pinned references")
    sys.path.insert(0, motion)
    spec = importlib.util.spec_from_file_location("facing_upstream", Path(root) / "data_process/utils/skeleton.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    positions = np.random.default_rng(43).normal(size=(20, 5, 3))
    expected = reference.get_root_facing_quat(positions, indices, body_axis).rotation_matrix(cont6d=False)
    np.testing.assert_allclose(facing_rotations(positions, indices, body_axis), expected, atol=1e-10)


def test_degenerate_pair_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        facing_rotations(np.zeros((3, 5, 3)), [0, 1])
