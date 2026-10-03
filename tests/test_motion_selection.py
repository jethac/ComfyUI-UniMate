import numpy as np
import pytest

from unimate_pack.motion_selection import frame_mask, joint_mask


def test_signed_frame_indices_are_resolved_against_actual_clip_length():
    mask = frame_mask("0,-1,0", 7, 60)
    assert mask.shape == (1, 1, 1, 60)
    assert np.flatnonzero(mask).tolist() == [0, 6]


@pytest.mark.parametrize("spec", ["", "7", "-8", "1.5"])
def test_invalid_frame_selection_fails_instead_of_clamping(spec):
    with pytest.raises(ValueError):
        frame_mask(spec, 7, 60)


def test_joint_selection_matches_raw_and_clean_names_without_case():
    mask = joint_mask("head,hip", ["Root|Hip", "HEAD", "LeftArm"], 5)
    assert mask.shape == (1, 5, 1, 1)
    assert np.flatnonzero(mask).tolist() == [0, 1]


def test_unmatched_joint_selection_is_an_error():
    with pytest.raises(ValueError, match="matched"):
        joint_mask("tail", ["Hip", "Head"], 5)
