import numpy as np
import pytest


def test_contact_names_choose_distal_joints_on_each_limb():
    from unimate_pack.foot_contacts import select_contact_joints
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    names = ['Root', 'LeftFoot', 'LeftToe', 'LeftToeEnd', 'RightFoot', 'RightToe', 'RightToeEnd']
    assert select_contact_joints(parents, names) == [3, 6]
    assert select_contact_joints(np.array([-1, 0, 0]), ['Root', 'Football', 'Paw']) == [2]


def test_contact_detection_normalizes_and_filters_brief_velocity_spikes():
    from unimate_pack.foot_contacts import contact_segments
    positions = np.zeros((15, 2, 3))
    positions[:, 1, 0] = np.arange(15) * 0.01
    positions[7, 1, 0] += 0.1
    segments, mask = contact_segments(positions, [1], ground_height=0, root_height=2)
    assert mask[:, 0].all()
    assert len(segments) == 1
    assert segments[0]['joint'] == 1
    assert (segments[0]['start'], segments[0]['end']) == (0, 15)
    np.testing.assert_allclose(segments[0]['anchor'], [positions[:, 1, 0].mean(), 0, 0])


def test_short_contacts_and_frames_above_height_threshold_are_not_locked():
    from unimate_pack.foot_contacts import contact_segments
    positions = np.zeros((2, 2, 3))
    segments, mask = contact_segments(positions, [1], ground_height=0, root_height=1)
    assert not segments and not mask.any()
    positions = np.zeros((9, 2, 3))
    positions[:, 1, 1] = 0.05
    segments, mask = contact_segments(positions, [1], ground_height=0, root_height=1)
    assert not segments and not mask.any()


@pytest.mark.parametrize('height', [0, -1, float('nan')])
def test_invalid_rest_height_fails_before_normalization(height):
    from unimate_pack.foot_contacts import contact_segments
    with pytest.raises(ValueError, match='height'):
        contact_segments(np.zeros((9, 2, 3)), [1], ground_height=0, root_height=height)


def test_contact_detection_obeys_cancellation():
    from unimate_pack.foot_contacts import contact_segments
    def stop():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        contact_segments(np.zeros((9, 2, 3)), [1], ground_height=0, root_height=1, check_cancel=stop)


def test_contact_detection_rejects_empty_skeletons():
    from unimate_pack.foot_contacts import contact_segments
    with pytest.raises(ValueError, match='positions'):
        contact_segments(np.zeros((9, 0, 3)), [], ground_height=0, root_height=1)
