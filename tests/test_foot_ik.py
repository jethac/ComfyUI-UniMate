import numpy as np


def test_contact_jacobian_matches_local_rotation_finite_differences():
    from unimate_pack.foot_ik import forward_pose, contact_jacobian, rotation_increment
    parents = np.array([-1, 0, 1, 2, 0])
    offsets = np.array([[0., 1., 0.], [.2, -.1, 0.], [0., -.5, .15],
                        [0., -.4, -.15], [-.2, -.1, 0.]])
    rotations = np.stack([rotation_increment(np.array([.1, -.2, .05]) * i) for i in range(5)])
    root = np.array([2., 1., -3.])
    positions, worlds = forward_pose(parents, offsets, rotations, root)
    jacobian = contact_jacobian(positions, worlds, 3, [1, 2])
    for column in range(6):
        changed = rotations.copy()
        delta = np.zeros(3)
        delta[column % 3] = 1e-6
        joint = [1, 2][column // 3]
        changed[joint] = changed[joint] @ rotation_increment(delta)
        moved, _ = forward_pose(parents, offsets, changed, root)
        changed[joint] = rotations[joint] @ rotation_increment(-delta)
        reverse, _ = forward_pose(parents, offsets, changed, root)
        np.testing.assert_allclose((moved[3] - reverse[3]) / 2e-6,
                                   jacobian[:, column], atol=1e-9)
    np.testing.assert_array_equal(positions[0], root)
    np.testing.assert_allclose(np.linalg.norm(positions[1:] - positions[parents[1:]], axis=1),
                               np.linalg.norm(offsets[1:], axis=1))


def test_rotation_increment_is_proper_for_zero_and_small_angles():
    from unimate_pack.foot_ik import rotation_increment
    np.testing.assert_array_equal(rotation_increment(np.zeros(3)), np.eye(3))
    for vector in ([1e-10, 0, 0], [.2, -.4, .3]):
        rotation = rotation_increment(np.array(vector))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-14)


def test_damped_limb_solver_reaches_anchor_without_moving_root_or_other_chain():
    from unimate_pack.foot_ik import forward_pose, solve_contact
    parents = np.array([-1, 0, 1, 2, 0])
    offsets = np.array([[0., 1., 0.], [.2, -.1, 0.], [0., -.5, .15],
                        [0., -.4, -.15], [-.2, -.1, 0.]])
    rotations = np.tile(np.eye(3), (5, 1, 1))
    root = np.array([0., 1., 0.])
    original, _ = forward_pose(parents, offsets, rotations, root)
    target = original[3] + np.array([.025, .02, .01])
    solved, report = solve_contact(parents, offsets, rotations, root, 3, [1, 2],
                                   target, scale=1.)
    positions, _ = forward_pose(parents, offsets, solved, root)
    assert report['residual'] < 1e-5
    np.testing.assert_allclose(positions[3], target, atol=1e-5)
    np.testing.assert_array_equal(positions[[0, 4]], original[[0, 4]])
    np.testing.assert_array_equal(solved[[0, 3, 4]], rotations[[0, 3, 4]])
    np.testing.assert_array_equal(rotations, np.tile(np.eye(3), (5, 1, 1)))


def test_damped_solver_reports_unreachable_target_and_obeys_cancellation():
    import pytest
    from unimate_pack.foot_ik import solve_contact
    parents = np.array([-1, 0, 1])
    offsets = np.array([[0., 1., 0.], [0., -.4, 0.], [0., -.4, 0.]])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    _, report = solve_contact(parents, offsets, rotations, offsets[0], 2, [1],
                              np.array([10., 0., 0.]), scale=1., iterations=4)
    assert report['residual'] > 9 and report['iterations'] <= 4
    def stop():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        solve_contact(parents, offsets, rotations, offsets[0], 2, [1],
                      np.zeros(3), scale=1., check_cancel=stop)


def test_limb_chain_stops_before_branch_and_root():
    from unimate_pack.foot_ik import limb_chain
    assert limb_chain(np.array([-1, 0, 1, 2, 1]), 3) == [2]
    assert limb_chain(np.array([-1, 0, 1, 2, 3, 4]), 5) == [4, 3, 2]
    assert limb_chain(np.array([-1, 0]), 1) == []


def test_boundary_weights_and_rotation_blending():
    from unimate_pack.foot_ik import boundary_weights, blend_rotation, rotation_increment
    np.testing.assert_allclose(boundary_weights(12),
                               [0, .25, .5, .75, 1, 1, 1, 1, .75, .5, .25, 0])
    np.testing.assert_allclose(boundary_weights(3), [0, .25, 0])
    source = rotation_increment(np.array([.1, .2, 0]))
    target = source @ rotation_increment(np.array([0, 0, .8]))
    np.testing.assert_array_equal(blend_rotation(source, target, 0), source)
    np.testing.assert_array_equal(blend_rotation(source, target, 1), target)
    np.testing.assert_allclose(blend_rotation(source, target, .5),
                               source @ rotation_increment(np.array([0, 0, .4])), atol=1e-14)
