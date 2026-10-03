import numpy as np


def legged_features():
    from unimate_pack.foot_ik import forward_pose, rotation_increment
    from unimate_pack.feature_encoding import encode_motion_features
    parents = np.array([-1, 0, 1, 2, 0])
    offsets = np.array([[0., 1., 0.], [.2, -.1, 0.], [0., -.5, .15],
                        [0., -.4, -.15], [-.2, -.1, 0.]])
    rotations = np.tile(np.eye(3), (21, 5, 1, 1))
    rotations[:, 1] = np.stack([rotation_increment(np.array([.01 * np.sin(t), 0, 0]))
                               for t in range(21)])
    positions = np.stack([forward_pose(parents, offsets, pose, offsets[0])[0] for pose in rotations])
    features = encode_motion_features(positions, rotations, parents, np.tile(np.eye(3), (21, 1, 1)))
    return features, parents, offsets


def portable_legged_motion():
    from fixtures.rig_generator import legged_glb
    from unimate_pack.contracts import make_asset, make_rig, make_motion, encode_arrays
    from unimate_pack.rig_math import parse_glb, prepare_document
    from unimate_pack.foot_ik import forward_pose, rotation_increment
    from unimate_pack.feature_encoding import encode_motion_features
    asset = make_asset(legged_glb(), 'legged.glb')
    cond, mapping = prepare_document(parse_glb(asset['glb'])[0], '+Z')
    rig = make_rig(asset, encode_arrays(**cond), mapping)
    parents, offsets = cond['parents'], cond['tpos_offsets']
    rotations = np.tile(np.eye(3), (61, len(parents), 1, 1))
    hip = cond['joint_names'].tolist().index('LeftHip')
    rotations[:, hip] = np.stack([rotation_increment(np.array([.01 * np.sin(t), 0, 0]))
                                 for t in range(61)])
    root = np.broadcast_to(offsets[0], (61, 3)).copy()
    root[:, 0] += .0001 * np.arange(61)
    poses = np.stack([forward_pose(parents, offsets, pose, position)[0]
                      for pose, position in zip(rotations, root)])
    facing = np.stack([rotation_increment(np.array([0, .1 * np.sin(t / 10), 0]))
                       for t in range(61)])
    features = encode_motion_features(poses, rotations, parents, facing)
    motion = make_motion(rig['rig_id'], encode_arrays(features=features),
                         {'canonical_root_origin': [3, 0, 4]})
    return rig, motion, cond


def test_portable_success_preserves_moving_root_facing_and_reencodes_velocity():
    from unimate_pack.foot_lock import lock_motion
    from unimate_pack.contracts import decode_arrays
    from unimate_pack.rig_math import decode_features, rotation_6d_matrices
    from unimate_pack.feature_encoding import encode_motion_features
    from unimate_pack.skeleton import recover_positions
    rig, motion, cond = portable_legged_motion()
    corrected, report = lock_motion(rig, motion, joint_names='leftfoot')
    assert corrected['features'] != motion['features'] and report['segments']
    source = decode_arrays(motion['features'])['features']
    features = decode_arrays(corrected['features'])['features']
    np.testing.assert_array_equal(features[:, 0], source[:, 0])
    parents, offsets = cond['parents'], cond['tpos_offsets']
    positions = recover_positions(features, parents, offsets, 'fk', root_origin=[3, 0, 4])
    ric = recover_positions(features, parents, offsets, 'ric', root_origin=[3, 0, 4])
    np.testing.assert_allclose(ric, positions, atol=2e-7)
    rotations, _ = decode_features(features, parents)
    facing = rotation_6d_matrices(features[:, 0, 3:9])
    rebuilt = encode_motion_features(positions, rotations, parents, facing)
    np.testing.assert_allclose(features[:-1, :, 9:12], rebuilt[:, :, 9:12], atol=2e-7)


def test_lock_features_preserves_root_other_chain_and_length_and_reduces_contact_error():
    from unimate_pack.foot_lock import lock_features
    from unimate_pack.skeleton import recover_positions
    features, parents, offsets = legged_features()
    source = features.copy()
    corrected, report = lock_features(features, parents, offsets, [3])
    np.testing.assert_array_equal(features, source)
    np.testing.assert_array_equal(corrected[:, [0, 4]], source[:, [0, 4]])
    assert corrected.shape == features.shape and corrected.dtype == features.dtype
    original = recover_positions(features, parents, offsets, 'fk')
    positions = recover_positions(corrected, parents, offsets, 'fk')
    np.testing.assert_array_equal(positions[:, [0, 4]], original[:, [0, 4]])
    np.testing.assert_allclose(np.linalg.norm(positions[:, 1:] - positions[:, parents[1:]], axis=-1),
                               np.broadcast_to(np.linalg.norm(offsets[1:], axis=-1),
                                               (len(features), len(parents) - 1)), atol=1e-7)
    anchor = np.array(report['segments'][0]['anchor'])
    assert np.linalg.norm(positions[5:-5, 3] - anchor, axis=-1).max() < 1e-5
    assert np.linalg.norm(original[5:-5, 3] - anchor, axis=-1).max() > .005
    np.testing.assert_array_equal(corrected[[0, -1], :, :9], source[[0, -1], :, :9])
    ric = recover_positions(corrected, parents, offsets, 'ric')
    np.testing.assert_allclose(ric[:, 3], positions[:, 3], atol=1e-7)


def test_no_contact_is_byte_identical():
    from unimate_pack.foot_lock import lock_features
    features, parents, offsets = legged_features()
    corrected, report = lock_features(features, parents, offsets, [])
    assert corrected.tobytes() == features.tobytes()
    assert report['segments'] == []


def test_workspace_budget_rejects_before_fk_allocation(monkeypatch):
    import pytest
    import unimate_pack.foot_lock as module
    features, parents, offsets = legged_features()
    monkeypatch.setattr(module, 'MAX_WORKSPACE_BYTES', 1, raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('FK allocated before budget check')
    monkeypatch.setattr(module, 'recover_positions', forbidden)
    with pytest.raises(ValueError, match='workspace budget'):
        module.lock_features(features, parents, offsets, [3])


def test_contact_overrides_reject_overlapping_limb_chains():
    import pytest
    from unimate_pack.foot_lock import lock_features
    features, parents, offsets = legged_features()
    with pytest.raises(ValueError, match='overlap'):
        lock_features(features, parents, offsets, [2, 3])


def test_portable_correction_preserves_identity_metadata_and_source():
    import copy
    from test_skeleton import prepared_motion
    from unimate_pack.foot_lock import lock_motion
    from unimate_pack.contracts import validate_motion
    rig, motion, _ = prepared_motion()
    source = copy.deepcopy(motion)
    corrected, report = lock_motion(rig, motion)
    validate_motion(corrected, rig)
    assert corrected['rig_id'] == motion['rig_id']
    assert corrected['features'] == motion['features']
    assert corrected['metadata']['canonical_root_origin'] == [3, 0, 4]
    assert corrected['metadata']['foot_lock'] == report
    assert motion == source


def test_portable_correction_rejects_other_rig_and_unknown_override():
    import pytest
    from test_skeleton import prepared_motion
    from unimate_pack.foot_lock import lock_motion
    rig, motion, _ = prepared_motion()
    with pytest.raises(ValueError, match='joint'):
        lock_motion(rig, motion, joint_names='MissingFoot')
    motion['rig_id'] = '0' * 64
    with pytest.raises(ValueError, match='identities'):
        lock_motion(rig, motion)


def test_portable_cancellation_during_solver_keeps_source_unchanged():
    import copy
    import pytest
    from unimate_pack.foot_lock import lock_motion
    rig, motion, _ = portable_legged_motion()
    source = copy.deepcopy(motion)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 40:
            raise InterruptedError('cancelled during correction')
    with pytest.raises(InterruptedError, match='during correction'):
        lock_motion(rig, motion, check_cancel=cancel)
    assert calls == 40 and motion == source


def test_selected_zero_height_rig_and_malformed_overrides_fail():
    import pytest
    from test_skeleton import prepared_motion
    from unimate_pack.foot_lock import lock_motion
    rig, motion, _ = prepared_motion()
    with pytest.raises(ValueError, match='height'):
        lock_motion(rig, motion, joint_names='Joint_5')
    rig, motion, _ = portable_legged_motion()
    for names in ['LeftFoot,', 'LeftFoot,leftfoot', 'Root', 'LeftKnee,LeftFoot']:
        with pytest.raises(ValueError, match='joint|overlap'):
            lock_motion(rig, motion, joint_names=names)


def test_original_legged_glb_has_ground_contacts_and_consistent_bind_pose():
    from fixtures.rig_generator import legged_glb
    from unimate_pack.contracts import make_asset, make_rig, encode_arrays
    from unimate_pack.rig_math import parse_glb, prepare_document
    from unimate_pack.foot_contacts import select_contact_joints
    asset = make_asset(legged_glb(), 'legged.glb')
    conditioning, mapping = prepare_document(parse_glb(asset['glb'])[0], '+Z')
    make_rig(asset, encode_arrays(**conditioning), mapping)
    contacts = select_contact_joints(conditioning['parents'], conditioning['joint_names'])
    assert len(contacts) == 2
    positions = conditioning['tpos_first_frame']
    assert positions[0, 1] > .1
    np.testing.assert_allclose(positions[contacts, 1], 0, atol=1e-7)


def test_corrected_legged_glb_channels_recover_every_canonical_pose():
    import copy
    from fixtures.rig_generator import legged_glb
    from test_blender_math import read_accessor, evaluated_vertices
    from unimate_pack.rig_math import parse_glb, prepare_document, animate_document, world_matrices
    from unimate_pack.foot_ik import forward_pose, rotation_increment
    from unimate_pack.feature_encoding import encode_motion_features
    from unimate_pack.skeleton import recover_positions
    from unimate_pack.foot_lock import lock_features
    source = legged_glb()
    document, source_binary = parse_glb(source)
    conditioning, mapping = prepare_document(document, '+Z')
    parents, offsets = conditioning['parents'], conditioning['tpos_offsets']
    rotations = np.tile(np.eye(3), (21, len(parents), 1, 1))
    left_hip = conditioning['joint_names'].tolist().index('LeftHip')
    rotations[:, left_hip] = np.stack([rotation_increment(np.array([.01 * np.sin(t), 0, 0]))
                                      for t in range(21)])
    poses = np.stack([forward_pose(parents, offsets, pose, offsets[0])[0] for pose in rotations])
    features = encode_motion_features(poses, rotations, parents, np.tile(np.eye(3), (21, 1, 1)))
    contacts = [conditioning['joint_names'].tolist().index('LeftFoot')]
    corrected, _ = lock_features(features, parents, offsets, contacts)
    expected = recover_positions(corrected, parents, offsets, 'fk')
    exported = animate_document(source, conditioning, mapping, corrected)
    result, binary = parse_glb(exported)
    assert binary[:len(source_binary)] == source_binary
    for field in ('skins', 'meshes', 'materials', 'images', 'textures'):
        assert result[field] == document[field]
    for frame in range(len(corrected)):
        pose_document = copy.deepcopy(result)
        animation = result['animations'][0]
        for channel in animation['channels']:
            sampler = animation['samplers'][channel['sampler']]
            values = read_accessor(result, binary, sampler['output'])
            pose_document['nodes'][channel['target']['node']][channel['target']['path']] = values[frame].tolist()
        worlds, _, _ = world_matrices(pose_document)
        actual = worlds[mapping['joint_indices'], :3, 3]
        canonical = np.c_[actual, np.ones(len(actual))] @ np.asarray(mapping['source_to_canonical']).T
        np.testing.assert_allclose(canonical[:, :3], expected[frame], atol=3e-7)
        assert np.isfinite(evaluated_vertices(exported, frame)).all()


def test_blender_playback_matches_corrected_legged_skinning(tmp_path):
    import os
    from pathlib import Path
    import pytest
    from test_blender_math import evaluated_vertices
    from unimate_pack.foot_lock import lock_motion
    from unimate_pack.blender import export_glb, blender_executable, _run_process
    if not os.environ.get('UNIMATE_BLENDER'):
        pytest.skip('Requires explicit installed Blender')
    rig, motion, _ = portable_legged_motion()
    corrected, report = lock_motion(rig, motion)
    assert len(report['contact_joints']) == 2 and report['segments']
    exported = export_glb(rig, corrected)
    path = tmp_path / 'foot locked.glb'
    path.write_bytes(exported)
    probe = Path(__file__).parent / 'fixtures' / 'blender_inspect.py'
    _run_process([blender_executable(), '--background', '--factory-startup',
        '--disable-autoexec', '--python-exit-code', '1', '--python', str(probe.resolve()),
        '--', str(path), str(tmp_path / 'playback.npz')], tmp_path)
    with np.load(tmp_path / 'playback.npz', allow_pickle=False) as playback:
        assert len(playback['vertices']) == 60
        for frame, actual in enumerate(playback['vertices']):
            expected = evaluated_vertices(exported, frame)
            error = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
            assert np.max(np.min(error, axis=0)) < 2e-5
            assert np.max(np.min(error, axis=1)) < 2e-5
