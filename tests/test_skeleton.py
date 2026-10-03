import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from unimate_pack.skeleton import recover_positions, recover_skeleton
from unimate_pack.contracts import (make_asset, make_rig, make_motion, encode_arrays,
                                    decode_arrays, validate_skeleton)
from unimate_pack.rig_math import prepare_document, parse_glb

sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
from rig_generator import synthetic_glb
from unimate_pack.skeleton_preview import render_skeleton


@pytest.mark.parametrize('method', ['fk', 'ric'])
def test_recovery_matches_pinned_upstream(method):
    root = os.environ.get('UNIMATE_REFERENCE')
    motion = os.environ.get('UNIMATE_MOTION_REFERENCE')
    if not root or not motion:
        pytest.skip('Requires pinned UniMate and local Motion reference')
    sys.path[:0] = [root, motion]
    from Quaternions import Quaternions
    spec = importlib.util.spec_from_file_location('skeleton_reference', Path(root) / 'unimate/utils/motion_utils.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(9)
    parents = np.array([-1, 0, 0, 1, 2])
    offsets = rng.normal(size=(5, 3))
    features = rng.normal(size=(17, 5, 12))
    rotations = Quaternions.from_euler(rng.normal(size=(17, 5, 3)) * 0.5)
    features[..., 3:9] = rotations.rotation_matrix(cont6d=True)
    if method == 'fk':
        expected = reference.recover_unimate_joint_pos_from_rot(features, parents, offsets)
    else:
        expected = reference.recover_unimate_joint_pos_from_ric(features)
    actual = recover_positions(features, parents, offsets, method)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('method', ['fk', 'ric'])
def test_root_uses_destination_facing_and_preserves_origin(method):
    features = np.zeros((3, 5, 12))
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = 2
    features[:, 0, 9] = 0.1
    features[1:, 0, 3:9] = [0, 0, -1, 0, 1, 0]
    result = recover_positions(features, [-1, 0, 1, 2, 3], np.zeros((5, 3)), method,
                               root_origin=[3, 0, 4])
    np.testing.assert_allclose(result[:, 0], [[3, 2, 4], [3, 2, 4.1], [3, 2, 4.2]])


def test_recovery_rejects_unknown_mode_and_invalid_origin():
    features = np.zeros((1, 5, 12))
    features[..., 3] = features[..., 7] = 1
    with pytest.raises(ValueError, match='mode'):
        recover_positions(features, [-1, 0, 1, 2, 3], np.zeros((5, 3)), 'unknown')
    with pytest.raises(ValueError, match='origin'):
        recover_positions(features, [-1, 0, 1, 2, 3], np.zeros((5, 3)), 'fk', root_origin=[0, 2, 0])


def prepared_motion():
    asset = make_asset(synthetic_glb(True), 'rig.glb')
    conditioning, mapping = prepare_document(parse_glb(asset['glb'])[0], '+Z')
    rig = make_rig(asset, encode_arrays(**conditioning), mapping)
    features = np.zeros((7, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = conditioning['tpos_first_frame'][0, 1]
    return rig, make_motion(rig['rig_id'], encode_arrays(features=features),
                            {'canonical_root_origin': [3, 0, 4]}), conditioning


def test_portable_skeleton_preserves_identity_names_rest_offsets_and_origin():
    rig, motion, conditioning = prepared_motion()
    result = recover_skeleton(rig, motion, 'fk')
    validate_skeleton(result, rig['rig_id'])
    arrays = decode_arrays(result['arrays'])
    expected = conditioning['tpos_first_frame'] + [3, 0, 4]
    np.testing.assert_allclose(arrays['positions'], np.broadcast_to(expected, (7, 7, 3)), atol=1e-6)
    np.testing.assert_array_equal(arrays['joint_names'], conditioning['joint_names'])
    assert result['fps'] == 30
    result['arrays'] = encode_arrays(**{**arrays, 'positions': arrays['positions'] + 1})
    with pytest.raises(ValueError, match='digest'):
        validate_skeleton(result)


def test_recovery_rejects_other_rig_and_honors_cancellation():
    rig, motion, _ = prepared_motion()
    motion['rig_id'] = '0' * 64
    with pytest.raises(ValueError, match='identities'):
        recover_skeleton(rig, motion, 'fk')
    motion['rig_id'] = rig['rig_id']
    calls = []
    def cancel():
        calls.append(True)
        if len(calls) == 4:
            raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        recover_skeleton(rig, motion, 'fk', check_cancel=cancel)


def test_preview_renders_every_frame_in_fixed_clip_bounds():
    rig, motion, _ = prepared_motion()
    features = decode_arrays(motion['features'])['features']
    features[:, 0, 9] = 0.1
    motion['features'] = encode_arrays(features=features)
    skeleton = recover_skeleton(rig, motion, 'fk')
    images = render_skeleton(skeleton, 'front', 128)
    assert images.shape == (7, 128, 128, 3)
    assert images.dtype == np.float32
    assert np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    assert np.any(images[-1] != images[0])
    for view in ('side', 'top'):
        assert render_skeleton(skeleton, view, 64).shape == (7, 64, 64, 3)
    with pytest.raises(ValueError, match='projection'):
        render_skeleton(skeleton, 'unknown', 128)
    with pytest.raises(ValueError, match='budget'):
        render_skeleton(skeleton, 'front', 4096)


def test_preview_honors_cancellation():
    rig, motion, _ = prepared_motion()
    skeleton = recover_skeleton(rig, motion, 'fk')
    calls = []
    def cancel():
        calls.append(True)
        if len(calls) == 3:
            raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        render_skeleton(skeleton, 'front', 64, check_cancel=cancel)


def test_skeleton_crosses_actual_cloud_codec(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'ComfyUI-Cloud-Offload/partition_protocol.py'
    if not path.is_file():
        pytest.skip('Requires Cloud Offload node checkout')
    spec = importlib.util.spec_from_file_location('skeleton_cloud_protocol', path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    rig, motion, _ = prepared_motion()
    for method in ('fk', 'ric'):
        value = recover_skeleton(rig, motion, method)
        protocol.validate_boundary_type('UNIMATE_SKELETON')
        bundle = tmp_path / f'{method}.part'
        protocol.dump_bundle(value, bundle)
        restored = protocol.load_bundle(bundle)
        assert restored == value
        validate_skeleton(restored, rig['rig_id'])
