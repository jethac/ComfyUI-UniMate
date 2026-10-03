"""Training augmentation against unchanged released NumPy/Quaternion bodies."""

import ast
import copy
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unimate_pack import training_augmentation as augmentation
from unimate_pack._vendor import topology_utils as topology

SOURCE_HASHES = {
    'unimate/dataset/mixture/augmentations.py': '691158d3c25559060853b53099bc5b26357031bd51d016fc4a7732397a9feed3',
    'unimate/utils/motion_utils.py': 'cddb2436e8fc4d02a2f5715398fd689c730b364b425bd8057bf65f083a93dd30',
    'unimate/utils/rotation_conversions.py': '0b3df0c50126f2298de3d9512936e404504e2089bf2a0629c1cd2ade6eed012f',
    'unimate/utils/topology_utils.py': '96ba6f69536de7bca48c1f2710754bd82b3c7dff831c977696bbd2e998a77dfd',
    'unimate/dataset/mixture/dataset.py': '413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e',
}
MOTION_HASHES = {
    'Quaternions.py': '91ac78c4da0cf82d34dd14c42cacdec6a97ae7519c3987a0e0288adc0416b028',
    'Animation.py': 'edbcf8b46902b4819683961cb580e390e8ccf372fcf00b2f5ca7477c16e42897',
    'AnimationStructure.py': '860aec9e722e77d31aab4bcd67ce7abac447238b662a126da0e18da3d92f4e71',
}


@pytest.fixture(scope='module')
def reference():
    root, motion = os.environ.get('UNIMATE_DATASET_REFERENCE'), os.environ.get('UNIMATE_MOTION_REFERENCE')
    if not root or not motion:
        pytest.skip('Requires explicit pinned UniMate and reference-only Motion checkout')
    assert subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip() == (
        '2c5b384715aa63d8639b1ed7eb74bfe614570c7a')
    for path, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(topology.__file__).read_bytes()).hexdigest() == SOURCE_HASHES['unimate/utils/topology_utils.py']
    for path, digest in MOTION_HASHES.items():
        assert hashlib.sha256((Path(motion) / path).read_bytes()).hexdigest() == digest
    sys.path.insert(0, motion)
    from Quaternions import Quaternions
    import Animation
    assert Path(sys.modules['Quaternions'].__file__).resolve() == (Path(motion) / 'Quaternions.py').resolve()
    assert Path(Animation.__file__).resolve() == (Path(motion) / 'Animation.py').resolve()
    namespace = dict(np=np, random=random, Quaternions=Quaternions,
                     **{name: getattr(topology, name) for name in (
                         'compute_edge_relations_and_distances', 'compute_joint_depths',
                         'compute_edge_indexs', 'compute_laplacian_eigenvectors')})
    for path, names in (
        ('unimate/utils/rotation_conversions.py', {'rotation_6d_to_matrix_np'}),
        ('unimate/utils/motion_utils.py', {'compute_rifke', 'hml_rotations_to_bvh_quaternions', 'fk_global_positions', 'realign_unimate_clip'}),
        ('unimate/dataset/mixture/augmentations.py', None),
    ):
        source = Path(root) / path
        tree = ast.parse(source.read_text(encoding='utf-8'))
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    source = Path(root) / 'unimate/dataset/mixture/dataset.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionDataset')
    names = {'_extract_aug_dict', '_apply_augmentations', '_aug_addition', '_aug_removal', '_aug_pooling', '_aug_perturbation'}
    owner.body = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    owner.bases = []
    namespace['aug_ops'] = SimpleNamespace(**namespace)
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def sample(branching=True, dtype=np.float64):
    from scipy.spatial.transform import Rotation
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8] if branching else [-1] + list(range(9)))
    joints, frames = len(parents), 9
    rng = np.random.default_rng(14)
    offsets = rng.uniform(.05, .3, (joints, 3))
    offsets[0] = [0, .8, 0]
    tpos = offsets.copy()
    for j, p in enumerate(parents[1:], 1):
        tpos[j] += tpos[p]
    matrices = Rotation.from_rotvec(rng.uniform(-.4, .4, (frames * joints, 3))).as_matrix().reshape(frames, joints, 3, 3)
    motion = rng.uniform(-.1, .1, (frames, joints, 12))
    motion[:, 0, :3] = [0, .8, 0]
    facing = Rotation.from_euler('y', np.linspace(.2, .6, frames)[:, None]).as_matrix()
    motion[:, 0, 3:9] = np.concatenate([facing[:, :, 0], facing[:, :, 1]], axis=-1)
    for j, p in enumerate(parents[1:], 1):
        motion[:, j, 3:9] = np.concatenate([matrices[:, p, :, 0], matrices[:, p, :, 1]], axis=-1)
    relations, distances = topology.compute_edge_relations_and_distances(parents.tolist())
    spectral, _ = topology.compute_laplacian_eigenvectors(parents, max_freqs=8)
    value = dict(motion=motion, parents=parents, edge_indexs=topology.compute_edge_indexs(parents),
                 joint_graph_dist=distances, joint_relations=relations,
                 joint_depths=topology.compute_joint_depths(parents), spectral_feats=spectral,
                 tpos=tpos, offsets=offsets, joint_names_emb=rng.normal(size=(joints, 7)),
                 mean=rng.normal(size=(joints, 12)), std=rng.uniform(.5, 2, (joints, 12)))
    for field in ('motion', 'tpos', 'offsets', 'joint_names_emb', 'mean', 'std'):
        value[field] = value[field].astype(dtype)
    return value


def reference_call(reference, value, operation, seed, **options):
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        value = copy.deepcopy(value)
        if operation == 'none':
            return value
        if operation == 'random':
            owner = reference['MotionDataset']()
            for op in ('addition', 'removal', 'pooling', 'perturbation'):
                setattr(owner, 'use_' + op + '_aug', op in options.get('enabled', ()))
            owner.max_freqs = 8
            return owner._apply_augmentations(value, value['mean'], value['std'])
        if operation == 'addition':
            return reference['apply_joint_addition_ellipsoid'](value, 8, sigma=options.get('sigma', .5),
                                                              lateral_ratio=options.get('lateral_ratio', .5))
        if operation == 'addition_linear':
            return reference['apply_joint_addition_linear'](value, 8)
        if operation == 'removal':
            rate = options.get('removal_rate')
            return reference['apply_joint_removal'](value, random.uniform(.05, .15) if rate is None else rate, 8)
        if operation == 'pooling':
            rate = options.get('pool_rate')
            return reference['apply_skeleton_pooling'](value, random.uniform(.1, .3) if rate is None else rate, 8)
        return reference['apply_joint_perturbation'](value, max_scale=options.get('max_scale', .1))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@pytest.mark.parametrize('branching', [False, True])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('operation', ['none', 'addition', 'addition_linear', 'removal', 'pooling', 'perturbation'])
@pytest.mark.parametrize('seed', [0, 17])
def test_all_operations_match_pinned_reference(reference, branching, dtype, operation, seed):
    value = sample(branching, dtype)
    original = copy.deepcopy(value)
    py_state, np_state = random.getstate(), np.random.get_state()
    expected = reference_call(reference, value, operation, seed, removal_rate=.75, pool_rate=.75)
    actual, report = augmentation.augment_sample(value, operation, seed, removal_rate=.75, pool_rate=.75)
    assert report['operation'] == operation
    assert actual.keys() == expected.keys()
    tolerance = {np.float16: .004, np.float32: 3e-6, np.float64: 1e-12}[dtype]
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], atol=tolerance, rtol=tolerance, err_msg=field)
        assert actual[field].dtype == expected[field].dtype, field
        np.testing.assert_array_equal(value[field], original[field])
        assert not np.shares_memory(actual[field], value[field])
    if dtype in (np.float16, np.float32):
        np.testing.assert_array_equal(actual['motion'], expected['motion'])
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and after[2:] == np_state[2:]
    np.testing.assert_array_equal(after[1], np_state[1])


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('enabled', [(), ('addition',), ('removal', 'pooling'),
                                   ('addition', 'removal', 'pooling', 'perturbation')])
def test_random_wrapper_matches_released_noop_choice_and_rates(reference, seed, enabled):
    value = sample()
    expected = reference_call(reference, value, 'random', seed, enabled=enabled)
    actual, report = augmentation.augment_sample(value, 'random', seed, enabled=enabled)
    assert report['operation'] in ('none', *enabled)
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-12, err_msg=field)


def test_workspace_rejection_precedes_copy_and_cancellation_propagates(monkeypatch):
    value = sample()
    def forbidden_copy(*_):
        raise AssertionError('Copy reached before workspace rejection')
    monkeypatch.setattr(augmentation, '_copy_sample', forbidden_copy)
    with pytest.raises(ValueError, match='workspace'):
        augmentation.augment_sample(value, 'addition', 0, max_workspace_bytes=1)
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        augmentation.augment_sample(value, 'addition', 0, cancel=cancel)


@pytest.mark.parametrize('change', [
    lambda v: v['parents'].__setitem__(2, 9),
    lambda v: v['motion'].__setitem__((0, 0, 1), np.nan),
    lambda v: v['std'].__setitem__((0, 0), 0),
    lambda v: v.update(joint_names_emb=np.zeros((2, 7))),
    lambda v: v.update(extra=np.zeros(1)),
])
def test_invalid_samples_fail(change):
    value = sample()
    change(value)
    with pytest.raises(ValueError):
        augmentation.augment_sample(value, 'perturbation', 0)


@pytest.mark.parametrize('options', [dict(seed=True), dict(seed=2**32), dict(sigma=-.1),
                                    dict(max_scale=1.1), dict(removal_rate=-.1), dict(pool_rate=2),
                                    dict(lateral_ratio=-1), dict(enabled=('wrong',))])
def test_invalid_options_fail(options):
    seed = options.pop('seed', 0)
    with pytest.raises(ValueError):
        augmentation.augment_sample(sample(), 'addition', seed, **options)


@pytest.mark.parametrize('operation', ['addition', 'addition_linear', 'removal', 'pooling', 'perturbation'])
def test_augmented_fk_and_ric_views_agree_in_float64(operation):
    from unimate_pack.skeleton import recover_positions
    actual, _ = augmentation.augment_sample(sample(), operation, 17, removal_rate=.75, pool_rate=.75)
    fk = recover_positions(actual['motion'], actual['parents'], actual['offsets'], 'fk')
    ric = recover_positions(actual['motion'], actual['parents'], actual['offsets'], 'ric')
    np.testing.assert_allclose(fk, ric, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('operation', ['addition', 'addition_linear'])
@pytest.mark.parametrize('branching', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 17])
def test_addition_preserves_existing_fk_and_interpolates_embedding_and_stats(operation, branching, seed):
    from unimate_pack.skeleton import recover_positions
    original = sample(branching)
    actual, report = augmentation.augment_sample(original, operation, seed, addition_policy='neutral_fk')
    assert report['addition_policy'] == 'neutral_fk'
    joint = random.Random(seed).choice(list(range(1, len(original['parents']))))
    parent = original['parents'][joint]
    expected_fk = recover_positions(original['motion'], original['parents'], original['offsets'], 'fk')
    actual_fk = recover_positions(actual['motion'], actual['parents'], actual['offsets'], 'fk')
    np.testing.assert_allclose(np.delete(actual_fk, joint, axis=1), expected_fk, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(actual['joint_names_emb'][joint],
                                  (original['joint_names_emb'][parent] + original['joint_names_emb'][joint]) / 2)
    np.testing.assert_array_equal(actual['mean'][joint], original['mean'][joint])
    np.testing.assert_array_equal(actual['std'][joint], original['std'][joint])
    np.testing.assert_array_equal(actual['motion'][:, joint, 9:], 0)
    np.testing.assert_array_equal(np.delete(actual['motion'][..., 9:], joint, axis=1), original['motion'][..., 9:])


@pytest.mark.parametrize('operation', ['addition', 'addition_linear'])
def test_released_addition_duplicates_local_rotation_and_can_change_existing_pose(reference, operation):
    from unimate_pack.skeleton import recover_positions
    original = sample()
    actual, report = augmentation.augment_sample(original, operation, 17)
    expected = reference_call(reference, original, operation, 17)
    joint = random.Random(17).choice(list(range(1, len(original['parents']))))
    np.testing.assert_array_equal(actual['motion'][:, joint + 1, 3:9], original['motion'][:, joint, 3:9])
    np.testing.assert_allclose(actual['motion'], expected['motion'], atol=1e-12)
    before = recover_positions(original['motion'], original['parents'], original['offsets'], 'fk')
    after = recover_positions(actual['motion'], actual['parents'], actual['offsets'], 'fk')
    assert np.max(np.abs(np.delete(after, joint, axis=1) - before)) > .01
    assert report['addition_policy'] == 'released'


def test_cancellation_during_radius_rejection_loop_is_checked_each_iteration():
    class RejectingRng:
        def uniform(self, *_args, **_kwargs):
            return np.array([.5, 0., 0.])
        def normal(self, *_args):
            return 2.
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError('rejection loop cancelled')
    with pytest.raises(InterruptedError, match='rejection loop'):
        augmentation._ellipsoid(np.array([0., 1., 0.]), RejectingRng(), .5, .5, cancel)
    assert calls == 4


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_signed_diagonal_roundtrip_matches_reference_at_rotation_ties(reference, dtype):
    from scipy.spatial.transform import Rotation
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    matrices = Rotation.from_rotvec(axes * np.pi).as_matrix().astype(dtype)
    quaternions = reference['Quaternions'].from_transforms(matrices)
    np.testing.assert_allclose(augmentation._quaternion_roundtrip(matrices), quaternions.transforms(), atol=1e-14)
    vectors = np.repeat([[.2, .5, -.4]], len(axes), axis=0)
    np.testing.assert_allclose(np.einsum('tij,tj->ti', augmentation._quaternion_roundtrip(matrices, quaternion_vector=True), vectors),
                               quaternions * vectors, atol=1e-14)


@pytest.mark.parametrize('operation,options', [('addition', dict(sigma=0)),
                                             ('addition', dict(lateral_ratio=0)),
                                             ('perturbation', dict(max_scale=1)),
                                             ('removal', dict(removal_rate=0)),
                                             ('pooling', dict(pool_rate=0))])
def test_degenerate_parameter_boundaries_retain_valid_released_operations(reference, operation, options):
    value = sample()
    expected = reference_call(reference, value, operation, 17, **options)
    actual, _ = augmentation.augment_sample(value, operation, 17, **options)
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-12, err_msg=field)


@pytest.mark.parametrize('seed', [1, 5])
def test_random_wrapper_uses_released_parameters_instead_of_direct_operation_overrides(reference, seed):
    value = sample()
    enabled = ('addition', 'removal', 'pooling', 'perturbation')
    expected = reference_call(reference, value, 'random', seed, enabled=enabled)
    actual, _ = augmentation.augment_sample(value, 'random', seed, enabled=enabled,
        sigma=.8, lateral_ratio=.2, max_scale=.5, removal_rate=1, pool_rate=1, max_path_len=3)
    for field in actual:
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-12, err_msg=field)


@pytest.mark.parametrize('operation', ['removal', 'pooling'])
def test_structural_operation_report_keeps_released_default_path_limit(operation):
    _, report = augmentation.augment_sample(sample(), operation, 17, max_path_len=3,
                                            removal_rate=.75, pool_rate=.75)
    assert report['parameters']['max_path_len'] == 5

@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('branching',[False,True])
def test_crop_realign_matches_released_quaternions(reference,dtype,branching):
    from unimate_pack import training_transforms
    value=sample(branching,dtype)
    clip=value['motion'][2:].copy()
    before=clip.copy()
    expected=reference['realign_unimate_clip'](clip,value['parents'])
    actual=training_transforms.realign_clip(clip,value['parents'])
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(clip,before)
    np.testing.assert_array_equal(actual[...,:3],before[...,:3])
    np.testing.assert_array_equal(actual[...,9:],before[...,9:])

