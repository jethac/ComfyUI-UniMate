import ast
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from unimate_pack import dataset_stats


def clips():
    rng = np.random.default_rng(71)
    result = []
    for dataset, name, frames, joints, shift in (
        ('truebones', 'shared', 5, 5, 0),
        ('objaverse', 'shared', 11, 7, 10),
        ('mixamo', 'other', 3, 6, 20),
        ('truebones', 'second', 7, 8, -4),
    ):
        features = rng.normal(size=(frames, joints, 12)) + shift
        features[:, 0, :] += 3
        features[..., 11] = 2
        result.append(dict(dataset_type=dataset, object_type=name, features=features))
    return result


@pytest.fixture(scope='module')
def reference():
    root = os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Requires pinned released dataset source')
    revision = subprocess.run(
        ['git', '-C', root, 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    assert revision == '2c5b384715aa63d8639b1ed7eb74bfe614570c7a'
    source = Path(root) / 'unimate/dataset/mixture/dataset.py'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        '413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e'
    )
    tree = ast.parse(raw.decode('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionDataset')
    names = {'_calculate_dataset_stats', '_stats_frame_weighted', '_stats_balanced'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 3
    namespace = dict(np=np, defaultdict=defaultdict, logger=SimpleNamespace(info=lambda *a: None))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('per_dataset', [False, True])
@pytest.mark.parametrize('balanced', [False, True])
@pytest.mark.parametrize('tie_std', [False, True])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_all_modes_match_pinned_reference(reference, per_dataset, balanced, tie_std, dtype):
    records = clips()
    for record in records:
        record['features'] = record['features'].astype(dtype)
    owner = SimpleNamespace(
        feature_len=12, _stats_path=None, dataset_stats={},
        dataset_object_count={r['dataset_type']: {} for r in records},
        train_motion_dict={str(i): dict(dataset_type=r['dataset_type'],
                                      object_type=r['object_type'], motion=r['features'])
                           for i, r in enumerate(records)},
        use_dataset_stats=per_dataset, balanced_stats=balanced, tie_std=tie_std,
    )
    for name in ('_stats_frame_weighted', '_stats_balanced'):
        setattr(owner, name, MethodType(reference[name], owner))
    reference['_calculate_dataset_stats'](owner)
    before = [r['features'].copy() for r in records]
    actual = dataset_stats.compute_statistics(
        records, per_dataset=per_dataset, balanced=balanced, tie_std=tie_std,
    )
    assert set(actual) == set(owner.dataset_stats)
    for dataset, expected in owner.dataset_stats.items():
        assert set(actual[dataset]) == set(expected)
        for field, values in expected.items():
            assert actual[dataset][field].dtype == np.float64
            np.testing.assert_allclose(actual[dataset][field], values, rtol=1e-12, atol=1e-12)
    for record, original in zip(records, before):
        np.testing.assert_array_equal(record['features'], original)
    if not per_dataset:
        keys = list(actual)
        assert not np.shares_memory(actual[keys[0]]['mean_root'], actual[keys[1]]['mean_root'])


def test_root_local_counts_constant_floor_and_duplicate_object_grouping():
    records = [dict(dataset_type=ds, object_type=obj,
                    features=np.full((frames, joints, 12), value, dtype=float))
               for ds, obj, frames, joints, value in
               [('truebones', 'shared', 2, 5, 0), ('objaverse', 'shared', 4, 7, 10),
                ('mixamo', 'other', 1, 6, 20)]]
    frame = dataset_stats.compute_statistics(records, per_dataset=False)['truebones']
    balanced = dataset_stats.compute_statistics(records, per_dataset=False, balanced=True)['truebones']
    np.testing.assert_allclose(frame['mean_root'], 60 / 7)
    np.testing.assert_allclose(frame['mean_local'], 340 / 37)
    np.testing.assert_allclose(balanced['mean_root'], 40 / 3)
    np.testing.assert_allclose(balanced['mean_local'], 13.75)
    constant = dataset_stats.compute_statistics(records[:1])['truebones']
    np.testing.assert_array_equal(constant['std_root'], np.full(12, 1e-8))
    np.testing.assert_array_equal(constant['std_local'], np.full(12, 1e-8))


@pytest.mark.parametrize('change', [
    lambda r: [],
    lambda r: [dict(r[0], extra=1)],
    lambda r: [dict(r[0], dataset_type='')],
    lambda r: [dict(r[0], object_type='../rig')],
    lambda r: [dict(r[0], features=np.ones((0, 5, 12)))],
    lambda r: [dict(r[0], features=np.ones((2, 1, 12)))],
    lambda r: [dict(r[0], features=np.ones((2, 5, 11)))],
    lambda r: [dict(r[0], features=np.ones((2, 5, 12), dtype=int))],
    lambda r: [dict(r[0], features=np.full((2, 5, 12), np.nan))],
    lambda r: [dict(r[0], features=np.full((2, 5, 12), 1e308))],
])
def test_invalid_clips_rejected(change):
    with pytest.raises(ValueError):
        dataset_stats.compute_statistics(change(clips()))


@pytest.mark.parametrize('options', [{'balanced': 1}, {'per_dataset': None}, {'tie_std': 'yes'}])
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        dataset_stats.compute_statistics(clips(), **options)


def test_workspace_guard_and_cancellation(monkeypatch):
    records = clips()
    monkeypatch.setattr(dataset_stats, 'MAX_WORKSPACE_BYTES', 100)
    with pytest.raises(ValueError, match='workspace'):
        dataset_stats.compute_statistics(records)
    monkeypatch.setattr(dataset_stats, 'MAX_WORKSPACE_BYTES', 512 * 1024 * 1024)
    before = [r['features'].copy() for r in records]
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == len(records) + 2:
            raise InterruptedError('cancelled during accumulation')

    with pytest.raises(InterruptedError):
        dataset_stats.compute_statistics(records, balanced=True, cancel=cancel)
    for record, original in zip(records, before):
        np.testing.assert_array_equal(record['features'], original)


def test_workspace_includes_balanced_group_and_output_arrays(monkeypatch):
    records = [dict(dataset_type='truebones', object_type=f'rig{i}',
                    features=np.zeros((1, 2, 12))) for i in range(20)]
    # Clip reductions fit; retaining twenty sets of balanced moments does not.
    monkeypatch.setattr(dataset_stats, 'MAX_WORKSPACE_BYTES', 6000)
    with pytest.raises(ValueError, match='workspace'):
        dataset_stats.compute_statistics(records, balanced=True)


def test_masked_nonfinite_features_cannot_fabricate_statistics():
    features = np.ma.array(np.ones((1, 2, 12)), mask=False)
    features[0, 0, 0] = np.ma.masked
    features.data[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        dataset_stats.compute_statistics([
            dict(dataset_type='truebones', object_type='rig', features=features),
        ])
