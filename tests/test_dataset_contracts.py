import copy
import hashlib

import numpy as np
import pytest

from unimate_pack.contracts import decode_arrays, encode_arrays
from unimate_pack import dataset_contracts as dc


def dataset_parts(joints=5):
    parents = np.arange(joints, dtype=np.int64) - 1
    conditioning = encode_arrays(
        parents=parents, offsets=np.zeros((joints, 3)),
        tpos_first_frame=np.zeros((joints, 3)),
        joint_names=np.array([f'joint{i}' for i in range(joints)]),
        clean_joint_names=np.array([f'joint{i}' for i in range(joints)]),
    )
    features = encode_arrays(features=np.ones((3, joints, 12), dtype=np.float32))
    top_id = hashlib.sha256(conditioning).hexdigest()
    feat_id = hashlib.sha256(features).hexdigest()
    manifest = dict(topologies=[dict(id=top_id, source_rig_id='a' * 64, conditioning=top_id)],
                    clips=[dict(id='clip1', topology_id=top_id, dataset_type='truebones',
                                object_type='rig', caption='左足 / lift', split='train',
                                features=feat_id, origin=[1, 0, 3], fps=30)])
    return manifest, {top_id: conditioning, feat_id: features}


@pytest.mark.parametrize('joints', [2, 151, 4096])
def test_shard_preserves_payloads_copies_manifest_and_accepts_training_topologies_above_70(joints):
    manifest, files = dataset_parts(joints)
    value = dc.make_dataset(manifest, files)
    dc.validate_dataset(value)
    assert value['schema'] == 'unimate.dataset.v1'
    assert value['files'] == files
    manifest['clips'][0]['caption'] = 'changed'
    files.clear()
    assert value['manifest']['clips'][0]['caption'] == '左足 / lift'
    records = dc.training_records(value)
    assert records[0]['features'].shape == (3, joints, 12)
    assert type(records[0]['features']) is np.ndarray
    assert records[0]['features'].dtype == np.float32


def test_eval_clips_do_not_enter_statistics_and_shared_payloads_are_valid():
    manifest, files = dataset_parts()
    duplicate = dict(manifest['clips'][0], id='clip2', split='eval', caption='other caption')
    manifest['clips'].append(duplicate)
    value = dc.make_dataset(manifest, files)
    assert len(value['files']) == 2
    assert len(dc.training_records(value)) == 1
    value['manifest']['clips'][0]['split'] = 'eval'
    with pytest.raises(ValueError, match='training'):
        dc.training_records(value)


@pytest.mark.parametrize('change', [
    lambda m, f: m.update(extra=1),
    lambda m, f: m['clips'][0].update(split='test'),
    lambda m, f: m['clips'][0].update(fps=True),
    lambda m, f: m['clips'][0].update(origin=[0, np.nan, 0]),
    lambda m, f: m['clips'][0].update(id='../bad'),
    lambda m, f: m['clips'][0].update(caption='bad\0text'),
    lambda m, f: m['clips'][0].update(topology_id='b' * 64),
    lambda m, f: m['topologies'][0].update(source_rig_id='not-a-digest'),
    lambda m, f: m['clips'].append(copy.deepcopy(m['clips'][0])),
    lambda m, f: f.update({'c' * 64: b'unreferenced'}),
    lambda m, f: f.update({m['clips'][0]['features']: b'changed'}),
])
def test_malformed_manifest_and_digests_fail(change):
    manifest, files = dataset_parts()
    change(manifest, files)
    with pytest.raises(ValueError):
        dc.make_dataset(manifest, files)


def replace_payload(manifest, files, field, payload):
    entry = manifest['topologies'][0] if field == 'conditioning' else manifest['clips'][0]
    old = entry[field]
    new = hashlib.sha256(payload).hexdigest()
    files.pop(old)
    files[new] = payload
    entry[field] = new
    if field == 'conditioning':
        entry['id'] = new
        for clip in manifest['clips']:
            clip['topology_id'] = new


@pytest.mark.parametrize('kind', ['parents', 'joints', 'width', 'integer', 'extra_features'])
def test_digest_valid_payloads_still_require_valid_topology_and_features(kind):
    manifest, files = dataset_parts()
    if kind == 'parents':
        arrays = decode_arrays(files[manifest['topologies'][0]['conditioning']])
        arrays['parents'][1] = 2
        replace_payload(manifest, files, 'conditioning', encode_arrays(**arrays))
    else:
        shape = (3, 6 if kind == 'joints' else 5, 11 if kind == 'width' else 12)
        arrays = {'features': np.ones(shape, dtype=int if kind == 'integer' else np.float32)}
        if kind == 'extra_features':
            arrays['extra'] = np.ones(1)
        replace_payload(manifest, files, 'features', encode_arrays(**arrays))
    with pytest.raises(ValueError):
        dc.make_dataset(manifest, files)


def test_aggregate_expanded_budget_cannot_be_bypassed_by_individually_valid_arrays(monkeypatch):
    manifest, files = dataset_parts()
    # Compressed payloads fit; aggregate decoded feature/conditioning arrays do not.
    monkeypatch.setattr(dc, 'MAX_EXPANDED_BYTES', 100)
    def decode_sentinel(payload):
        raise AssertionError('decoded before aggregate expansion check')
    monkeypatch.setattr(dc, 'decode_arrays', decode_sentinel)
    with pytest.raises(ValueError, match='expanded'):
        dc.make_dataset(manifest, files)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_training_arrays_keep_original_precision_and_are_shared_read_only(dtype):
    manifest, files = dataset_parts()
    payload = encode_arrays(features=np.full((3, 5, 12), 0.125, dtype=dtype))
    replace_payload(manifest, files, 'features', payload)
    manifest['clips'].append(dict(manifest['clips'][0], id='clip2'))
    records = dc.training_records(dc.make_dataset(manifest, files))
    assert records[0]['features'].dtype == np.dtype(dtype)
    assert records[0]['features'] is records[1]['features']
    assert not records[0]['features'].flags.writeable


def test_cancellation_during_validation_and_training_extraction_preserves_value():
    manifest, files = dataset_parts()
    value = dc.make_dataset(manifest, files)
    before = copy.deepcopy(value)
    calls = 0

    def cancel_at(target):
        def cancel():
            nonlocal calls
            calls += 1
            if calls == target:
                raise InterruptedError('cancelled')
        return cancel

    with pytest.raises(InterruptedError):
        dc.validate_dataset(value, cancel=cancel_at(2))
    calls = 0
    with pytest.raises(InterruptedError):
        dc.training_records(value, cancel=cancel_at(5))
    assert value == before


@pytest.mark.parametrize('joints', [1, 4097])
def test_topology_joint_limits(joints):
    manifest, files = dataset_parts(joints)
    with pytest.raises(ValueError, match='parents'):
        dc.make_dataset(manifest, files)
