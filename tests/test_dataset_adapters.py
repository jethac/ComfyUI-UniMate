import copy
import hashlib
import json

import numpy as np
import pytest

from unimate_pack import dataset_builder, statistics, statistics_io
from unimate_pack.contracts import decode_arrays
from unimate_pack.dataset_contracts import training_records
from unimate_pack.dataset_stats import compute_statistics
from test_foot_lock import portable_legged_motion


def paired_dataset():
    rig, motion, _ = portable_legged_motion()
    labels = json.dumps([{'id': 'a', 'caption': '左 / lift'}, {'id': 'b', 'split': 'eval'}])
    value = dataset_builder.build_dataset([rig, rig], [motion, motion], labels, 'objaverse')
    return value, rig, motion


def test_builder_preserves_source_bytes_identity_origin_labels_and_deduplicates():
    value, rig, motion = paired_dataset()
    manifest = value['manifest']
    assert len(manifest['topologies']) == 1
    assert len(value['files']) == 2
    assert manifest['topologies'][0]['source_rig_id'] == rig['rig_id']
    assert value['files'][manifest['topologies'][0]['conditioning']] == rig['conditioning']
    assert value['files'][manifest['clips'][0]['features']] == motion['features']
    assert manifest['clips'][0]['origin'] == [3, 0, 4]
    assert manifest['clips'][0]['caption'] == '左 / lift'
    assert manifest['clips'][1]['split'] == 'eval'
    assert len(training_records(value)) == 1


@pytest.mark.parametrize('kind', ['length', 'identity', 'labels_length', 'labels_extra', 'labels_duplicate'])
def test_builder_rejects_ambiguous_or_mismatched_pairs(kind):
    rig, motion, _ = portable_legged_motion()
    rigs, motions, labels = [rig, rig], [motion, motion], '[]'
    if kind == 'length':
        rigs.pop()
    elif kind == 'identity':
        motion['rig_id'] = 'b' * 64
    elif kind == 'labels_length':
        labels = '[{}]'
    elif kind == 'labels_extra':
        labels = '[{"path":"bad"},{}]'
    else:
        labels = '[{"id":"same"},{"id":"same"}]'
    with pytest.raises(ValueError):
        dataset_builder.build_dataset(rigs, motions, labels, 'objaverse')


def test_builder_cancellation_preserves_source_and_defaults_use_prompt():
    rig, motion, _ = portable_legged_motion()
    motion['metadata']['prompt'] = 'walk'
    source = copy.deepcopy(motion)
    value = dataset_builder.build_dataset([rig], [motion], '[]', 'mixamo')
    assert value['manifest']['clips'][0]['caption'] == 'walk'
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        dataset_builder.build_dataset([rig], [motion], '[]', 'mixamo', cancel=cancel)
    assert motion == source


def test_distinct_mesh_rigs_with_identical_conditioning_share_topology_and_keep_clip_provenance():
    from unimate_pack.contracts import make_asset, make_rig, make_motion
    from unimate_pack.rig_math import parse_glb, pack_glb
    rig, motion, _ = portable_legged_motion()
    document, binary = parse_glb(rig['asset']['glb'])
    document['materials'][0]['name'] = 'different appearance metadata'
    other = make_rig(make_asset(pack_glb(document, binary), 'other.glb'),
                     rig['conditioning'], rig['mapping'])
    other_motion = make_motion(other['rig_id'], motion['features'], motion['metadata'])
    assert other['rig_id'] != rig['rig_id']
    dataset = dataset_builder.build_dataset([rig, other], [motion, other_motion], '[]', 'objaverse')
    assert len(dataset['manifest']['topologies']) == 1
    assert dataset['manifest']['topologies'][0]['source_rig_id'] is None
    assert [clip['source_rig_id'] for clip in dataset['manifest']['clips']] == [rig['rig_id'], other['rig_id']]


@pytest.mark.parametrize('per_dataset', [False, True])
@pytest.mark.parametrize('balanced', [False, True])
@pytest.mark.parametrize('tie_std', [False, True])
def test_portable_statistics_match_numeric_results_and_authenticate_source(per_dataset, balanced, tie_std):
    dataset, _, _ = paired_dataset()
    value = statistics.dataset_statistics(dataset, per_dataset=per_dataset, balanced=balanced, tie_std=tie_std)
    statistics.validate_statistics(value)
    assert value['clip_count'] == 1
    assert value['datasets'] == ['objaverse']
    assert value['sha256'] == hashlib.sha256(value['arrays']).hexdigest()
    expected = compute_statistics(training_records(dataset), per_dataset=per_dataset,
                                  balanced=balanced, tie_std=tie_std)
    actual = decode_arrays(value['arrays'])
    for field in expected['objaverse']:
        np.testing.assert_array_equal(actual[field][0], expected['objaverse'][field])
    changed = copy.deepcopy(dataset)
    changed['manifest']['clips'][0]['caption'] = 'other source'
    other = statistics.dataset_statistics(changed)
    assert other['dataset_sha256'] != value['dataset_sha256']


def test_statistics_archive_roundtrip_preserves_contract_and_original_array_bytes():
    dataset, _, _ = paired_dataset()
    value = statistics.dataset_statistics(dataset)
    payload = statistics_io.dump_statistics(value)
    assert statistics_io.load_statistics(payload) == value
    assert statistics_io.dump_statistics(value) == payload


@pytest.mark.parametrize('outer', [False, True])
def test_statistics_expansion_budget_rejects_before_numeric_decode(monkeypatch, outer):
    import io
    dataset, _, _ = paired_dataset()
    value = statistics.dataset_statistics(dataset)
    stream = io.BytesIO()
    np.savez_compressed(stream, padding=np.zeros(4096, dtype=np.float64))
    payload = stream.getvalue()
    assert len(payload) < 1024
    def forbidden_decode(_):
        raise AssertionError('Numeric decode reached before archive size rejection')
    monkeypatch.setattr(statistics, 'MAX_STATISTICS_ARRAY_BYTES', 1024, raising=False)
    monkeypatch.setattr(statistics_io, 'MAX_STATISTICS_ARCHIVE_BYTES', 1024, raising=False)
    monkeypatch.setattr(statistics, 'decode_arrays', forbidden_decode)
    monkeypatch.setattr(statistics_io, 'decode_arrays', forbidden_decode)
    with pytest.raises(ValueError, match='large|size limit'):
        if outer:
            statistics_io.load_statistics(payload)
        else:
            value.update(arrays=payload, sha256=hashlib.sha256(payload).hexdigest())
            statistics.validate_statistics(value)


@pytest.mark.parametrize('change', [
    lambda v: v.update(clip_count=True),
    lambda v: v.update(upstream_revision='wrong'),
    lambda v: v['datasets'].append('objaverse'),
    lambda v: v['options'].update(balanced=1),
    lambda v: v.update(arrays=b'changed'),
    lambda v: v.update(dataset_sha256='wrong'),
    lambda v: v.update(extra=1),
])
def test_malformed_statistics_rejected(change):
    dataset, _, _ = paired_dataset()
    value = statistics.dataset_statistics(dataset)
    change(value)
    with pytest.raises(ValueError):
        statistics.validate_statistics(value)
