import ast
import copy
from collections import defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unimate_pack import dataset_selection as selection
from unimate_pack.dataset_contracts import make_dataset
from test_dataset_contracts import dataset_parts


def mixed_dataset():
    manifest, files = dataset_parts()
    template = manifest['clips'][0]
    clips = []
    for dataset, obj, count in (('truebones', 'shared', 4), ('truebones', 'other', 2),
                                ('mixamo', 'shared', 3), ('mixamo', 'small', 1),
                                ('objaverse', 'first', 3), ('objaverse', 'second', 2),
                                ('objaverse', 'third', 1)):
        for _ in range(count):
            clips.append(dict(template, id=f'clip-{len(clips)}', dataset_type=dataset, object_type=obj))
    manifest['clips'] = clips
    return make_dataset(manifest, files)


@pytest.fixture(scope='module')
def reference():
    root = os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Requires pinned released dataset source')
    assert subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip() == (
        '2c5b384715aa63d8639b1ed7eb74bfe614570c7a')
    source = Path(root) / 'unimate/dataset/mixture/dataset.py'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e'
    tree = ast.parse(raw.decode('utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionDataset')
    methods = {'_split_names', '_eval_clips_explicit', '_eval_clips_by_object_type', '_eval_clips_by_clip'}
    owner.body = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    owner.bases = []
    sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MixtureSampler')
    class LegacySamplerBase(torch.utils.data.Sampler):
        # Torch 2.11 removed the deprecated no-op data_source constructor.
        def __init__(self, data_source):
            pass
    namespace = dict(np=np, torch=torch, data=SimpleNamespace(Sampler=LegacySamplerBase), random=random,
                     defaultdict=defaultdict, logger=logging.getLogger('selection-reference'))
    exec(compile(ast.Module(body=[owner, sampler], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('ratio', [0, .1, .5, .99])
@pytest.mark.parametrize('seed', [0, 17, 2**64-1])
@pytest.mark.parametrize('options', [{}, {'modes': {'truebones': 'clip', 'mixamo': 'object_type'}},
                                    {'explicit_eval_objects': {'truebones': ['other', 'missing']}}])
def test_splits_match_released_membership_order_and_preserve_source(reference, ratio, seed, options):
    dataset = mixed_dataset()
    original = copy.deepcopy(dataset)
    clips = dataset['manifest']['clips']
    modes = dict(truebones='object_type', objaverse='object_type', mixamo='clip', **{})
    modes.update(options.get('modes', {}))
    train, evaluation = reference['MotionDataset']._split_names(
        [clip['id'] for clip in clips], {clip['id']: clip for clip in clips}, ratio, seed,
        modes, options.get('explicit_eval_objects'))
    actual, report = selection.split_dataset(dataset, ratio, seed, json.dumps(options))
    assert [c['id'] for c in actual['manifest']['clips'] if c['split'] == 'train'] == train
    assert [c['id'] for c in actual['manifest']['clips'] if c['split'] == 'eval'] == evaluation
    assert [c['id'] for c in actual['manifest']['clips']] == [c['id'] for c in clips]
    assert actual['files'] == dataset['files']
    assert report['train_ids'] == train and report['eval_ids'] == evaluation
    assert dataset == original


@pytest.mark.parametrize('options', ['{"extra":1}', '{"modes":{"mixamo":"wrong"}}',
                                    '{"modes":{},"modes":{}}', '{"explicit_eval_objects":{"mixamo":"shared"}}',
                                    '{"explicit_eval_objects":{"truebones":["shared","other"]}}'])
def test_invalid_options_and_drained_dataset_fail(options):
    with pytest.raises(ValueError):
        selection.split_dataset(mixed_dataset(), .5, 0, options)


def test_split_report_includes_explicit_objects_in_an_absent_dataset():
    dataset = mixed_dataset()
    actual, report = selection.split_dataset(dataset, 0, 0,
        '{"explicit_eval_objects":{"absent_dataset":["missing","missing"]}}')
    assert actual == dataset
    assert report['missing_objects'] == {'absent_dataset': ['missing']}


@pytest.mark.parametrize('alpha', [0, .5, 1, -.5, 2])
@pytest.mark.parametrize('dataset_alpha', [None, 0, .25, 1])
@pytest.mark.parametrize('epoch', [0, 3, 2**64-1])
def test_sampling_weights_and_indices_match_released_epoch_sampler(reference, alpha, dataset_alpha, epoch):
    dataset = mixed_dataset()
    dataset['manifest']['clips'][-1]['split'] = 'eval'
    clips = [c for c in dataset['manifest']['clips'] if c['split'] == 'train']
    owner = SimpleNamespace(motion_dataset=SimpleNamespace(
        train_name_list=[c['id'] for c in clips], train_motion_dict={c['id']: c for c in clips}))
    expected = reference['MixtureSampler'](owner, alpha, dataset_alpha)
    expected.set_epoch(epoch)
    rng_before = torch.random.get_rng_state().clone()
    value = selection.sampling_plan(dataset, alpha=alpha, dataset_alpha=dataset_alpha, epoch=epoch)
    selection.validate_sampling(value, dataset)
    from unimate_pack.contracts import decode_arrays
    arrays = decode_arrays(value['arrays'])
    np.testing.assert_array_equal(arrays['weights'], expected.weights.numpy())
    np.testing.assert_array_equal(arrays['indices'], list(expected))
    assert value['clip_ids'] == [c['id'] for c in clips]
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_sampling_identity_validation_and_cancellation():
    dataset = mixed_dataset()
    value = selection.sampling_plan(dataset)
    changed = copy.deepcopy(dataset)
    changed['manifest']['clips'][0]['caption'] = 'other'
    with pytest.raises(ValueError, match='source|identity'):
        selection.validate_sampling(value, changed)
    def cancel():
        raise InterruptedError('stop')
    with pytest.raises(InterruptedError):
        selection.split_dataset(dataset, .2, 0, '{}', cancel=cancel)
    with pytest.raises(InterruptedError):
        selection.sampling_plan(dataset, cancel=cancel)


@pytest.mark.parametrize('field,value', [('alpha', True), ('alpha', float('nan')),
                                       ('dataset_alpha', float('inf')), ('epoch', True), ('epoch', -1)])
def test_sampling_invalid_options_fail(field, value):
    with pytest.raises(ValueError):
        selection.sampling_plan(mixed_dataset(), **{field: value})


@pytest.mark.parametrize('ratio,seed', [(-.1, 0), (1, 0), (float('nan'), 0), (.5, True), (.5, -1)])
def test_split_invalid_scalar_options_fail(ratio, seed):
    with pytest.raises(ValueError):
        selection.split_dataset(mixed_dataset(), ratio, seed, '{}')


@pytest.mark.parametrize('change', [
    lambda v: v.update(schema='wrong'), lambda v: v.update(epoch=True),
    lambda v: v.update(upstream_revision='wrong'), lambda v: v.update(extra=0),
    lambda v: v.update(sha256='b' * 64), lambda v: v['clip_ids'].append(v['clip_ids'][0]),
    lambda v: v.update(dataset_sha256='wrong'), lambda v: v.update(arrays=b'broken'),
])
def test_malformed_sampling_contract_fails(change):
    value = selection.sampling_plan(mixed_dataset())
    change(value)
    with pytest.raises(ValueError):
        selection.validate_sampling(value)


def test_sampling_rejects_wrong_epoch_arrays_even_with_recomputed_digest():
    dataset = mixed_dataset()
    value = selection.sampling_plan(dataset)
    value['epoch'] += 1
    with pytest.raises(ValueError, match='source/options/epoch'):
        selection.validate_sampling(value, dataset)
