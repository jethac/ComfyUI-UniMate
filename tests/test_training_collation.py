"""Released collation compared as complete nested batch values."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from test_training_augmentation import sample
from unimate_pack import training_collation
from unimate_pack.training_samples import assemble_sample


@pytest.fixture(scope='module')
def reference():
    root=os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned source required')
    path=Path(root)/'unimate/dataset/mixture/collate.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '7cb7672ba97572282ec7a3153abd21aeb4917ecf7f77bf7d3079e13a9f888eb2'
    assert hashlib.sha256(Path(training_collation._source_collate.__code__.co_filename).read_bytes()).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('reference_collate',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mixture_batch_collate


def samples(dtype):
    values=[]
    for index in range(3):
        aug=sample(dtype=dtype)
        # Removal gives a genuinely smaller ordered topology.
        if index==1:
            from unimate_pack.training_augmentation import augment_sample
            aug,_=augment_sample(aug,'removal',17,removal_rate=.3)
        tokens=np.arange((index+1)*7,dtype=np.float32).reshape(index+1,7)/10
        value=assemble_sample(aug,object_type=f'fixture{index}',caption='walk',
            caption_tokens=tokens,caption_emb=tokens.mean(0),mode='first_frame',
            max_motion_length=12,max_joints=16)
        if index==2:
            value['spectral_feats']=value['spectral_feats'][:,:3].copy()
            value['motion_length']=0
            value['motion'][:]=0
        values.append(value)
    return values


def assert_equal(actual,expected):
    if isinstance(expected,torch.Tensor):
        assert actual.dtype==expected.dtype and actual.device==expected.device
        assert torch.equal(actual,expected)
    elif isinstance(expected,np.ndarray):
        np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:
            assert_equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for left,right in zip(actual,expected):
            assert_equal(left,right)
    else:
        assert actual==expected


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('optional',['all','mixed_caption','no_caption','partial_spectral','bare'])
def test_complete_batch_matches_released(reference,dtype,optional):
    batch=samples(dtype)
    if optional in ('mixed_caption','no_caption'):
        for value in (batch[1:] if optional=='mixed_caption' else batch):
            for key in ('caption_emb','caption_tokens','caption'):
                value.pop(key)
    if optional=='partial_spectral':
        batch[1].pop('spectral_feats')
    if optional=='bare':
        for value in batch:
            for key in ('spectral_feats','joint_names_emb','offsets','tpos_first_frame_parents',
                        'caption_emb','caption_tokens','caption','object_type'):
                value.pop(key)
    original=copy.deepcopy(batch)
    expected=reference(batch)
    actual=training_collation.collate_samples(batch)
    assert_equal(actual,expected)
    assert_equal(batch,original)
    if optional=='mixed_caption':
        assert actual[1]['caption_mask'][1,0]
        assert actual[1]['caption_tokens'][1].count_nonzero()==0
    for i,value in enumerate(batch):
        assert torch.all(actual[1]['std'][i,len(value['parents']):]==1)
    # Returned unpadded parents/edges must not alias mutable caller arrays.
    actual[1]['parents'][0][0]=4
    np.testing.assert_array_equal(batch[0]['parents'],original[0]['parents'])


@pytest.mark.parametrize('change',[
    lambda b: b[1].update(max_joints=17),
    lambda b: b[1].update(motion_length=100),
    lambda b: b[1].update(std=np.zeros((7,12))),
    lambda b: b[1].update(caption_tokens=np.ones((2,8))),
    lambda b: b[1].update(joint_relations=np.zeros((7,7),dtype=float)),
])
def test_invalid_batch_rejected(change):
    batch=samples(np.float32)
    change(batch)
    with pytest.raises(ValueError):
        training_collation.collate_samples(batch)


def test_budget_precedes_source_and_cancellation(monkeypatch):
    batch=samples(np.float32)
    def forbidden(*args):
        raise AssertionError('source allocated')
    monkeypatch.setattr(training_collation,'_source_collate',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        training_collation.collate_samples(batch,max_workspace_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        training_collation.collate_samples(batch,cancel=cancel)


def test_empty_filtered_batch(reference):
    assert training_collation.collate_samples([None])==reference([None])

@pytest.mark.parametrize('field,value',[
    ('std',np.float64(1e-100)),
    ('joint_relations',np.uint64(2**64-1)),
])
def test_rejects_values_unsafe_after_source_cast(field,value):
    batch=samples(np.float32)
    dtype=np.float64 if field=='std' else np.uint64
    batch[0][field]=batch[0][field].astype(dtype)
    batch[0][field].flat[0]=value
    with pytest.raises(ValueError):
        training_collation.collate_samples(batch)
