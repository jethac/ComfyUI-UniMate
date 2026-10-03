"""Portable source batches preserve tensors, topology and ordered provenance."""
import hashlib
import copy

import numpy as np
import pytest
import torch

from test_training_collation import samples,reference  # noqa: F401 fixture
from test_training_sample_contracts import PROVENANCE,OPTIONS
from unimate_pack import training_batch_contracts as contracts
from unimate_pack.training_sample_contracts import make_training_sample
from unimate_pack.contracts import decode_arrays,encode_arrays


def values():
    return [make_training_sample(s,{**PROVENANCE,'clip_id':f'clip{i}'},OPTIONS)
            for i,s in enumerate(samples(np.float64))]


def equal(actual,expected):
    if isinstance(expected,torch.Tensor):
        assert actual.device.type=='cpu' and actual.dtype==expected.dtype
        assert torch.equal(actual,expected)
    elif isinstance(expected,np.ndarray):
        np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:
            equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected):
            equal(a,e)
    else:
        assert actual==expected


def test_source_batch_roundtrip(reference):  # noqa: F811 fixture
    source=samples(np.float64)
    portable=contracts.collate_training_samples(values())
    equal(contracts.validate_training_batch(portable),reference(source))
    assert [p['provenance']['clip_id'] for p in portable['samples']]==['clip0','clip1','clip2']
    first=contracts.validate_training_batch(portable)
    first[0].fill_(9)
    equal(contracts.validate_training_batch(portable),reference(source))


def test_repeated_samples_and_empty_rejection():
    v=values()[0]
    result=contracts.collate_training_samples([v,v])
    assert len(result['samples'])==2
    assert result['samples'][0]==result['samples'][1]
    with pytest.raises(ValueError):
        contracts.collate_training_samples([])


@pytest.mark.parametrize('field',['sha256','identity_sha256','upstream_revision'])
def test_invalid_envelope(field):
    v=contracts.collate_training_samples(values())
    v[field]='f'*64
    with pytest.raises(ValueError):
        contracts.validate_training_batch(v)


@pytest.mark.parametrize('change',[
    lambda a:a['joint_mask'].__setitem__((0,0,0,15),True),
    lambda a:a['std'].__setitem__((0,15,0),0),
    lambda a:a['motion'].__setitem__((0,15,0,0),1),
    lambda a:a['parents'].__setitem__(0,0),
    lambda a:a.update(caption_mask=a['caption_mask'].astype(np.int64)),
])
def test_forged_numeric_invariants(change):
    v=contracts.collate_training_samples(values())
    arrays=decode_arrays(v['arrays'])
    change(arrays)
    v['arrays']=encode_arrays(**arrays)
    v['sha256']=hashlib.sha256(v['arrays']).hexdigest()
    v['identity_sha256']=contracts._identity(v)
    with pytest.raises(ValueError):
        contracts.validate_training_batch(v)


def test_predecode_budget_and_cancel(monkeypatch):
    v=contracts.collate_training_samples(values())
    def forbidden(*args,**kwargs):
        raise AssertionError('decoder reached before budget rejection')
    monkeypatch.setattr(contracts,'decode_arrays',forbidden)
    with pytest.raises(ValueError,match='budget'):
        contracts.validate_training_batch(v,max_workspace_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        contracts.validate_training_batch(v,cancel=cancel)


def test_aggregate_inputs_reject_before_decode(monkeypatch):
    v=values()[0]
    import unimate_pack.training_batch_contracts as module
    def forbidden(*args,**kwargs):
        raise AssertionError('sample decoder reached')
    monkeypatch.setattr(module,'validate_training_sample',forbidden)
    with pytest.raises(ValueError,match='budget'):
        module.collate_training_samples([v,v],max_workspace_bytes=1)


@pytest.mark.parametrize('key,text',[('caption',''),('caption','x\0y'),('object_type','../bad')])
def test_invalid_batch_labels(key,text):
    v=copy.deepcopy(contracts.collate_training_samples(values()))
    v['metadata'][key][0]=text
    v['identity_sha256']=contracts._identity(v)
    with pytest.raises(ValueError):
        contracts.validate_training_batch(v)


def test_original_parent_dtype_retained():
    from unimate_pack.training_sample_contracts import validate_training_sample
    numeric=validate_training_sample(values()[0])
    numeric['parents']=numeric['parents'].astype(np.int16)
    portable=make_training_sample(numeric,PROVENANCE,OPTIONS)
    batch=contracts.collate_training_samples([portable])
    _,cond=contracts.validate_training_batch(batch)
    assert cond['parents'][0].dtype==np.int16
