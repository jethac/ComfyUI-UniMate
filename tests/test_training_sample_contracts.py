"""Portable encoded samples preserve precision and explicit source identity."""
import hashlib

import numpy as np
import pytest

from test_training_augmentation import sample
from unimate_pack.training_samples import assemble_sample
from unimate_pack import training_sample_contracts as contracts

PROVENANCE=dict(dataset_sha256='a'*64,clip_id='clip',statistics_sha256='b'*64,text_cache_sha256='c'*64)
OPTIONS=dict(mode='tpos',crop_seed=17,start_idx=None,realign_feature=True,augmentation={})


def numeric(dtype=np.float32):
    return assemble_sample(sample(dtype=dtype),object_type='fixture',caption='walk',
        caption_tokens=np.ones((3,7),dtype=np.float32),caption_emb=np.ones(7,dtype=np.float32),
        mode='tpos',max_motion_length=4,max_joints=16,crop_seed=17)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_sample_roundtrip_precision_identity_and_ownership(dtype):
    source=numeric(dtype)
    value=contracts.make_training_sample(source,PROVENANCE,OPTIONS)
    result=contracts.validate_training_sample(value,expected_provenance=PROVENANCE)
    assert value['schema']=='unimate.training_sample.v1'
    assert hashlib.sha256(value['arrays']).hexdigest()==value['sha256']
    for key,expected in source.items():
        if isinstance(expected,np.ndarray):
            assert result[key].dtype==expected.dtype
            np.testing.assert_array_equal(result[key],expected)
            assert not np.shares_memory(result[key],expected)
        else:
            assert result[key]==expected
    result['parents'][0]=3
    assert contracts.validate_training_sample(value)['parents'][0]==-1
    assert source['parents'][0]==-1


@pytest.mark.parametrize('change',[
    lambda v:v.update(sha256='d'*64),
    lambda v:v['provenance'].update(clip_id='../clip'),
    lambda v:v['metadata'].update(motion_length=100),
    lambda v:v['options'].update(realign_feature=1),
    lambda v:v['options'].update(crop_seed=True),
    lambda v:v['metadata'].update(extra='field'),
])
def test_malformed_value_rejected(change):
    value=contracts.make_training_sample(numeric(),PROVENANCE,OPTIONS)
    change(value)
    with pytest.raises(ValueError):
        contracts.validate_training_sample(value)


def test_stale_expected_identity_and_precopy_budget(monkeypatch):
    value=contracts.make_training_sample(numeric(),PROVENANCE,OPTIONS)
    with pytest.raises(ValueError,match='identity'):
        contracts.validate_training_sample(value,expected_provenance={**PROVENANCE,'dataset_sha256':'f'*64})
    def forbidden(**arrays):
        raise AssertionError('encoder allocated before budget')
    monkeypatch.setattr(contracts,'encode_arrays',forbidden)
    with pytest.raises(ValueError,match='budget'):
        contracts.make_training_sample(numeric(),PROVENANCE,OPTIONS,max_array_bytes=1)


def test_sample_validation_never_allocates_source_tensors(monkeypatch):
    from unimate_pack import training_collation
    def forbidden(*args):
        raise AssertionError('collation called during portable validation')
    monkeypatch.setattr(training_collation,'_source_collate',forbidden)
    value=contracts.make_training_sample(numeric(),PROVENANCE,OPTIONS)
    contracts.validate_training_sample(value)


def test_decode_budget_preflight_and_explicit_crop_consistency(monkeypatch):
    value=contracts.make_training_sample(numeric(),PROVENANCE,OPTIONS)
    def forbidden(payload):
        raise AssertionError('decoded before workspace rejection')
    monkeypatch.setattr(contracts,'decode_arrays',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        contracts.validate_training_sample(value,max_workspace_bytes=1)


def test_explicit_crop_options_match_recorded_start():
    with pytest.raises(ValueError,match='crop'):
        contracts.make_training_sample(numeric(),PROVENANCE,{**OPTIONS,'start_idx':999})
