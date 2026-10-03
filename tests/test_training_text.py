"""Portable ragged text views and exact released pooling policies."""
import hashlib

import numpy as np
import pytest

from unimate_pack import training_text

IDENTITY=dict(encoder_type='t5',encoder_version='google/flan-t5-base',artifact_sha256='a'*64)


def encoder(texts):
    sequences=[np.arange((len(text)%4+1)*7,dtype=np.float32).reshape(-1,7)/11 for text in texts]
    for i,text in enumerate(texts):
        if text=='':
            sequences[i][:]=0
    pooled=np.stack([seq.mean(0)+np.float32(.000001) if text else seq.mean(0)
                     for text,seq in zip(texts,sequences)])
    return sequences,pooled


def test_ragged_cache_roundtrip_and_distinct_pooling_views():
    calls=[]
    def counted(texts):
        calls.append(texts)
        return encoder(texts)
    cache=training_text.build_text_cache(['root','walk','','root','left knee'],counted,IDENTITY,chunk_size=2)
    assert calls==[['root','walk'],['','left knee']]
    training_text.validate_text_cache(cache,encoder_identity=IDENTITY)
    assert cache['schema']=='unimate.text_cache.v1'
    for text in cache['texts']:
        seq,pool=encoder([text])
        actual_seq,actual_pool=training_text.text_views(cache,text,policy='cached')
        np.testing.assert_array_equal(actual_seq,seq[0])
        np.testing.assert_array_equal(actual_pool,pool[0])
        _,fresh_pool=training_text.text_views(cache,text,policy='fresh')
        np.testing.assert_array_equal(fresh_pool,seq[0].mean(axis=0))
    assert hashlib.sha256(cache['arrays']).hexdigest()==cache['sha256']


def test_cache_hits_do_not_instantiate_encoder():
    cache=training_text.build_text_cache(['root','walk'],encoder,IDENTITY)
    def forbidden(texts):
        raise AssertionError('encoder called for complete cache')
    rebuilt=training_text.build_text_cache(['walk','root'],forbidden,IDENTITY,existing=cache)
    assert rebuilt['arrays']==cache['arrays']
    assert rebuilt['texts']==cache['texts']


@pytest.mark.parametrize('change',[
    lambda c:c.update(sha256='b'*64),
    lambda c:c.update(texts=['duplicate','duplicate']),
    lambda c:c['encoder'].update(artifact_sha256='invalid'),
    lambda c:c.update(schema='unimate.text_cache.v2'),
])
def test_malformed_cache(change):
    cache=training_text.build_text_cache(['root','walk'],encoder,IDENTITY)
    change(cache)
    with pytest.raises(ValueError):
        training_text.validate_text_cache(cache)


def test_encoder_identity_missing_text_and_policy_fail():
    cache=training_text.build_text_cache(['root'],encoder,IDENTITY)
    with pytest.raises(ValueError):
        training_text.validate_text_cache(cache,encoder_identity={**IDENTITY,'artifact_sha256':'b'*64})
    with pytest.raises(ValueError):
        training_text.text_views(cache,'absent')
    with pytest.raises(ValueError):
        training_text.text_views(cache,'root',policy='arbitrary')


def test_budget_cancellation_and_invalid_encoder_output():
    with pytest.raises(ValueError,match='budget'):
        training_text.build_text_cache(['root'],encoder,IDENTITY,max_array_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        training_text.build_text_cache(['root'],encoder,IDENTITY,cancel=cancel)
    with pytest.raises(ValueError):
        training_text.build_text_cache(['root'],lambda texts:([np.full((1,7),np.nan)],np.zeros((1,7))),IDENTITY)


def test_fresh_pooling_rejects_overflow_without_changing_source_arithmetic():
    def overflowing(texts):
        return [np.full((2,1),np.finfo(np.float32).max,dtype=np.float32)],np.zeros((1,1),dtype=np.float32)
    cache=training_text.build_text_cache(['root'],overflowing,IDENTITY)
    with pytest.raises(ValueError,match='pooled'):
        training_text.text_views(cache,'root',policy='fresh')
