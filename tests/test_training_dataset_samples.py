"""Dataset-bound production from actual portable inputs."""
import copy

import numpy as np
import pytest

from test_dataset_adapters import paired_dataset
from unimate_pack.contracts import decode_arrays
from unimate_pack.statistics import dataset_statistics
from unimate_pack.training_text import build_text_cache
from unimate_pack.training_sample_contracts import validate_training_sample
from unimate_pack import training_dataset_samples as producer

IDENTITY=dict(encoder_type='t5',encoder_version='google/flan-t5-base',artifact_sha256='a'*64)


def inputs():
    dataset,rig,_=paired_dataset()
    stats=dataset_statistics(dataset)
    cond=decode_arrays(rig['conditioning'])
    names=cond['clean_joint_names'].tolist()
    texts=names+[dataset['manifest']['clips'][0]['caption']]
    def encode(texts):
        seq=[np.arange(14,dtype=np.float32).reshape(2,7)/10 for _ in texts]
        return seq,np.stack([s.mean(0) for s in seq])
    cache=build_text_cache(texts,encode,IDENTITY)
    return dataset,stats,cache


@pytest.mark.parametrize('operation',['none','addition','removal','pooling','perturbation'])
@pytest.mark.parametrize('mode',['tpos','first_frame'])
def test_producer_binds_identity_and_source_arrays(operation,mode):
    dataset,stats,cache=inputs()
    original=copy.deepcopy((dataset,stats,cache))
    value=producer.produce_training_sample(dataset,stats,cache,'a',mode=mode,
        max_motion_length=8,max_joints=16,augmentation=operation,augmentation_seed=17,crop_seed=11)
    result=validate_training_sample(value)
    assert value['provenance']['clip_id']=='a'
    assert value['provenance']['dataset_sha256']==stats['dataset_sha256']
    assert value['provenance']['statistics_sha256']==producer.statistics_identity(stats)
    assert value['provenance']['text_cache_sha256']==producer.text_cache_identity(cache)
    assert result['motion'].shape[0]==8 and len(result['parents'])<=16
    assert result['caption']==dataset['manifest']['clips'][0]['caption']
    assert (dataset,stats,cache)==original
    assert value['options']['augmentation']['operation']==operation


@pytest.mark.parametrize('kind',['eval','missing','stale_stats','missing_text','empty_caption'])
def test_unusable_inputs_rejected(kind):
    dataset,stats,cache=inputs()
    clip='a'
    if kind=='eval':
        clip='b'
    elif kind=='missing':
        clip='absent'
    elif kind=='stale_stats':
        stats['dataset_sha256']='f'*64
    elif kind=='missing_text':
        cache['texts'][0]='changed'
    else:
        dataset['manifest']['clips'][0]['caption']=''
    with pytest.raises(ValueError):
        producer.produce_training_sample(dataset,stats,cache,clip,max_motion_length=8,max_joints=16)


def test_predecode_budget_and_cancellation(monkeypatch):
    dataset,stats,cache=inputs()
    def forbidden(*args,**kwargs):
        raise AssertionError('dataset decoded before workspace check')
    monkeypatch.setattr(producer,'validate_dataset',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        producer.produce_training_sample(dataset,stats,cache,'a',max_motion_length=8,max_joints=16,max_workspace_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        producer.produce_training_sample(dataset,stats,cache,'a',max_motion_length=8,max_joints=16,cancel=cancel)

# Import registers the pinned source fixture for this module.
from test_training_augmentation import reference  # noqa: E402,F401


@pytest.mark.parametrize('mode',['tpos','first_frame'])
@pytest.mark.parametrize('ground_rest',[True,False])
def test_bound_producer_matches_unchanged_source_metadata_and_getitem(reference,mode,ground_rest):  # noqa: F811
    import random
    from unimate_pack.training_text import text_views
    dataset,stats,cache=inputs()
    clip=dataset['manifest']['clips'][0]
    cond=decode_arrays(dataset['files'][clip['topology_id']])
    owner=reference['MotionDataset']()
    owner.max_freqs=8
    owner.ground_motion_height=ground_rest
    names=reference['model_joint_names'](cond,clip['object_type'])
    owner._joint_name_emb={name:text_views(cache,name)[1] for name in names}
    meta=owner._precompute_object_type_meta(clip['object_type'],cond)
    views=text_views(cache,clip['caption'])
    data=dict(motion=decode_arrays(dataset['files'][clip['features']])['features'],
              offsets=meta['offsets'],joint_names_emb=meta['joint_names_emb'],
              dataset_type=clip['dataset_type'],object_type=clip['object_type'],
              caption=clip['caption'],caption_tokens=views[0],caption_emb=views[1],
              **meta['condition'])
    stats_arrays=decode_arrays(stats['arrays'])
    row=stats['datasets'].index(clip['dataset_type'])
    owner.dataset_stats={clip['dataset_type']:{key:array[row] for key,array in stats_arrays.items()}}
    owner.train_motion_dict={'a':data}
    owner.train_name_list=['a']
    owner.debug_validate_aug=False
    owner.realign_feature=True
    owner.topology_condition_type=mode
    owner.max_motion_length=8
    owner.max_joints=16
    owner.feature_len=12
    for name in ('addition','removal','pooling','perturbation'):
        setattr(owner,'use_'+name+'_aug',False)
    # No augmentation's source wrapper still consumes random.choice([None]);
    # compare the explicitly independent crop stream by bypassing that no-op draw.
    owner._apply_augmentations=lambda data,mean,std: owner._extract_aug_dict(data,mean,std)
    state=random.getstate()
    try:
        random.seed(11)
        expected=owner[0]
    finally:
        random.setstate(state)
    value=producer.produce_training_sample(dataset,stats,cache,'a',mode=mode,
        max_motion_length=8,max_joints=16,crop_seed=11,ground_rest=ground_rest)
    actual=validate_training_sample(value)
    assert actual.keys()==expected.keys()
    for key,result in expected.items():
        if isinstance(result,np.ndarray):
            assert actual[key].dtype==result.dtype
            np.testing.assert_array_equal(actual[key],result)
        else:
            assert actual[key]==result
