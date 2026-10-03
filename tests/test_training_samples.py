"""Sample assembly against unchanged MotionDataset.__getitem__."""
import copy
import random

import numpy as np
import pytest

from test_training_augmentation import reference, sample, reference_call  # noqa: F401
from unimate_pack import training_samples


@pytest.mark.parametrize('mode', ['tpos','first_frame'])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('operation',['none','addition','removal','pooling','perturbation'])
@pytest.mark.parametrize('length',[4,12])
def test_assembly_matches_released_getitem(reference,mode,dtype,operation,length):  # noqa: F811
    aug=reference_call(reference,sample(dtype=dtype),operation,17)
    tokens=np.arange(21,dtype=np.float32).reshape(3,7)/10
    pooled=tokens.mean(0)
    owner=reference['MotionDataset']()
    owner.train_name_list=['clip']
    owner.train_motion_dict={'clip': dict(dataset_type='objaverse',object_type='fixture',
                                        caption='walk',caption_emb=pooled,caption_tokens=tokens)}
    owner._get_normalization_stats=lambda data,ds: (aug['mean'],aug['std'])
    owner._apply_augmentations=lambda data,mean,std: copy.deepcopy(aug)
    owner.debug_validate_aug=False
    owner.realign_feature=True
    owner.topology_condition_type=mode
    owner.max_motion_length=length
    owner.max_joints=16
    owner.feature_len=12
    state=random.getstate()
    try:
        random.seed(11)
        expected=owner[0]
    finally:
        random.setstate(state)
    original=copy.deepcopy(aug)
    actual=training_samples.assemble_sample(aug,object_type='fixture',caption='walk',
        caption_tokens=tokens,caption_emb=pooled,mode=mode,max_motion_length=length,
        max_joints=16,crop_seed=11)
    assert actual.keys()==expected.keys()
    for key,value in expected.items():
        if isinstance(value,np.ndarray):
            assert actual[key].dtype==value.dtype
            np.testing.assert_array_equal(actual[key],value)
        else:
            assert actual[key]==value
    for key,value in aug.items():
        np.testing.assert_array_equal(value,original[key])
        for output in actual.values():
            if isinstance(output,np.ndarray):
                assert not np.shares_memory(output,value)
    assert random.getstate()==state


@pytest.mark.parametrize('options',[
    dict(caption=''),dict(max_joints=2),dict(max_motion_length=True),
    dict(realign_feature=1),dict(caption_emb=np.zeros(8)),
    dict(caption_tokens=np.full((3,7),np.nan)),dict(max_workspace_bytes=10),
])
def test_assembly_rejects_invalid_inputs(options):
    args=dict(object_type='fixture',caption='walk',caption_tokens=np.ones((3,7)),
              caption_emb=np.ones(7),mode='tpos',max_motion_length=4,max_joints=16)
    args.update(options)
    with pytest.raises(ValueError):
        training_samples.assemble_sample(sample(),**args)


def test_assembly_cancellation():
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        training_samples.assemble_sample(sample(),object_type='fixture',caption='walk',
            caption_tokens=np.ones((3,7)),caption_emb=np.ones(7),mode='tpos',
            max_motion_length=4,max_joints=16,cancel=cancel)


def test_sample_budget_rejects_before_dense_topology_validation(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('dense validation ran before budget check')
    monkeypatch.setattr(training_samples,'_validate_sample',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        training_samples.assemble_sample(sample(),object_type='fixture',caption='walk',
            caption_tokens=np.ones((3,7)),caption_emb=np.ones(7),mode='tpos',
            max_motion_length=4,max_joints=16,max_workspace_bytes=1)
