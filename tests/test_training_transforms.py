"""Exact released sample-transform comparisons."""
import hashlib
import importlib.util
import os
from pathlib import Path
import random

import numpy as np
import pytest

from unimate_pack import training_transforms as adapter


@pytest.fixture(scope='module')
def reference():
    root = os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned source required')
    path = Path(root) / 'unimate/dataset/transforms.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '0c16cd112fb5df83778d245df06dbeaa040546fda56611421286340ae0fe231a'
    spec = importlib.util.spec_from_file_location('reference_transforms', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['first_frame', 'tpos'])
@pytest.mark.parametrize('frames,start', [(3,None),(12,None),(12,9),(12,0)])
@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
def test_crop_exact(reference, mode, frames, start, dtype):
    motion = np.arange(frames*4*12, dtype=dtype).reshape(frames,4,12)
    state = random.getstate()
    try:
        random.seed(17)
        expected, index = reference.apply_cropping(motion, mode, 5, start)
    finally:
        random.setstate(state)
    actual, actual_index = adapter.apply_cropping(motion, mode, 5, start_idx=start, seed=17)
    assert actual_index == index
    np.testing.assert_array_equal(actual, expected)
    assert random.getstate() == state


@pytest.mark.parametrize('mode', ['first_frame','tpos'])
@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
def test_conditions_normalization_padding_parent_features(reference,mode,dtype):
    motion = np.arange(3*4*12,dtype=dtype).reshape(3,4,12)/10
    rest = motion[0].copy()
    mean, std = rest/3, np.ones_like(rest)*2
    expected = reference.extract_conditions(motion, rest, mode)
    actual = adapter.extract_conditions(motion, rest, mode)
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key])
        assert not np.shares_memory(actual[key],motion)
        norm = adapter.apply_normalization(actual[key],mean,std)
        np.testing.assert_array_equal(norm,reference.apply_normalization(expected[key],mean,std))
    normalized = adapter.apply_normalization(actual['motion'],mean,std)
    padded,length = adapter.apply_padding(normalized,5)
    exp_padded,exp_length = reference.apply_padding(normalized,5)
    assert length == exp_length and padded.dtype == exp_padded.dtype
    np.testing.assert_array_equal(padded,exp_padded)
    parents=np.array([-1,0,0,2])
    actual_parent=adapter.build_parent_features(rest,parents)
    np.testing.assert_array_equal(actual_parent['tpos_first_frame_parents'], reference.build_parent_features(rest,parents)['tpos_first_frame_parents'])


@pytest.mark.parametrize('call', [
    lambda: adapter.apply_cropping(np.zeros((2,3,12)),'bad',4),
    lambda: adapter.apply_cropping(np.zeros((2,3,12)),'tpos',0),
    lambda: adapter.apply_cropping(np.zeros((2,3,12)),'tpos',4,start_idx=True),
    lambda: adapter.apply_normalization(np.zeros((2,3,12)),np.zeros((3,12)),np.zeros((3,12))),
    lambda: adapter.apply_padding(np.zeros((2,3,12)),1),
    lambda: adapter.build_parent_features(np.zeros((3,12)),np.array([-1,2,0])),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_budget_and_cancellation():
    motion=np.zeros((2,3,12))
    with pytest.raises(ValueError,match='workspace'):
        adapter.apply_padding(motion,100,max_workspace_bytes=10)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        adapter.apply_cropping(motion,'tpos',2,cancel=cancel)


def test_single_first_frame_has_empty_motion_then_padding(reference):
    motion=np.zeros((1,3,12))
    cond=adapter.extract_conditions(motion,motion[0],'first_frame')
    norm=adapter.apply_normalization(cond['motion'],np.zeros((3,12)),np.ones((3,12)))
    actual,length=adapter.apply_padding(norm,4)
    expected,expected_length=reference.apply_padding(norm,4)
    assert length == expected_length == 0
    np.testing.assert_array_equal(actual,expected)
