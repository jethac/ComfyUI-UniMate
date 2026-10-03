"""Validated numeric sample transforms adapted from MIT UniMate.

Copyright (c) 2026 Linzhan Mou. See _vendor/LICENSE-UniMate and SOURCES.json.
"""
import random

import numpy as np


def _check(cancel):
    if cancel is not None:
        cancel()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _array(value, dimensions, empty_time=False):
    if (type(value) is not np.ndarray or value.ndim not in dimensions
            or value.dtype.kind != 'f' or not np.isfinite(value).all()
            or any(size == 0 for size in (value.shape[1:] if empty_time and value.ndim == 3 else value.shape))):
        raise ValueError('Expected a nonempty finite plain floating array')


def apply_cropping(motion, mode, max_motion_length, *, start_idx=None, seed=0, cancel=None):
    _check(cancel)
    _array(motion, (3,))
    _integer(max_motion_length, 'max_motion_length', 1)
    _integer(seed, 'seed')
    if mode not in ('tpos', 'first_frame'):
        raise ValueError('Invalid topology condition type')
    target = max_motion_length + (mode == 'first_frame')
    if start_idx is not None:
        _integer(start_idx, 'start_idx')
        if start_idx >= len(motion):
            raise ValueError('start_idx exceeds clip length')
    else:
        start_idx = (random.Random(seed).randint(0, len(motion)-target)
                     if mode == 'tpos' and len(motion) > target else 0)
    result = motion[start_idx:start_idx+target]
    _check(cancel)
    return result, start_idx


def extract_conditions(motion, tpos, mode, *, cancel=None):
    _check(cancel)
    _array(motion, (3,))
    _array(tpos, (2,))
    if tpos.shape != motion.shape[1:]:
        raise ValueError('Rest condition shape differs from motion')
    if mode == 'first_frame':
        return {'tpos_first_frame': motion[0].copy(), 'motion': motion[1:].copy()}
    if mode == 'tpos':
        return {'tpos_first_frame': tpos.copy(), 'motion': motion.copy()}
    raise ValueError('Invalid topology condition type')


def apply_normalization(motion, mean, std, *, cancel=None):
    _check(cancel)
    _array(motion, (2, 3), empty_time=True)
    _array(mean, (2,))
    _array(std, (2,))
    if mean.shape != std.shape or mean.shape != motion.shape[-2:] or np.any(std <= 0):
        raise ValueError('Invalid normalization statistics')
    result = ((motion - mean[None, :]) / std[None, :]
              if motion.ndim == 3 else (motion - mean) / std)
    result = np.nan_to_num(result)
    _check(cancel)
    return result


def apply_padding(motion, max_motion_length, *, max_workspace_bytes=512*1024*1024, cancel=None):
    _check(cancel)
    _array(motion, (3,), empty_time=True)
    _integer(max_motion_length, 'max_motion_length', 1)
    _integer(max_workspace_bytes, 'max_workspace_bytes', 1)
    length = len(motion)
    if length > max_motion_length:
        raise ValueError('Crop motion before padding')
    estimate = motion.nbytes + 16*max_motion_length*int(np.prod(motion.shape[1:]))
    if estimate > max_workspace_bytes:
        raise ValueError('Padding exceeds workspace budget')
    if length < max_motion_length:
        pad = np.zeros((max_motion_length-length, *motion.shape[1:]))
        motion = np.concatenate([motion, pad], axis=0)
    _check(cancel)
    return motion, length


def build_parent_features(condition, parents, *, cancel=None):
    _check(cancel)
    _array(condition, (2,))
    if (type(parents) is not np.ndarray or parents.dtype.kind not in 'iu'
            or parents.shape != (len(condition),) or parents[0] != -1
            or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1))):
        raise ValueError('Invalid ordered rooted parents')
    result = condition.copy()
    for joint, parent in enumerate(parents):
        _check(cancel)
        if parent != -1:
            result[joint] = condition[parent]
    return {'tpos_first_frame_parents': result}


def realign_clip(motion, parents, *, cancel=None, max_workspace_bytes=512*1024*1024):
    """Rebase facing at cropped frame zero, preserving source arithmetic."""
    from .rig_math import rotation_6d_matrices
    from .training_augmentation import _matrix_quaternions

    _check(cancel)
    _array(motion, (3,))
    if motion.shape[-1] != 12:
        raise ValueError('Expected twelve motion features')
    _integer(max_workspace_bytes, 'max_workspace_bytes', 1)
    if 16*motion.nbytes > max_workspace_bytes:
        raise ValueError('Realignment exceeds workspace budget')
    build_parent_features(motion[0], parents, cancel=cancel)
    rotation_6d_matrices(motion[..., 3:9])

    def decode(raw):
        first = raw[..., :3] / np.linalg.norm(raw[..., :3], axis=-1, keepdims=True)
        third = np.cross(first, raw[..., 3:])
        third /= np.linalg.norm(third, axis=-1, keepdims=True)
        second = np.cross(third, first)
        return _matrix_quaternions(np.stack([first, second, third], axis=-1))

    def compose(left, right):
        w,x,y,z = np.moveaxis(left,-1,0)
        a,b,c,d = np.moveaxis(right,-1,0)
        return np.stack([a*w-b*x-c*y-d*z,
                         a*x+b*w-c*z+d*y,
                         a*y+b*z+c*w-d*x,
                         a*z-b*y+c*x+d*w],axis=-1)

    def encode(q):
        w,x,y,z = np.moveaxis(q,-1,0)
        scale = 2.0/(q*q).sum(-1)
        return np.stack([1-scale*(y*y+z*z),scale*(x*y+z*w),scale*(x*z-y*w),
                         scale*(x*y-z*w),1-scale*(x*x+z*z),scale*(y*z+x*w)],axis=-1)

    result=motion.copy()
    initial=decode(motion[:1,0,3:9])
    result[:,0,3:9]=encode(compose(decode(motion[:,0,3:9]),initial*np.array([1,-1,-1,-1])))
    for joint,parent in enumerate(parents):
        _check(cancel)
        if parent == 0:
            result[:,joint,3:9]=encode(compose(initial,decode(motion[:,joint,3:9])))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite realignment result')
    _check(cancel)
    return result
