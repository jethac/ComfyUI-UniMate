"""Canonical UniMate skeleton recovery without Blender or the Motion library."""

import numpy as np
import hashlib

from .contracts import (MAX_ARRAY_BYTES, validate_rig, validate_motion, validate_skeleton,
                        decode_arrays, encode_arrays)
from .rig_math import decode_features, rotation_6d_matrices


def recover_positions(features, parents, offsets, method, *, root_origin=(0, 0, 0),
                      check_cancel=lambda: None):
    check_cancel()
    if method not in ('fk', 'ric'):
        raise ValueError('Skeleton recovery mode must be fk or ric')
    features = np.asarray(features)
    parents = np.asarray(parents)
    offsets = np.asarray(offsets)
    if (features.ndim != 3 or features.shape[0] < 1 or features.shape[2] != 12
            or features.dtype.kind != 'f' or features.nbytes > MAX_ARRAY_BYTES
            or not np.isfinite(features).all()):
        raise ValueError('Skeleton recovery requires finite (T,J,12) features')
    joints = features.shape[1]
    if (joints < 1 or parents.shape != (joints,) or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or np.any(parents[1:] < 0)
            or np.any(parents[1:] >= np.arange(1, joints))
            or offsets.shape != (joints, 3) or not np.isfinite(offsets).all()):
        raise ValueError('Skeleton recovery requires ordered parents and finite offsets')
    origin = np.asarray(root_origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all() or origin[1] != 0:
        raise ValueError('Canonical root origin must be finite XZ with zero Y')
    facing = rotation_6d_matrices(features[:, 0, 3:9])
    velocity = np.zeros((len(features), 3))
    velocity[1:, 0] = features[:-1, 0, 9]
    velocity[1:, 2] = features[:-1, 0, 11]
    root = np.cumsum(np.einsum('tji,tj->ti', facing, velocity), axis=0)
    root[:, 1] = features[:, 0, 1]
    root += origin
    positions = np.empty((*features.shape[:2], 3), dtype=np.float64)
    positions[:, 0] = root
    if method == 'ric':
        positions[:, 1:] = np.einsum('tji,tkj->tki', facing, features[:, 1:, :3])
        positions[:, 1:, 0] += root[:, 0, None]
        positions[:, 1:, 2] += root[:, 2, None]
    else:
        rotations, _ = decode_features(features, parents)
        global_rotations = np.empty_like(rotations)
        global_rotations[:, 0] = rotations[:, 0]
        for joint, parent in enumerate(parents[1:], 1):
            check_cancel()
            positions[:, joint] = positions[:, parent] + global_rotations[:, parent] @ offsets[joint]
            global_rotations[:, joint] = global_rotations[:, parent] @ rotations[:, joint]
    check_cancel()
    if not np.isfinite(positions).all():
        raise ValueError('Skeleton recovery produced nonfinite positions')
    return positions


def recover_skeleton(rig, motion, method, *, check_cancel=lambda: None):
    check_cancel()
    validate_rig(rig)
    validate_motion(motion, rig['rig_id'])
    conditioning = decode_arrays(rig['conditioning'])
    features = decode_arrays(motion['features'])['features']
    if features.shape[1] != len(conditioning['parents']):
        raise ValueError('Motion joint count does not match prepared skeleton')
    positions = recover_positions(features, conditioning['parents'], conditioning['tpos_offsets'],
        method, root_origin=motion['metadata'].get('canonical_root_origin', [0, 0, 0]),
        check_cancel=check_cancel)
    arrays = encode_arrays(positions=positions.astype(np.float32),
                           parents=conditioning['parents'], joint_names=conditioning['joint_names'])
    value = dict(schema='unimate.skeleton.v1', rig_id=rig['rig_id'], arrays=arrays,
                 sha256=hashlib.sha256(arrays).hexdigest(), fps=30, method=method)
    validate_skeleton(value)
    check_cancel()
    return value
