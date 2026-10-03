"""Numeric training augmentations adapted from MIT UniMate.

Copyright (c) 2026 Linzhan Mou. License and source lineage are retained in
_vendor/LICENSE-UniMate and _vendor/SOURCES.json.

Local RNGs preserve legacy seeded streams. Motion/Quaternion dependencies are
replaced by reference-compared matrix FK; velocity channels follow source behavior.
"""

import math
import random

import numpy as np

from ._vendor import topology_utils as topology
from .rig_math import rotation_6d_matrices
from .statistics import UPSTREAM_REVISION

FIELDS = {'motion', 'parents', 'edge_indexs', 'joint_graph_dist', 'joint_relations',
          'joint_depths', 'spectral_feats', 'tpos', 'offsets', 'joint_names_emb', 'mean', 'std'}
OPERATIONS = ('addition', 'removal', 'pooling', 'perturbation')


def _cancel(callback):
    if callback is not None:
        callback()


def _copy_sample(sample):
    return {field: array.copy() for field, array in sample.items()}


def _validate_sample(sample, max_freqs, cancel):
    if type(sample) is not dict or set(sample) != FIELDS:
        raise ValueError('Training augmentation requires the exact source array fields')
    for array in sample.values():
        _cancel(cancel)
        if (type(array) is not np.ndarray or array.dtype.kind not in 'iuf'
                or not np.isfinite(array).all()):
            raise ValueError('Training augmentation requires finite plain numeric arrays')
    motion, parents = sample['motion'], sample['parents']
    if (motion.ndim != 3 or motion.shape[0] < 1 or motion.shape[2] != 12
            or motion.dtype.kind != 'f' or motion.dtype.itemsize not in (2, 4, 8)):
        raise ValueError('Training motion requires floating (T,J,12) features')
    joints = motion.shape[1]
    if (joints < 1 or parents.shape != (joints,) or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or np.any(parents[1:] < 0)
            or np.any(parents[1:] >= np.arange(1, joints))):
        raise ValueError('Training augmentation requires ordered connected parents')
    for field, shape in (('tpos', (joints, 3)), ('offsets', (joints, 3)),
                         ('mean', (joints, 12)), ('std', (joints, 12)),
                         ('spectral_feats', (joints, max_freqs))):
        if sample[field].shape != shape or sample[field].dtype.kind != 'f':
            raise ValueError('Invalid per-joint training array shape or dtype')
    embeddings = sample['joint_names_emb']
    if (embeddings.ndim != 2 or embeddings.shape[0] != joints
            or embeddings.shape[1] < 1 or embeddings.dtype.kind != 'f'
            or np.any(sample['std'] <= 0)):
        raise ValueError('Invalid joint embeddings or normalization standard deviations')
    if (sample['edge_indexs'].dtype.kind not in 'iu'
            or not np.array_equal(sample['edge_indexs'], topology.compute_edge_indexs(parents))
            or sample['joint_depths'].dtype.kind not in 'iu'
            or not np.array_equal(sample['joint_depths'], topology.compute_joint_depths(parents))):
        raise ValueError('Training topology edges/depths differ from parents')
    for field in ('joint_graph_dist', 'joint_relations'):
        if sample[field].shape != (joints, joints) or sample[field].dtype.kind not in 'iu':
            raise ValueError('Invalid training topology matrix')
    path_cap = int(sample['joint_graph_dist'].max())
    if not 0 <= path_cap <= 32766 or (joints > 1 and path_cap < 1):
        raise ValueError('Invalid training graph distance cap')
    _cancel(cancel)
    relations, distances = topology.compute_edge_relations_and_distances(parents.tolist(), max_path_len=path_cap)
    if (not np.array_equal(sample['joint_relations'], relations)
            or not np.array_equal(sample['joint_graph_dist'], distances)):
        raise ValueError('Training topology matrices differ from parents')


def _recompute_positions(aug, cancel):
    motion, parents, offsets = aug['motion'], aug['parents'], aug['offsets']
    frames, joints = motion.shape[:2]
    raw = motion[..., 3:9]
    rotation_6d_matrices(raw)  # Explicitly reject degenerate axes before divisions.
    first = raw[..., :3] / np.linalg.norm(raw[..., :3], axis=-1, keepdims=True)
    third = np.cross(first, raw[..., 3:])
    third /= np.linalg.norm(third, axis=-1, keepdims=True)
    second = np.cross(third, first)
    hml = np.stack([first, second, third], axis=-1)
    rotations = np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy()
    for j, p in enumerate(parents[1:], 1):
        _cancel(cancel)
        rotations[:, p] = hml[:, j]
    # Source FK passes float64 BVH matrices through signed-diagonal quaternions.
    rotations = _quaternion_roundtrip(rotations)
    positions = np.zeros((frames, joints, 3))
    positions[:, 0, 1] = motion[:, 0, 1]
    global_rotations = rotations.copy()
    for j, p in enumerate(parents[1:], 1):
        _cancel(cancel)
        positions[:, j] = positions[:, p] + np.einsum('tij,j->ti', global_rotations[:, p], offsets[j])
        global_rotations[:, j] = global_rotations[:, p] @ rotations[:, j]
    # Facing maps world coordinates into the RIFKE frame; root is pinned in XZ.
    facing = _quaternion_roundtrip(hml[:, 0], quaternion_vector=True)
    aug['motion'][:, 1:, :3] = np.einsum('tij,tkj->tki', facing, positions[:, 1:])


def _quaternion_roundtrip(matrices, *, quaternion_vector=False):
    """Standard signed-diagonal quaternion equations in source dtype/order.

    No Motion implementation is imported or distributed. Quaternion-vector
    conjugation preserves its squared norm; FK's rotation matrix assumes one.
    Retain that distinction for reference parity on low-precision inputs.
    """
    a, b, c = (matrices[..., i, i] for i in range(3))
    squares = [(a + b + c + 1) / 4, (a - b - c + 1) / 4,
               (-a + b - c + 1) / 4, (-a - b + c + 1) / 4]
    components = [np.sqrt(np.maximum(value, 0)) for value in squares]
    largest = [np.logical_and.reduce([value >= other for other in components]) for value in components]
    skew = [matrices[..., 2, 1] - matrices[..., 1, 2],
            matrices[..., 0, 2] - matrices[..., 2, 0],
            matrices[..., 1, 0] - matrices[..., 0, 1]]
    symmetric = [matrices[..., 1, 0] + matrices[..., 0, 1],
                 matrices[..., 0, 2] + matrices[..., 2, 0],
                 matrices[..., 2, 1] + matrices[..., 1, 2]]
    sign_rules = [(None, *skew), (skew[0], None, symmetric[0], symmetric[1]),
                  (skew[1], symmetric[0], None, symmetric[2]),
                  (skew[2], symmetric[1], symmetric[2], None)]
    for dominant, rules in enumerate(sign_rules):
        for index, rule in enumerate(rules):
            if rule is not None:
                components[index][largest[dominant]] *= np.sign(rule[largest[dominant]])
    w, x, y, z = (component.astype(np.float64) for component in components)
    diagonal = w*w + x*x + y*y + z*z if quaternion_vector else np.ones_like(w)
    result = np.empty((*matrices.shape[:-2], 3, 3))
    result[..., 0, 0] = diagonal - 2*(y*y + z*z)
    result[..., 1, 1] = diagonal - 2*(x*x + z*z)
    result[..., 2, 2] = diagonal - 2*(x*x + y*y)
    result[..., 0, 1], result[..., 1, 0] = 2*(x*y - w*z), 2*(x*y + w*z)
    result[..., 0, 2], result[..., 2, 0] = 2*(x*z + w*y), 2*(x*z - w*y)
    result[..., 1, 2], result[..., 2, 1] = 2*(y*z - w*x), 2*(y*z + w*x)
    return result


def _recompute_tpos(aug, cancel):
    positions = np.zeros((len(aug['parents']), 3))
    for j, p in enumerate(aug['parents']):
        _cancel(cancel)
        positions[j] = aug['offsets'][j] if p == -1 else positions[p] + aug['offsets'][j]
    aug['tpos'][:] = positions


def _recompute_all(aug, max_freqs, cancel, max_path_len=5):
    _recompute_positions(aug, cancel)
    _recompute_tpos(aug, cancel)
    _cancel(cancel)
    parents = aug['parents']
    aug['edge_indexs'] = topology.compute_edge_indexs(parents)
    aug['joint_relations'], aug['joint_graph_dist'] = topology.compute_edge_relations_and_distances(
        parents.tolist(), max_path_len=max_path_len)
    aug['joint_depths'] = topology.compute_joint_depths(parents)
    _cancel(cancel)
    aug['spectral_feats'], _ = topology.compute_laplacian_eigenvectors(parents, max_freqs=max_freqs)
    _cancel(cancel)


def _children(parents):
    children = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0 and child != parent:
            children[parent].append(child)
    return children


def _weighted(candidates, offsets, count, rng):
    lengths = np.array([np.linalg.norm(offsets[j]) for j in candidates])
    weights = np.maximum(lengths, 1e-8) ** (-.5)
    weights /= weights.sum()
    return rng.choice(candidates, size=min(count, len(candidates)), replace=False, p=weights).tolist()


def _delete(aug, joint):
    aug['motion'] = np.delete(aug['motion'], joint, axis=1)
    aug['parents'] = np.delete(aug['parents'], joint, axis=0)
    aug['parents'][aug['parents'] > joint] -= 1
    for field in ('tpos', 'offsets', 'joint_names_emb', 'joint_depths', 'mean', 'std'):
        aug[field] = np.delete(aug[field], joint, axis=0)


def _ellipsoid(offset, rng, sigma, lateral_ratio, cancel):
    length = np.linalg.norm(offset)
    if length < 1e-8:
        return np.zeros(3)
    while True:
        _cancel(cancel)
        direction = rng.uniform(-1, 1, size=3)
        norm = np.linalg.norm(direction)
        if 0 < norm <= 1:
            direction /= norm
            break
    while True:
        _cancel(cancel)
        radius = abs(rng.normal(0, sigma))
        if radius <= 1:
            break
    scaled = radius * direction * np.array([length, length * lateral_ratio, length * lateral_ratio])
    e1 = offset / length
    reference = np.array([0., 1., 0.])
    if abs(np.dot(e1, reference)) > .9:
        reference = np.array([1., 0., 0.])
    e2 = np.cross(e1, reference)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3]) @ scaled


def _addition(aug, py_rng, np_rng, linear, sigma, lateral_ratio, max_freqs, max_path_len, cancel, neutral):
    motion, parents, offsets = aug['motion'], aug['parents'], aug['offsets']
    if len(parents) < 2:
        return
    joint = py_rng.choice(list(range(1, len(parents))))
    parent = int(parents[joint])
    fraction = py_rng.uniform(.3, .7)
    new_offset = offsets[joint] * fraction
    if not linear:
        new_offset = new_offset + _ellipsoid(offsets[joint], np_rng, sigma, lateral_ratio, cancel)
    remainder = offsets[joint] * (1 - fraction) if linear else offsets[joint] - new_offset
    new_motion = np.zeros((len(motion), 1, 12), dtype=motion.dtype)
    new_motion[:, 0, 3:9] = motion[:, joint, 3:9]
    aug['motion'] = np.concatenate([motion[:, :joint], new_motion, motion[:, joint:]], axis=1)
    if neutral:
        # The old child's shifted HML slot now stores the inserted joint's
        # local rotation. Identity avoids duplicating the old parent's rotation.
        aug['motion'][:, joint + 1, 3:9] = [1, 0, 0, 0, 1, 0]
    aug['offsets'] = np.concatenate([offsets[:joint], new_offset[None], offsets[joint:]], axis=0)
    aug['offsets'][joint + 1] = remainder
    tpos = aug['tpos']
    aug['tpos'] = np.concatenate([tpos[:joint], (tpos[parent] + new_offset)[None], tpos[joint:]], axis=0)
    embeddings = aug['joint_names_emb']
    new_embedding = (embeddings[parent] + embeddings[joint]) / 2
    aug['joint_names_emb'] = np.concatenate([embeddings[:joint], new_embedding[None], embeddings[joint:]], axis=0)
    mean, std = aug['mean'], aug['std']
    aug['mean'] = np.concatenate([mean[:joint], mean[joint:joint + 1].copy(), mean[joint:]], axis=0)
    aug['std'] = np.concatenate([std[:joint], std[joint:joint + 1], std[joint:]], axis=0)
    new_parents = parents.copy()
    new_parents[new_parents >= joint] += 1
    aug['parents'] = np.concatenate([new_parents[:joint], [parent], new_parents[joint:]])
    aug['parents'][joint + 1] = joint
    _recompute_all(aug, max_freqs, cancel, max_path_len=max_path_len)


def _removal(aug, rate, rng, max_freqs, cancel):
    children = _children(aug['parents'])
    candidates = [j for j in range(1, len(children)) if not children[j]]
    count = min(round(len(candidates) * rate), 3)
    if not candidates or not count:
        return
    for joint in sorted(_weighted(candidates, aug['offsets'], count, rng), reverse=True):
        _cancel(cancel)
        if not _children(aug['parents'])[joint]:
            _delete(aug, joint)
    _recompute_all(aug, max_freqs, cancel)


def _pooling(aug, rate, rng, max_freqs, cancel):
    children = _children(aug['parents'])
    candidates = [j for j in range(1, len(children)) if len(children[j]) == 1]
    if not candidates:
        return
    count = max(1, int(len(candidates) * rate))
    for joint in sorted(_weighted(candidates, aug['offsets'], count, rng), reverse=True):
        _cancel(cancel)
        children = _children(aug['parents'])
        if len(children[joint]) != 1:
            continue
        child, parent = children[joint][0], int(aug['parents'][joint])
        aug['motion'][:, child, 3:9] = aug['motion'][:, joint, 3:9]
        aug['offsets'][child] = aug['offsets'][joint] + aug['offsets'][child]
        aug['parents'][child] = parent
        _delete(aug, joint)
    _recompute_all(aug, max_freqs, cancel)


def _scalar(value):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Augmentation options require finite numbers')
        return float(value)
    except OverflowError as error:
        raise ValueError('Augmentation option exceeds numeric range') from error


def augment_sample(sample, operation, seed, *, enabled=(), removal_rate=None, pool_rate=None,
                   max_scale=.1, sigma=.5, lateral_ratio=.5, max_freqs=8, max_path_len=5,
                   max_workspace_bytes=512 * 1024 * 1024, cancel=None, addition_policy='released'):
    _cancel(cancel)
    if operation not in ('none', 'addition_linear', 'random', *OPERATIONS):
        raise ValueError('Unknown training augmentation operation')
    if addition_policy not in ('released', 'neutral_fk'):
        raise ValueError('Unknown training addition policy')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Augmentation seed requires an unsigned 32-bit integer')
    if type(enabled) not in (tuple, list) or any(op not in OPERATIONS for op in enabled):
        raise ValueError('Invalid enabled training augmentations')
    for value, maximum in ((max_freqs, 4096), (max_path_len, 32766)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid topology frequency/path bound')
    sigma, lateral_ratio, max_scale = map(_scalar, (sigma, lateral_ratio, max_scale))
    if sigma < 0 or lateral_ratio < 0 or not 0 <= max_scale <= 1:
        raise ValueError('Invalid ellipsoid or perturbation options')
    for rate in (removal_rate, pool_rate):
        if rate is not None and not 0 <= _scalar(rate) <= 1:
            raise ValueError('Augmentation rate must be in [0,1]')
    if (type(sample) is not dict or type(max_workspace_bytes) is not int
            or max_workspace_bytes < 1 or type(sample.get('motion')) is not np.ndarray
            or sample['motion'].ndim != 3):
        raise ValueError('Invalid training sample or workspace budget')
    frames, joints = sample['motion'].shape[:2]
    headroom = joints + 1
    estimate = (8 * sum(a.nbytes for a in sample.values() if type(a) is np.ndarray)
                + 12 * 8 * headroom**2 + 16 * 8 * frames * headroom * 3)
    if estimate > max_workspace_bytes:
        raise ValueError('Training augmentation exceeds numeric workspace budget')
    _validate_sample(sample, max_freqs, cancel)
    aug = _copy_sample(sample)
    py_rng, np_rng = random.Random(seed), np.random.RandomState(seed)
    chosen = operation
    if operation == 'random':
        candidates = ['none'] + [op for op in OPERATIONS if op in enabled]
        chosen = py_rng.choice(candidates) if len(candidates) > 1 else 'none'
        # The released dataset wrappers fix these operation parameters.
        sigma, lateral_ratio, max_scale, max_path_len = .5, .5, .1, 5
    rate = None
    if chosen in ('addition', 'addition_linear'):
        _addition(aug, py_rng, np_rng, chosen == 'addition_linear', sigma, lateral_ratio,
                  max_freqs, max_path_len, cancel, addition_policy == 'neutral_fk')
    elif chosen == 'removal':
        rate = py_rng.uniform(.05, .15) if removal_rate is None or operation == 'random' else removal_rate
        _removal(aug, rate, np_rng, max_freqs, cancel)
    elif chosen == 'pooling':
        rate = py_rng.uniform(.1, .3) if pool_rate is None or operation == 'random' else pool_rate
        _pooling(aug, rate, np_rng, max_freqs, cancel)
    elif chosen == 'perturbation':
        scales = np_rng.uniform(1 - max_scale, 1 + max_scale, size=len(aug['parents']))
        for joint in range(1, len(aug['parents'])):
            _cancel(cancel)
            aug['offsets'][joint] *= scales[joint]
        _recompute_positions(aug, cancel)
        _recompute_tpos(aug, cancel)
    _validate_sample(aug, max_freqs, cancel)
    _cancel(cancel)
    return aug, dict(upstream_revision=UPSTREAM_REVISION, operation=chosen, requested_operation=operation,
                     addition_policy=addition_policy,
                     seed=seed, rate=rate, input_joints=joints, output_joints=len(aug['parents']),
                     parameters=dict(sigma=sigma, lateral_ratio=lateral_ratio, max_scale=max_scale,
                                     max_freqs=max_freqs,
                                     max_path_len=5 if chosen in ('removal', 'pooling') else max_path_len),
                     workspace_estimate_bytes=estimate, velocity_policy='released_preserve_existing_channels')
