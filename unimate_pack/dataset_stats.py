"""Released root/local normalization modes on unnormalized numeric clips."""

from collections import defaultdict

import numpy as np

from .contracts import MAX_ARRAY_BYTES

MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
_FIELDS = ('mean_root', 'std_root', 'mean_local', 'std_local')


def _label(value):
    if (type(value) is not str or not value or len(value) > 255
            or value in ('.', '..') or any(c in value for c in '/\\:\0')):
        raise ValueError('Dataset and object labels must be portable names')


def _check(clips, options, cancel):
    if type(clips) is not list or not clips:
        raise ValueError('Statistics require a nonempty list of training clips')
    if any(type(option) is not bool for option in options):
        raise ValueError('Statistics options must be Boolean')
    total, largest = 0, 0
    datasets, groups = set(), set()
    per_dataset, balanced, _ = options
    for record in clips:
        if cancel is not None:
            cancel()
        if type(record) is not dict or set(record) != {'dataset_type', 'object_type', 'features'}:
            raise ValueError('Invalid training clip fields')
        _label(record['dataset_type'])
        _label(record['object_type'])
        datasets.add(record['dataset_type'])
        if balanced:
            groups.add((record['dataset_type'] if per_dataset else None, record['object_type']))
        features = record['features']
        if (type(features) is not np.ndarray or features.dtype.kind != 'f'
                or features.dtype.itemsize > 8
                or features.ndim != 3 or features.shape[0] < 1
                or features.shape[1] < 2 or features.shape[2] != 12):
            raise ValueError('Training features must be floating (T,J,12), with T>0 and J>=2')
        total += features.nbytes
        largest = max(largest, features.size * 8)
        # Includes retained group moments, stacked reductions and output copies.
        estimate = total + 3 * largest + (16 * len(groups) + 4 * len(datasets)) * 12 * 8
        if total > MAX_ARRAY_BYTES or estimate > MAX_WORKSPACE_BYTES:
            raise ValueError('Statistics workspace estimate exceeds allocation limit')
        if not np.isfinite(features).all():
            raise ValueError('Training features must be finite')


def _moments(records, cancel):
    sum_root, sq_root = np.zeros(12), np.zeros(12)
    sum_local, sq_local = np.zeros(12), np.zeros(12)
    n_root, n_local = 0, 0
    with np.errstate(over='ignore', invalid='ignore'):
        for record in records:
            if cancel is not None:
                cancel()
            # The released reductions retain input precision before adding
            # into float64 accumulators. Upcasting first changes float32 runs.
            features = record['features']
            root = features[:, 0, :]
            local = features[:, 1:, :].reshape(-1, 12)
            sum_root += root.sum(axis=0)
            sq_root += (root ** 2).sum(axis=0)
            sum_local += local.sum(axis=0)
            sq_local += (local ** 2).sum(axis=0)
            n_root += features.shape[0]
            n_local += local.shape[0]
        mean_root = sum_root / n_root
        mean_local = sum_local / n_local
        var_root = np.maximum(sq_root / n_root - mean_root ** 2, 0.0)
        var_local = np.maximum(sq_local / n_local - mean_local ** 2, 0.0)
    values = mean_root, var_root, mean_local, var_local
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError('Training moments overflow float64')
    return values


def _pool(records, balanced, tie_std, cancel):
    if balanced:
        groups = defaultdict(list)
        for record in records:
            groups[record['object_type']].append(record)
        moments = [_moments(group, cancel) for group in groups.values()]
        roots = [item[0] for item in moments]
        locals_ = [item[2] for item in moments]
        mean_root = np.mean(roots, axis=0)
        mean_local = np.mean(locals_, axis=0)
        var_root = np.mean([item[1] for item in moments], axis=0) + np.var(roots, axis=0)
        var_local = np.mean([item[3] for item in moments], axis=0) + np.var(locals_, axis=0)
    else:
        mean_root, var_root, mean_local, var_local = _moments(records, cancel)
    std_root = np.maximum(np.sqrt(np.maximum(var_root, 0.0)), 1e-8)
    std_local = np.maximum(np.sqrt(np.maximum(var_local, 0.0)), 1e-8)
    if tie_std:
        for std in (std_root, std_local):
            for start, end in ((0, 3), (3, 9), (9, 12)):
                std[start:end] = std[start:end].mean()
    values = mean_root, std_root, mean_local, std_local
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError('Training statistics overflow float64')
    return dict(zip(_FIELDS, values))


def compute_statistics(clips, *, per_dataset=True, balanced=False, tie_std=False, cancel=None):
    """Compute pinned UniMate statistics; only supply clips in the training split.

    Each record contains dataset_type, object_type and unnormalized features.
    The workspace limit is a conservative estimate, not a measured peak.
    """
    _check(clips, (per_dataset, balanced, tie_std), cancel)
    datasets = defaultdict(list)
    for record in clips:
        datasets[record['dataset_type']].append(record)
    if per_dataset:
        return {name: _pool(records, balanced, tie_std, cancel)
                for name, records in datasets.items()}
    pooled = _pool(clips, balanced, tie_std, cancel)
    return {name: {field: values.copy() for field, values in pooled.items()}
            for name in datasets}
