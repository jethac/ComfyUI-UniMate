"""Portable dataset-normalization outputs with explicit source identity."""

import hashlib
import io
import zipfile

import numpy as np

from .contracts import _bytes, _digest, _members, decode_arrays, encode_arrays
from .dataset_contracts import MAX_ENTRIES, _fields, _label, manifest_bytes, training_records
from .dataset_stats import compute_statistics

UPSTREAM_REVISION = '2c5b384715aa63d8639b1ed7eb74bfe614570c7a'
MAX_STATISTICS_ARRAY_BYTES = 8 * 1024 * 1024
ARRAY_FIELDS = {'mean_root', 'std_root', 'mean_local', 'std_local'}
VALUE_FIELDS = {'schema', 'dataset_sha256', 'clip_count', 'datasets', 'options',
                'upstream_revision', 'arrays', 'sha256'}


def preflight_arrays(payload, limit, count):
    """Bound declared expansion before reading any numeric member."""
    _bytes(payload, limit, 'Statistics payload')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            _members(archive, limit, limit, count)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError('Invalid statistics numeric archive') from error


def validate_statistics(value):
    _fields(value, VALUE_FIELDS)
    if value['schema'] != 'unimate.statistics.v1' or value['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Unsupported statistics schema or source revision')
    for field in ('dataset_sha256', 'sha256'):
        _digest(value[field], field)
    if type(value['clip_count']) is not int or not 1 <= value['clip_count'] <= MAX_ENTRIES:
        raise ValueError('Statistics require a positive training clip count')
    labels = value['datasets']
    if type(labels) is not list or not 1 <= len(labels) <= MAX_ENTRIES:
        raise ValueError('Statistics require bounded dataset labels')
    for label in labels:
        _label(label)
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate statistics dataset label')
    _fields(value['options'], {'per_dataset', 'balanced', 'tie_std'})
    if any(type(option) is not bool for option in value['options'].values()):
        raise ValueError('Statistics options must be Boolean')
    preflight_arrays(value['arrays'], MAX_STATISTICS_ARRAY_BYTES, 4)
    if hashlib.sha256(value['arrays']).hexdigest() != value['sha256']:
        raise ValueError('Statistics array digest mismatch')
    arrays = decode_arrays(value['arrays'])
    if set(arrays) != ARRAY_FIELDS:
        raise ValueError('Statistics require four root/local arrays')
    for field, array in arrays.items():
        if array.dtype != np.dtype('float64') or array.shape != (len(labels), 12):
            raise ValueError('Statistics require float64 (datasets,12) arrays')
        if field.startswith('std_') and np.any(array <= 0):
            raise ValueError('Statistics standard deviations must be positive')


def dataset_statistics(dataset, *, per_dataset=True, balanced=False, tie_std=False, cancel=None):
    records = training_records(dataset, cancel=cancel)
    stats = compute_statistics(records, per_dataset=per_dataset, balanced=balanced,
                               tie_std=tie_std, cancel=cancel)
    labels = sorted(stats)
    arrays = encode_arrays(**{field: np.stack([stats[label][field] for label in labels])
                              for field in sorted(ARRAY_FIELDS)})
    envelope = dict(schema='unimate.dataset.file.v1', manifest=dataset['manifest'],
                    files=sorted(dataset['files']))
    value = dict(schema='unimate.statistics.v1',
                 dataset_sha256=hashlib.sha256(manifest_bytes(envelope)).hexdigest(),
                 clip_count=len(records), datasets=labels,
                 options=dict(per_dataset=per_dataset, balanced=balanced, tie_std=tie_std),
                 upstream_revision=UPSTREAM_REVISION, arrays=arrays,
                 sha256=hashlib.sha256(arrays).hexdigest())
    validate_statistics(value)
    return value
