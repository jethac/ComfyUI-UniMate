"""Released dataset split rules and portable CPU epoch sampling."""

from collections import defaultdict
import copy
import hashlib
import json
import math
import random

import numpy as np

from .contracts import MAX_JSON_BYTES, _digest, _unique_json_pairs, decode_arrays, encode_arrays
from .dataset_contracts import MAX_ENTRIES, _fields, _label, make_dataset, manifest_bytes, validate_dataset
from .statistics import UPSTREAM_REVISION, preflight_arrays

DEFAULT_MODES = dict(mixamo='clip', truebones='object_type', objaverse='object_type')
MAX_SAMPLING_BYTES = 1024 * 1024


def _cancel(callback):
    if callback is not None:
        callback()


def _number(value):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Expected a finite numeric option')
        return float(value)
    except OverflowError as error:
        raise ValueError('Numeric option exceeds range') from error


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('Seed/epoch requires an unsigned 64-bit integer')


def _identity(dataset):
    return hashlib.sha256(manifest_bytes(dict(schema='unimate.dataset.file.v1',
        manifest=dataset['manifest'], files=sorted(dataset['files'])))).hexdigest()


def split_dataset(dataset, ratio, seed, options, *, cancel=None):
    ratio = _number(ratio)
    if not 0 <= ratio < 1:
        raise ValueError('Split ratio must be in [0,1)')
    _seed(seed)
    if type(options) is not str or len(options.encode('utf-8')) > MAX_JSON_BYTES:
        raise ValueError('Split options require bounded JSON text')
    options = json.loads(options, object_pairs_hook=_unique_json_pairs)
    if type(options) is not dict or not set(options) <= {'modes', 'explicit_eval_objects'}:
        raise ValueError('Unsupported split options')
    modes = dict(DEFAULT_MODES)
    overrides = options.get('modes', {})
    explicit = options.get('explicit_eval_objects', {})
    if type(overrides) is not dict or type(explicit) is not dict:
        raise ValueError('Split rules require dataset mappings')
    for label, mode in overrides.items():
        _label(label)
        if mode not in ('clip', 'object_type'):
            raise ValueError('Unsupported dataset split mode')
        modes[label] = mode
    for label, objects in explicit.items():
        _label(label)
        if type(objects) is not list or len(objects) > MAX_ENTRIES:
            raise ValueError('Explicit holdouts require a bounded object label list')
        for obj in objects:
            _label(obj)
    validate_dataset(dataset, cancel=cancel)
    grouped = defaultdict(lambda: defaultdict(list))
    for clip in dataset['manifest']['clips']:
        _cancel(cancel)
        grouped[clip['dataset_type']][clip['object_type']].append(clip['id'])
    rng, evaluation, missing = random.Random(seed), set(), {}
    for label in sorted(set(explicit) - set(grouped)):
        _cancel(cancel)
        if explicit[label]:
            missing[label] = sorted(set(explicit[label]))
    for label in sorted(grouped):
        _cancel(cancel)
        objects = grouped[label]
        held_out = set(explicit.get(label, []))
        if held_out:
            selected = sorted(held_out & set(objects))
            absent = sorted(held_out - set(objects))
            if absent:
                missing[label] = absent
            if len(selected) >= len(objects):
                raise ValueError('Explicit holdout would leave no training object type')
            for obj in selected:
                evaluation.update(objects[obj])
        elif ratio > 0:
            if modes.get(label, 'clip') == 'object_type':
                names = sorted(objects)
                count = min(round(len(names) * ratio), max(0, len(names) - 1))
                if count:
                    rng.shuffle(names)
                    for obj in names[:count]:
                        evaluation.update(objects[obj])
            else:
                for obj in sorted(objects):
                    _cancel(cancel)
                    names = sorted(objects[obj])
                    count = min(round(len(names) * ratio), max(0, len(names) - 1))
                    if count:
                        rng.shuffle(names)
                        evaluation.update(names[:count])
    manifest = copy.deepcopy(dataset['manifest'])
    for clip in manifest['clips']:
        clip['split'] = 'eval' if clip['id'] in evaluation else 'train'
    value = make_dataset(manifest, dataset['files'], cancel=cancel)
    report = dict(upstream_revision=UPSTREAM_REVISION, source_sha256=_identity(dataset),
                  ratio=ratio, seed=seed, modes=modes,
                  explicit_eval_objects=explicit, missing_objects=missing,
                  train_ids=[c['id'] for c in manifest['clips'] if c['split'] == 'train'],
                  eval_ids=[c['id'] for c in manifest['clips'] if c['split'] == 'eval'])
    _cancel(cancel)
    return value, report


def _weights(clips, alpha, dataset_alpha, cancel):
    groups, datasets = defaultdict(list), defaultdict(list)
    for index, clip in enumerate(clips):
        _cancel(cancel)
        groups[(clip['dataset_type'], clip['object_type'])].append(index)
    for key in groups:
        datasets[key[0]].append(key)
    weights = np.zeros(len(clips))
    try:
        if dataset_alpha is None:
            for indices in groups.values():
                _cancel(cancel)
                weights[indices] = len(indices) ** (-alpha)
        else:
            for keys in datasets.values():
                _cancel(cancel)
                count = sum(len(groups[key]) for key in keys)
                share = count ** (1 - dataset_alpha)
                masses = {key: len(groups[key]) ** (1 - alpha) for key in keys}
                normalizer = sum(masses.values())
                for key in keys:
                    weights[groups[key]] = share * masses[key] / normalizer / len(groups[key])
        total = weights.sum()
        if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isfinite(total) or total <= 0:
            raise ValueError('Sampling exponents produce invalid weights')
        weights /= total
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Sampling weights underflow after normalization')
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError('Sampling exponents exceed numeric range') from error
    return weights


def _indices(weights, epoch):
    import torch
    generator = torch.Generator(device='cpu')
    generator.manual_seed(epoch)
    return torch.multinomial(torch.as_tensor(weights, dtype=torch.double, device='cpu'),
                             len(weights), replacement=True, generator=generator).numpy()


def sampling_plan(dataset, *, alpha=.5, dataset_alpha=None, epoch=0, cancel=None):
    alpha = _number(alpha)
    if dataset_alpha is not None:
        dataset_alpha = _number(dataset_alpha)
    _seed(epoch)
    validate_dataset(dataset, cancel=cancel)
    clips = [clip for clip in dataset['manifest']['clips'] if clip['split'] == 'train']
    if not clips:
        raise ValueError('Sampling requires training clips')
    weights = _weights(clips, alpha, dataset_alpha, cancel)
    _cancel(cancel)
    arrays = encode_arrays(weights=weights, indices=_indices(weights, epoch))
    value = dict(schema='unimate.sampling.v1', dataset_sha256=_identity(dataset),
                 clip_ids=[c['id'] for c in clips], alpha=alpha, dataset_alpha=dataset_alpha,
                 epoch=epoch, upstream_revision=UPSTREAM_REVISION, arrays=arrays,
                 sha256=hashlib.sha256(arrays).hexdigest())
    validate_sampling(value)
    _cancel(cancel)
    return value


def validate_sampling(value, dataset=None):
    _fields(value, {'schema', 'dataset_sha256', 'clip_ids', 'alpha', 'dataset_alpha',
                    'epoch', 'upstream_revision', 'arrays', 'sha256'})
    if value['schema'] != 'unimate.sampling.v1' or value['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Unsupported sampling schema or revision')
    _number(value['alpha'])
    if value['dataset_alpha'] is not None:
        _number(value['dataset_alpha'])
    _seed(value['epoch'])
    for field in ('dataset_sha256', 'sha256'):
        _digest(value[field], field)
    ids = value['clip_ids']
    if type(ids) is not list or not 1 <= len(ids) <= MAX_ENTRIES:
        raise ValueError('Sampling requires bounded training clip IDs')
    for clip_id in ids:
        _label(clip_id)
    if len(set(ids)) != len(ids):
        raise ValueError('Sampling clip IDs must be unique')
    preflight_arrays(value['arrays'], MAX_SAMPLING_BYTES, 2)
    if hashlib.sha256(value['arrays']).hexdigest() != value['sha256']:
        raise ValueError('Sampling payload digest mismatch')
    arrays = decode_arrays(value['arrays'])
    if set(arrays) != {'weights', 'indices'}:
        raise ValueError('Sampling requires weights and indices')
    weights, indices = arrays['weights'], arrays['indices']
    if (weights.dtype != np.dtype('float64') or indices.dtype != np.dtype('int64')
            or weights.shape != (len(ids),) or indices.shape != (len(ids),)):
        raise ValueError('Sampling requires float64 weights/int64 indices per clip')
    if (np.any(weights <= 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
            or np.any(indices < 0) or np.any(indices >= len(ids))):
        raise ValueError('Invalid sampling probability or index')
    if dataset is not None:
        validate_dataset(dataset)
        clips = [clip for clip in dataset['manifest']['clips'] if clip['split'] == 'train']
        if _identity(dataset) != value['dataset_sha256'] or [c['id'] for c in clips] != ids:
            raise ValueError('Sampling source dataset identity or training order mismatch')
        expected = _weights(clips, value['alpha'], value['dataset_alpha'], None)
        if not np.array_equal(weights, expected) or not np.array_equal(indices, _indices(expected, value['epoch'])):
            raise ValueError('Sampling values differ from declared source/options/epoch')
