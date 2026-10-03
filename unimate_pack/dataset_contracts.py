"""Content-addressed numeric training shards, independent of mesh restrictions."""

import copy
import hashlib
import io
import json
import math
import zipfile

from .contracts import MAX_ARRAY_BYTES, MAX_JSON_BYTES, _digest, _members, decode_arrays

MAX_EXPANDED_BYTES = MAX_ARRAY_BYTES
MAX_ENTRIES = 4096
MAX_FILES = 8192
_TOPOLOGY_FIELDS = {'id', 'source_rig_id', 'conditioning'}
_CLIP_FIELDS = {'id', 'topology_id', 'dataset_type', 'object_type', 'caption',
                'split', 'features', 'origin', 'fps'}


def _fields(value, fields):
    if type(value) is not dict or set(value) != fields:
        raise ValueError('Invalid dataset fields')


def _label(value):
    if (type(value) is not str or not value or len(value) > 255
            or value in ('.', '..') or any(c in value for c in '/\\:\0')):
        raise ValueError('Dataset identifiers must be portable labels')


def manifest_bytes(manifest):
    try:
        result = json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                            sort_keys=True, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError('Invalid dataset JSON manifest') from error
    if len(result) > MAX_JSON_BYTES:
        raise ValueError('Dataset manifest exceeds size limit')
    return result


def _manifest(manifest):
    _fields(manifest, {'topologies', 'clips'})
    for key in ('topologies', 'clips'):
        if type(manifest[key]) is not list or not 1 <= len(manifest[key]) <= MAX_ENTRIES:
            raise ValueError('Dataset requires bounded nonempty topology and clip lists')
    topology_ids, clip_ids, referenced = set(), set(), set()
    for entry in manifest['topologies']:
        _fields(entry, _TOPOLOGY_FIELDS)
        _digest(entry['id'], 'topology')
        _digest(entry['conditioning'], 'conditioning')
        if entry['id'] != entry['conditioning'] or entry['id'] in topology_ids:
            raise ValueError('Topology ID must uniquely identify conditioning')
        if entry['source_rig_id'] is not None:
            _digest(entry['source_rig_id'], 'source rig')
        topology_ids.add(entry['id'])
        referenced.add(entry['conditioning'])
    for entry in manifest['clips']:
        if type(entry) is not dict:
            raise ValueError('Invalid dataset clip fields')
        _fields(entry, _CLIP_FIELDS | ({'source_rig_id'} if 'source_rig_id' in entry else set()))
        if entry.get('source_rig_id') is not None:
            _digest(entry['source_rig_id'], 'clip source rig')
        for key in ('id', 'dataset_type', 'object_type'):
            _label(entry[key])
        if entry['id'] in clip_ids:
            raise ValueError('Duplicate dataset clip ID')
        clip_ids.add(entry['id'])
        _digest(entry['topology_id'], 'topology')
        if entry['topology_id'] not in topology_ids:
            raise ValueError('Clip references missing topology')
        _digest(entry['features'], 'features')
        referenced.add(entry['features'])
        if type(entry['split']) is not str or entry['split'] not in ('train', 'eval'):
            raise ValueError('Dataset split must be train or eval')
        if type(entry['fps']) is not int or entry['fps'] != 30:
            raise ValueError('Dataset clips require FPS 30')
        caption = entry['caption']
        if type(caption) is not str or '\0' in caption:
            raise ValueError('Caption must be plain text')
        try:
            if len(caption.encode('utf-8')) > 16384:
                raise ValueError('Dataset caption exceeds size limit')
        except UnicodeError as error:
            raise ValueError('Invalid caption Unicode') from error
        origin = entry['origin']
        if type(origin) is not list or len(origin) != 3:
            raise ValueError('Clip origin requires three finite numbers')
        try:
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in origin):
                raise ValueError('Clip origin requires three finite numbers')
        except OverflowError as error:
            raise ValueError('Clip origin exceeds numeric range') from error
    manifest_bytes(dict(schema='unimate.dataset.file.v1', manifest=manifest, files=sorted(referenced)))
    return referenced


def _files(files, referenced, cancel):
    if type(files) is not dict or not 1 <= len(files) <= MAX_FILES or set(files) != referenced:
        raise ValueError('Dataset files must match all referenced payloads')
    total, expanded = 0, 0
    for digest, payload in files.items():
        if cancel is not None:
            cancel()
        _digest(digest, 'dataset file')
        if type(payload) is not bytes or not payload:
            raise ValueError('Dataset files require nonempty bytes')
        total += len(payload)
        if total > MAX_ARRAY_BYTES:
            raise ValueError('Dataset payloads exceed shard size limit')
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Dataset file digest mismatch')
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                infos = _members(archive, MAX_ARRAY_BYTES, MAX_ARRAY_BYTES, 128)
                expanded += sum(info.file_size for info in infos)
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError('Invalid dataset numeric payload') from error
        if expanded > MAX_EXPANDED_BYTES:
            raise ValueError('Dataset expanded payloads exceed shard size limit')


def _topology(payload):
    arrays = decode_arrays(payload)
    required = {'parents', 'offsets', 'tpos_first_frame', 'joint_names', 'clean_joint_names'}
    if not required <= set(arrays):
        raise ValueError('Dataset topology lacks required conditioning arrays')
    parents = arrays['parents']
    if (parents.ndim != 1 or parents.dtype.kind not in 'iu'
            or not 2 <= len(parents) <= 4096 or parents[0] != -1
            or any(not 0 <= parent < child for child, parent in enumerate(parents[1:], 1))):
        raise ValueError('Dataset topology requires connected ordered parents')
    joints = len(parents)
    for field in ('offsets', 'tpos_first_frame'):
        array = arrays[field]
        if array.dtype.kind != 'f' or array.shape != (joints, 3):
            raise ValueError('Dataset rest positions and offsets must be floating (J,3)')
    for field in ('joint_names', 'clean_joint_names'):
        array = arrays[field]
        if array.dtype.kind != 'U' or array.shape != (joints,):
            raise ValueError('Dataset joint names must be Unicode (J,)')
    return joints


def _features(payload):
    arrays = decode_arrays(payload)
    if set(arrays) != {'features'}:
        raise ValueError('Dataset motion payload contains only features')
    features = arrays['features']
    if (features.dtype.kind != 'f' or features.dtype.itemsize not in (2, 4, 8)
            or features.ndim != 3 or features.shape[0] < 1 or features.shape[2] != 12):
        raise ValueError('Dataset features require finite floating (T,J,12)')
    return features


def validate_dataset(value, *, cancel=None):
    _fields(value, {'schema', 'manifest', 'files'})
    if value['schema'] != 'unimate.dataset.v1':
        raise ValueError('Unsupported dataset schema')
    referenced = _manifest(value['manifest'])
    _files(value['files'], referenced, cancel)
    topologies = {}
    for entry in value['manifest']['topologies']:
        if cancel is not None:
            cancel()
        topologies[entry['id']] = _topology(value['files'][entry['conditioning']])
    # Decode each feature payload once; retain only its joint count here.
    feature_counts = {}
    for entry in value['manifest']['clips']:
        if cancel is not None:
            cancel()
        digest = entry['features']
        if digest not in feature_counts:
            feature_counts[digest] = _features(value['files'][digest]).shape[1]
        if feature_counts[digest] != topologies[entry['topology_id']]:
            raise ValueError('Dataset features and topology joint counts differ')


def make_dataset(manifest, files, *, cancel=None):
    value = dict(schema='unimate.dataset.v1', manifest=manifest, files=files)
    validate_dataset(value, cancel=cancel)
    return dict(schema=value['schema'], manifest=copy.deepcopy(manifest), files=dict(files))


def training_records(value, *, cancel=None):
    """Return deduplicated read-only arrays from train clips only."""
    validate_dataset(value, cancel=cancel)
    records, cache = [], {}
    for entry in value['manifest']['clips']:
        if cancel is not None:
            cancel()
        if entry['split'] != 'train':
            continue
        digest = entry['features']
        if digest not in cache:
            cache[digest] = _features(value['files'][digest])
            cache[digest].flags.writeable = False
        records.append(dict(dataset_type=entry['dataset_type'], object_type=entry['object_type'],
                            features=cache[digest]))
    if not records:
        raise ValueError('Dataset has no training clips for statistics')
    return records
