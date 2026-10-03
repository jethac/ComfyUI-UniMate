"""Bounded portable encoded training samples with explicit source provenance."""
import copy
import hashlib

import numpy as np

from .contracts import _digest, _json, decode_arrays, encode_arrays
from .dataset_contracts import _fields, _label, manifest_bytes
from .statistics import UPSTREAM_REVISION, preflight_arrays
from .training_collation import validate_samples
from .training_transforms import _check, _integer

MAX_SAMPLE_BYTES=64*1024*1024
ARRAY_FIELDS={'motion','parents','edge_indexs','offsets','joint_graph_dist',
    'joint_relations','joint_depths','spectral_feats','joint_names_emb','mean','std',
    'tpos_first_frame','tpos_first_frame_parents','caption_emb','caption_tokens'}
META_FIELDS={'max_motion_length','max_joints','motion_length','start_idx','object_type','caption'}
PROVENANCE_FIELDS={'dataset_sha256','clip_id','statistics_sha256','text_cache_sha256'}
VALUE_FIELDS={'schema','upstream_revision','provenance','options','metadata','arrays','sha256','identity_sha256'}


def _provenance(value):
    _fields(value,PROVENANCE_FIELDS)
    _label(value['clip_id'])
    for key in PROVENANCE_FIELDS-{'clip_id'}:
        _digest(value[key],key)


def _options(value):
    _fields(value,{'mode','crop_seed','start_idx','realign_feature','augmentation'})
    if value['mode'] not in ('tpos','first_frame') or type(value['realign_feature']) is not bool:
        raise ValueError('Invalid training sample options')
    _integer(value['crop_seed'],'crop_seed')
    if value['crop_seed']>=2**64:
        raise ValueError('Crop seed exceeds uint64 range')
    if value['start_idx'] is not None:
        _integer(value['start_idx'],'start_idx')
    if type(value['augmentation']) is not dict:
        raise ValueError('Augmentation report must be an object')
    _json(value)
    manifest_bytes(value)


def _identity(value):
    envelope={key:value[key] for key in ('schema','upstream_revision','provenance','options','metadata','sha256')}
    return hashlib.sha256(manifest_bytes(envelope)).hexdigest()


def validate_training_sample(value, *, expected_provenance=None, cancel=None,
                             max_workspace_bytes=512*1024*1024):
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    _fields(value,VALUE_FIELDS)
    if value['schema']!='unimate.training_sample.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Unsupported training sample schema/source')
    _provenance(value['provenance'])
    if expected_provenance is not None:
        _provenance(expected_provenance)
        if value['provenance']!=expected_provenance:
            raise ValueError('Training sample source identity mismatch')
    _options(value['options'])
    _fields(value['metadata'],META_FIELDS)
    for key in ('max_motion_length','max_joints','motion_length','start_idx'):
        _integer(value['metadata'][key],key,1 if key in ('max_motion_length','max_joints') else 0)
    if value['options']['start_idx'] is not None and value['options']['start_idx']!=value['metadata']['start_idx']:
        raise ValueError('Explicit crop start differs from recorded start')
    _label(value['metadata']['object_type'])
    caption=value['metadata']['caption']
    if type(caption) is not str or not caption or '\0' in caption:
        raise ValueError('Invalid training caption')
    _digest(value['sha256'],'sample arrays')
    _digest(value['identity_sha256'],'sample identity')
    if _identity(value)!=value['identity_sha256']:
        raise ValueError('Training sample identity digest mismatch')
    if max_workspace_bytes<4:
        raise ValueError('Sample decoding exceeds workspace budget')
    try:
        preflight_arrays(value['arrays'],min(MAX_SAMPLE_BYTES,max_workspace_bytes//4),len(ARRAY_FIELDS))
    except ValueError as error:
        raise ValueError('Sample archive exceeds workspace/archive budget or is invalid') from error
    if hashlib.sha256(value['arrays']).hexdigest()!=value['sha256']:
        raise ValueError('Training sample array digest mismatch')
    _check(cancel)
    arrays=decode_arrays(value['arrays'])
    if set(arrays)!=ARRAY_FIELDS:
        raise ValueError('Invalid training sample arrays')
    sample={**copy.deepcopy(value['metadata']),**arrays}
    validate_samples([sample],cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    _check(cancel)
    return sample


def make_training_sample(sample, provenance, options, *, cancel=None,
                         max_array_bytes=MAX_SAMPLE_BYTES,max_workspace_bytes=512*1024*1024):
    _check(cancel)
    _provenance(provenance)
    _options(options)
    _integer(max_array_bytes,'max_array_bytes',1)
    if max_array_bytes>MAX_SAMPLE_BYTES:
        raise ValueError('Sample array budget exceeds limit')
    if type(sample) is not dict or set(sample)!=ARRAY_FIELDS|META_FIELDS:
        raise ValueError('Invalid encoded sample fields')
    if options['start_idx'] is not None and options['start_idx']!=sample['start_idx']:
        raise ValueError('Explicit crop start differs from recorded start')
    arrays={key:sample[key] for key in sorted(ARRAY_FIELDS)}
    if any(type(array) is not np.ndarray for array in arrays.values()):
        raise ValueError('Sample requires plain numeric arrays')
    if sum(array.nbytes for array in arrays.values())>max_array_bytes:
        raise ValueError('Sample arrays exceed budget')
    validate_samples([sample],cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    payload=encode_arrays(**arrays)
    if len(payload)>max_array_bytes:
        raise ValueError('Serialized sample arrays exceed budget')
    value=dict(schema='unimate.training_sample.v1',upstream_revision=UPSTREAM_REVISION,
        provenance=copy.deepcopy(provenance),options=copy.deepcopy(options),
        metadata={key:copy.deepcopy(sample[key]) for key in META_FIELDS},arrays=payload,
        sha256=hashlib.sha256(payload).hexdigest())
    value['identity_sha256']=_identity(value)
    validate_training_sample(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    return value
