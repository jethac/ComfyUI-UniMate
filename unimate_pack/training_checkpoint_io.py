"""Checkpoint envelope persistence; full resume validation stays session-bound."""
import hashlib
import struct

import numpy as np

from .bundle import _json
from .contracts import MAX_JSON_BYTES,_digest
from .dataset_contracts import _fields,manifest_bytes
from .statistics import UPSTREAM_REVISION
from .training_checkpoint import (FIELDS,MAX_BYTES,_json_value,_header,_decode_tree,_identity,_position)
from .training_transforms import _check,_integer

MAGIC=b'UMTRAIN1'
MAX_ARCHIVE_BYTES=16+MAX_JSON_BYTES+MAX_BYTES
DEFAULT_WORKSPACE=8*1024**3
STATE_FIELDS={'paradigm','loss_options','options','device','model','trainable','modes',
    'optimizer','scheduler','ema','scaler','rng_cpu','rng_device','updates','batches','optimizer_updates'}


def _budget(size,workspace):
    _integer(workspace,'max_workspace_bytes',1)
    # Input/output byte copies plus bounded metadata/tree and scan temporaries.
    if size*2+64*1024**2>workspace:
        raise ValueError('Checkpoint file exceeds workspace budget')


def _finite_payload(payload,cancel):
    length=struct.unpack('<Q',payload[:8])[0]
    header=_json(payload[8:8+length])
    data=memoryview(payload)[8+length:]
    for entry in header.values():
        dtype=entry['dtype']
        if dtype not in ('F64','F32','F16','BF16','BOOL'):
            continue
        start,end=entry['data_offsets']
        numpy_dtype={'F64':'<f8','F32':'<f4','F16':'<f2','BF16':'<u2','BOOL':'u1'}[dtype]
        # Chunk boundaries are multiples of every supported item size.
        for offset in range(start,end,4*1024**2):
            _check(cancel)
            values=np.frombuffer(data[offset:min(offset+4*1024**2,end)],dtype=numpy_dtype)
            if dtype=='BF16':
                valid=not np.any((values&0x7f80)==0x7f80)
            elif dtype=='BOOL':
                valid=not np.any(values>1)
            else:
                valid=np.isfinite(values).all()
            if not valid:
                raise ValueError('Nonfinite or invalid numeric checkpoint file data')


def _validate(value,cancel,workspace):
    _check(cancel)
    _fields(value,FIELDS)
    payload=value['tensors']
    if type(payload) is not bytes or not 16<=len(payload)<=MAX_BYTES:
        raise ValueError('Invalid checkpoint tensor bytes')
    _budget(len(payload)+MAX_JSON_BYTES+16,workspace)
    metadata={key:item for key,item in value.items() if key!='tensors'}
    _json_value(metadata)
    if value['schema']!='unimate.training_checkpoint.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Unsupported checkpoint file source/schema')
    for name in ('sha256','identity_sha256'):
        _digest(value[name],name)
    if hashlib.sha256(payload).hexdigest()!=value['sha256'] or _identity(value)!=value['identity_sha256']:
        raise ValueError('Checkpoint file content digest mismatch')
    # Header validation builds TensorInfo only; the session's eight-copy tensor
    # decode budget is not an allocation made by this envelope reader.
    info=_header(payload,MAX_BYTES*8)
    state=_decode_tree(value['state'],info)
    _fields(state,STATE_FIELDS)
    _integer(state['updates'],'updates')
    _integer(state['batches'],'batches')
    _position(value['position'],state['batches'])
    _fields(value['binding'],{'architecture','datasets','sampling'})
    _fields(value['binding']['architecture'],{'class_name','config','initial_weights_sha256'})
    for key in ('runtime',):
        if type(value[key]) is not dict:
            raise ValueError('Invalid checkpoint file '+key)
    _finite_payload(payload,cancel)
    _check(cancel)


def dump_training_checkpoint(value,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _validate(value,cancel,max_workspace_bytes)
    metadata={key:item for key,item in value.items() if key!='tensors'}
    raw=manifest_bytes(dict(schema='unimate.training_checkpoint.file.v1',checkpoint=metadata))
    _check(cancel)
    return MAGIC+struct.pack('<Q',len(raw))+raw+value['tensors']


def load_training_checkpoint(payload,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _check(cancel)
    if type(payload) is not bytes or not 32<=len(payload)<=MAX_ARCHIVE_BYTES:
        raise ValueError('Invalid checkpoint archive bytes')
    _budget(len(payload),max_workspace_bytes)
    if payload[:8]!=MAGIC:
        raise ValueError('Unsupported checkpoint archive magic')
    length=struct.unpack('<Q',payload[8:16])[0]
    if not 1<=length<=MAX_JSON_BYTES or 16+length>=len(payload):
        raise ValueError('Invalid checkpoint archive metadata length')
    metadata=_json(payload[16:16+length])
    _fields(metadata,{'schema','checkpoint'})
    if metadata['schema']!='unimate.training_checkpoint.file.v1' or type(metadata['checkpoint']) is not dict:
        raise ValueError('Unsupported checkpoint archive schema')
    value={**metadata['checkpoint'],'tensors':payload[16+length:]}
    _validate(value,cancel,max_workspace_bytes)
    return value
