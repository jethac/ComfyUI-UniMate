"""Portable ragged training text views, without pickle or live encoder state."""
import copy
import hashlib

import numpy as np

from .contracts import _digest, decode_arrays, encode_arrays
from .dataset_contracts import _fields, manifest_bytes
from .statistics import UPSTREAM_REVISION, preflight_arrays
from .training_transforms import _check, _integer

MAX_TEXT_BYTES=64*1024*1024
MAX_TEXTS=4096
FIELDS={'schema','encoder','upstream_revision','texts','arrays','sha256'}


def _encoder_identity(identity):
    _fields(identity,{'encoder_type','encoder_version','artifact_sha256'})
    _digest(identity['artifact_sha256'],'encoder artifact')
    for key in ('encoder_type','encoder_version'):
        if type(identity[key]) is not str or not identity[key] or len(identity[key].encode('utf-8'))>1024 or '\0' in identity[key]:
            raise ValueError('Invalid encoder identity')


def _texts(texts):
    if type(texts) is not list or not 1<=len(texts)<=MAX_TEXTS:
        raise ValueError('Expected bounded nonempty text list')
    if any(type(text) is not str or '\0' in text or len(text.encode('utf-8'))>1024*1024 for text in texts):
        raise ValueError('Invalid training text')
    manifest_bytes(dict(prompt=texts))


def validate_text_cache(value, *, encoder_identity=None):
    _fields(value,FIELDS)
    if value['schema']!='unimate.text_cache.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Unsupported text cache schema or source')
    _encoder_identity(value['encoder'])
    if encoder_identity is not None:
        _encoder_identity(encoder_identity)
        if value['encoder']!=encoder_identity:
            raise ValueError('Text cache encoder identity mismatch')
    _texts(value['texts'])
    if value['texts']!=sorted(set(value['texts'])):
        raise ValueError('Cache texts must be sorted and unique')
    _digest(value['sha256'],'text arrays')
    preflight_arrays(value['arrays'],MAX_TEXT_BYTES,3)
    if hashlib.sha256(value['arrays']).hexdigest()!=value['sha256']:
        raise ValueError('Text array digest mismatch')
    arrays=decode_arrays(value['arrays'])
    if set(arrays)!={'tokens','lengths','pooled'}:
        raise ValueError('Invalid text arrays')
    tokens,lengths,pooled=(arrays[key] for key in ('tokens','lengths','pooled'))
    count=len(value['texts'])
    if (lengths.dtype!=np.dtype('int64') or lengths.shape!=(count,) or np.any(lengths<1)
            or np.any(lengths>512) or tokens.dtype!=np.dtype('float32') or tokens.ndim!=2
            or not 1<=tokens.shape[1]<=4096 or tokens.shape[0]!=sum(map(int,lengths))
            or pooled.dtype!=np.dtype('float32') or pooled.shape!=(count,tokens.shape[1])):
        raise ValueError('Invalid ragged token/pooled dimensions')
    return arrays


def text_views(value, text, *, policy='cached'):
    arrays=validate_text_cache(value)
    if policy not in ('cached','fresh') or type(text) is not str or text not in value['texts']:
        raise ValueError('Missing cache text or invalid pooling policy')
    index=value['texts'].index(text)
    offset=sum(map(int,arrays['lengths'][:index]))
    tokens=arrays['tokens'][offset:offset+int(arrays['lengths'][index])].copy()
    with np.errstate(over='ignore', invalid='ignore'):
        pooled=(arrays['pooled'][index].copy() if policy=='cached' else tokens.mean(axis=0))
    if not np.isfinite(pooled).all():
        raise ValueError('Nonfinite pooled training text')
    return tokens,pooled


def build_text_cache(texts, encode, encoder_identity, *, chunk_size=256, existing=None,
                     max_array_bytes=MAX_TEXT_BYTES, cancel=None):
    """Encode missing unique strings; retain actual token and encoder-pooled views.

    Callable encode returns (ragged float32 sequences, float32 pooled matrix).
    Callers own model selection and ComfyUI device/cancellation management.
    """
    _check(cancel)
    _texts(texts)
    _encoder_identity(encoder_identity)
    _integer(chunk_size,'chunk_size',1)
    _integer(max_array_bytes,'max_array_bytes',1)
    if chunk_size>256 or max_array_bytes>MAX_TEXT_BYTES:
        raise ValueError('Chunk or array budget exceeds cache limits')
    records={}
    requested=list(dict.fromkeys(texts))
    size=0
    if existing is not None:
        arrays=validate_text_cache(existing,encoder_identity=encoder_identity)
        offset=0
        for i,text in enumerate(existing['texts']):
            length=int(arrays['lengths'][i])
            if text in requested:
                records[text]=(arrays['tokens'][offset:offset+length],arrays['pooled'][i])
                size+=records[text][0].nbytes+records[text][1].nbytes+8
            offset+=length
    todo=[text for text in requested if text not in records]
    dimension=None
    for start in range(0,len(todo),chunk_size):
        _check(cancel)
        chunk=todo[start:start+chunk_size]
        sequences,pooled=encode(chunk)
        if (type(sequences) not in (list,tuple) or len(sequences)!=len(chunk)
                or type(pooled) is not np.ndarray or pooled.dtype!=np.dtype('float32')
                or pooled.ndim!=2 or pooled.shape[0]!=len(chunk) or not np.isfinite(pooled).all()):
            raise ValueError('Invalid encoder pooled output')
        for i,(text,tokens) in enumerate(zip(chunk,sequences)):
            _check(cancel)
            if (type(tokens) is not np.ndarray or tokens.dtype!=np.dtype('float32')
                    or tokens.ndim!=2 or not 1<=tokens.shape[0]<=512
                    or not 1<=tokens.shape[1]<=4096 or tokens.shape[1]!=pooled.shape[1]
                    or not np.isfinite(tokens).all()):
                raise ValueError('Invalid encoder token output')
            if dimension is not None and dimension!=tokens.shape[1]:
                raise ValueError('Inconsistent encoder dimensions')
            dimension=tokens.shape[1]
            size+=tokens.nbytes+pooled[i].nbytes+8
            if size>max_array_bytes:
                raise ValueError('Text arrays exceed budget')
            records[text]=(tokens.copy(),pooled[i].copy())
    if size>max_array_bytes:
        raise ValueError('Text arrays exceed budget')
    ordered=sorted(records)
    dimensions={records[text][0].shape[1] for text in ordered}
    if len(dimensions)!=1:
        raise ValueError('Cache hit/miss embedding dimensions differ')
    _check(cancel)
    payload=encode_arrays(tokens=np.concatenate([records[text][0] for text in ordered]),
        lengths=np.array([len(records[text][0]) for text in ordered],dtype=np.int64),
        pooled=np.stack([records[text][1] for text in ordered]))
    if len(payload)>max_array_bytes:
        raise ValueError('Serialized text arrays exceed budget')
    value=dict(schema='unimate.text_cache.v1',encoder=copy.deepcopy(encoder_identity),
               upstream_revision=UPSTREAM_REVISION,texts=ordered,arrays=payload,
               sha256=hashlib.sha256(payload).hexdigest())
    validate_text_cache(value)
    _check(cancel)
    return value
