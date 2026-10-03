"""Portable released batches; CPU tensors exist only at the consuming boundary."""
import copy
import hashlib
import io
import zipfile

import numpy as np
import torch

from .contracts import _digest,_members,decode_arrays,encode_arrays
from .dataset_contracts import _fields,_label,manifest_bytes
from .statistics import UPSTREAM_REVISION,preflight_arrays
from .training_collation import collate_samples
from .training_sample_contracts import _provenance,validate_training_sample
from .training_transforms import _check,_integer

FLOAT_FIELDS={'motion','tpos_first_frame','mean','std','spectral_feats','joint_names_emb',
              'offsets','tpos_first_frame_parents','caption_emb','caption_tokens'}
INT_FIELDS={'crop_start_ind','motion_length','n_joints','joint_relations','graph_dist',
            'joint_depths','parents','edge_indexs'}
BOOL_FIELDS={'lengths_mask','joint_mask','caption_mask'}
FIELDS={'schema','upstream_revision','samples','metadata','arrays','sha256','identity_sha256'}
MAX_BYTES=256*1024*1024


def _identity(value):
    return hashlib.sha256(manifest_bytes({k:v for k,v in value.items()
        if k not in ('arrays','identity_sha256')})).hexdigest()


def _envelope(value):
    _fields(value,FIELDS)
    if value['schema']!='unimate.training_batch.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Unsupported training batch schema/source')
    refs=value['samples']
    if type(refs) is not list or not 1<=len(refs)<=4096:
        raise ValueError('Expected 1–4096 ordered sample references')
    for ref in refs:
        _fields(ref,{'identity_sha256','provenance'})
        _digest(ref['identity_sha256'],'sample identity')
        _provenance(ref['provenance'])
    _fields(value['metadata'],{'object_type','caption','parent_dtypes'})
    for key in ('object_type','caption','parent_dtypes'):
        labels=value['metadata'][key]
        if type(labels) is not list or len(labels)!=len(refs):
            raise ValueError('Batch metadata differs from sample count')
        if any(type(s) is not str or len(s.encode('utf-8'))>1024*1024 for s in labels):
            raise ValueError('Invalid batch labels')
    for dtype in value['metadata']['parent_dtypes']:
        if dtype not in ('|i1','<i2','<i4','<i8'):
            raise ValueError('Unsupported source parent dtype')
    for label in value['metadata']['object_type']:
        _label(label)
    if any(not caption or '\0' in caption for caption in value['metadata']['caption']):
        raise ValueError('Invalid training caption')
    manifest_bytes({k:v for k,v in value.items() if k!='arrays'})
    for key in ('sha256','identity_sha256'):
        _digest(value[key],key)
    if _identity(value)!=value['identity_sha256']:
        raise ValueError('Training batch identity digest mismatch')


def validate_training_batch(value, *, cancel=None,max_workspace_bytes=512*1024*1024):
    """Validate source invariants and return owned source-format CPU tensors."""
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    _envelope(value)
    try:
        preflight_arrays(value['arrays'],min(MAX_BYTES,max_workspace_bytes//8),len(FLOAT_FIELDS|INT_FIELDS|BOOL_FIELDS))
    except ValueError as error:
        raise ValueError('Batch archive exceeds workspace/archive budget or is invalid') from error
    if hashlib.sha256(value['arrays']).hexdigest()!=value['sha256']:
        raise ValueError('Training batch array digest mismatch')
    _check(cancel)
    a=decode_arrays(value['arrays'])
    if a.keys()!=FLOAT_FIELDS|INT_FIELDS|BOOL_FIELDS:
        raise ValueError('Invalid batch array fields')
    for fields,dtype in ((FLOAT_FIELDS,np.float32),(INT_FIELDS,np.int64),(BOOL_FIELDS,np.bool_)):
        if any(a[key].dtype!=dtype or not np.isfinite(a[key]).all() for key in fields):
            raise ValueError('Invalid batch array dtype or values')
    b=len(value['samples'])
    if a['motion'].ndim!=4 or a['motion'].shape[0]!=b or a['motion'].shape[2]!=12:
        raise ValueError('Invalid batch motion shape')
    _,j,_,t=a['motion'].shape
    if not 1<=j<=4096 or not 1<=t<=4096:
        raise ValueError('Invalid batch capacities')
    def shape(key,expected):
        if a[key].shape!=expected:
            raise ValueError('Invalid batch shape: '+key)
    for key in ('n_joints','motion_length','crop_start_ind'):
        shape(key,(b,))
    counts=a['n_joints']
    if np.any(counts<1) or np.any(counts>j):
        raise ValueError('Invalid batch joint counts')
    for key in ('tpos_first_frame','mean','std','tpos_first_frame_parents'):
        shape(key,(b,j,12))
    shape('offsets',(b,j,3))
    shape('joint_depths',(b,j))
    for key in ('joint_relations','graph_dist'):
        shape(key,(b,j,j))
    shape('joint_mask',(b,1,1,j))
    shape('lengths_mask',(b,1,1,t))
    for key in ('spectral_feats','joint_names_emb'):
        if a[key].ndim!=3 or a[key].shape[:2]!=(b,j) or a[key].shape[2]<1:
            raise ValueError('Invalid batch embedding shape')
    d=a['joint_names_emb'].shape[2]
    shape('caption_emb',(b,d))
    if a['caption_tokens'].ndim!=3 or a['caption_tokens'].shape[0]!=b or a['caption_tokens'].shape[2]!=d:
        raise ValueError('Invalid batch caption tokens')
    k=a['caption_tokens'].shape[1]
    if not 1<=k<=512 or d>4096:
        raise ValueError('Invalid batch caption capacity')
    shape('caption_mask',(b,k))
    shape('parents',(int(counts.sum()),))
    shape('edge_indexs',(2,2*int((counts-1).sum())))
    samples=[]
    parent_start=edge_start=0
    for i,n in enumerate(counts):
        _check(cancel)
        n=int(n)
        token_count=int(a['caption_mask'][i].sum())
        if token_count<1 or not np.array_equal(a['caption_mask'][i],np.arange(k)<token_count):
            raise ValueError('Invalid batch caption mask')
        parents=a['parents'][parent_start:parent_start+n]
        dtype=np.dtype(value['metadata']['parent_dtypes'][i])
        if np.any(parents<np.iinfo(dtype).min) or np.any(parents>np.iinfo(dtype).max):
            raise ValueError('Parent dtype cannot represent indices')
        sample=dict(motion=a['motion'][i,:n].transpose(2,0,1),parents=parents.astype(dtype),
            edge_indexs=a['edge_indexs'][:,edge_start:edge_start+2*(n-1)],
            max_joints=j,max_motion_length=t,motion_length=int(a['motion_length'][i]),
            start_idx=int(a['crop_start_ind'][i]),joint_graph_dist=a['graph_dist'][i,:n,:n],
            joint_relations=a['joint_relations'][i,:n,:n],joint_depths=a['joint_depths'][i,:n],
            caption_tokens=a['caption_tokens'][i,:token_count],caption_emb=a['caption_emb'][i],
            object_type=value['metadata']['object_type'][i],caption=value['metadata']['caption'][i])
        for key in ('tpos_first_frame','mean','std','spectral_feats','joint_names_emb','offsets','tpos_first_frame_parents'):
            sample[key]=a[key][i,:n]
        samples.append(sample)
        parent_start+=n
        edge_start+=2*(n-1)
    result=collate_samples(samples,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    rebuilt=_arrays(*result)
    if any(not np.array_equal(rebuilt[key],a[key]) for key in a):
        raise ValueError('Batch padding/masks differ from released collator')
    _check(cancel)
    return result


def _arrays(motion,cond):
    result={'motion':motion.numpy()}
    result.update({key:item.numpy() for key,item in cond.items() if isinstance(item,torch.Tensor)})
    result['parents']=np.concatenate(cond['parents']).astype(np.int64)
    result['edge_indexs']=np.concatenate([edge.numpy() for edge in cond['edge_indexs']],axis=1)
    return result


def collate_training_samples(values, *, cancel=None,max_workspace_bytes=512*1024*1024):
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    if type(values) not in (list,tuple) or not 1<=len(values)<=4096:
        raise ValueError('Expected 1–4096 training samples')
    # Preflight every payload before decoding any of them.
    total=0
    for value in values:
        _check(cancel)
        if type(value) is not dict or type(value.get('arrays')) is not bytes:
            raise ValueError('Invalid portable training sample')
        try:
            with zipfile.ZipFile(io.BytesIO(value['arrays'])) as archive:
                total+=sum(info.file_size for info in _members(archive,MAX_BYTES,MAX_BYTES,15))
        except (OSError,zipfile.BadZipFile) as error:
            raise ValueError('Invalid sample archive') from error
        if total>min(MAX_BYTES,max_workspace_bytes//8):
            raise ValueError('Aggregate sample decoding exceeds workspace budget')
    samples=[validate_training_sample(v,cancel=cancel,max_workspace_bytes=max_workspace_bytes) for v in values]
    motion,cond=collate_samples(samples,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    arrays=_arrays(motion,cond)
    if sum(a.nbytes for a in arrays.values())>min(MAX_BYTES,max_workspace_bytes//8):
        raise ValueError('Batch serialization exceeds workspace budget')
    payload=encode_arrays(**arrays)
    value=dict(schema='unimate.training_batch.v1',upstream_revision=UPSTREAM_REVISION,
        samples=[dict(identity_sha256=v['identity_sha256'],provenance=copy.deepcopy(v['provenance'])) for v in values],
        metadata=dict(object_type=cond['object_type'],caption=cond['caption'],
                      parent_dtypes=[p.dtype.str for p in cond['parents']]),arrays=payload,
        sha256=hashlib.sha256(payload).hexdigest())
    value['identity_sha256']=_identity(value)
    validate_training_batch(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    return value
