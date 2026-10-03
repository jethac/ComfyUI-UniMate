"""Validated CPU collation boundary around pinned MIT UniMate batch math."""
import numpy as np
import torch

from ._vendor.mixture_collate import mixture_batch_collate as _source_collate
from .training_transforms import _check, _integer

REQUIRED={'motion','parents','edge_indexs','max_joints','motion_length','start_idx',
          'tpos_first_frame','mean','std','joint_depths','joint_relations','joint_graph_dist'}
OPTIONAL={'max_motion_length','spectral_feats','joint_names_emb','offsets',
          'tpos_first_frame_parents','caption_emb','caption_tokens','object_type','caption','split_tag'}


def validate_samples(batch, *, max_workspace_bytes=512*1024*1024, cancel=None):
    """Validate numeric samples without allocating source batch tensors."""
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    if type(batch) not in (list,tuple) or len(batch)>4096:
        raise ValueError('Expected bounded sample list')
    batch=[value for value in batch if value is not None]
    if not batch:
        return []
    size=None
    input_bytes=0
    max_spectral,max_tokens,text_dim=0,1,None
    for value in batch:
        _check(cancel)
        if type(value) is not dict or not REQUIRED<=value.keys() or value.keys()-REQUIRED-OPTIONAL:
            raise ValueError('Invalid sample fields')
        _integer(value['max_joints'],'max_joints',1)
        _integer(value['motion_length'],'motion_length')
        _integer(value['start_idx'],'start_idx')
        if any(value[key]>np.iinfo(np.int64).max for key in ('max_joints','motion_length','start_idx')):
            raise ValueError('Scalar exceeds int64 range')
        arrays={key:array for key,array in value.items() if key not in
                {'max_joints','max_motion_length','motion_length','start_idx','object_type','caption','split_tag'}}
        for array in arrays.values():
            if type(array) is not np.ndarray or array.dtype.kind not in 'iuf':
                raise ValueError('Expected plain numeric sample arrays')
            input_bytes+=array.nbytes
        motion=value['motion']
        if motion.ndim!=3 or motion.shape[-1]!=12 or motion.dtype.kind!='f':
            raise ValueError('Invalid motion array')
        frames,joints,_=motion.shape
        capacity=value['max_joints']
        if frames<1 or joints<1 or joints>capacity or value['motion_length']>frames:
            raise ValueError('Invalid motion capacity or length')
        current=(capacity,frames)
        if size is None:
            size=current
        elif size!=current:
            raise ValueError('Samples have inconsistent padded capacities')
        if 'max_motion_length' in value:
            _integer(value['max_motion_length'],'max_motion_length',1)
            if value['max_motion_length']!=frames:
                raise ValueError('Motion differs from configured padded length')
        def shape(key,expected,kind):
            arr=value[key]
            if arr.shape!=expected or arr.dtype.kind not in kind:
                raise ValueError(f'Invalid {key} shape/dtype')
        shape('parents',(joints,),'i')
        parents=value['parents']
        if parents[0]!=-1 or np.any(parents[1:]<0) or np.any(parents[1:]>=np.arange(1,joints)):
            raise ValueError('Invalid ordered rooted parents')
        shape('edge_indexs',(2,2*(joints-1)),'iu')
        if np.any(value['edge_indexs']<0) or np.any(value['edge_indexs']>=joints):
            raise ValueError('Invalid edge indices')
        for key in ('tpos_first_frame','mean','std'):
            shape(key,(joints,12),'f')
        shape('joint_depths',(joints,),'iu')
        for key in ('joint_relations','joint_graph_dist'):
            shape(key,(joints,joints),'iu')
            if np.any(value[key]<0):
                raise ValueError('Negative embedding indices')
        if 'spectral_feats' in value:
            arr=value['spectral_feats']
            if arr.ndim!=2 or arr.shape[0]!=joints or arr.shape[1]<1 or arr.dtype.kind!='f':
                raise ValueError('Invalid spectral features')
            max_spectral=max(max_spectral,arr.shape[1])
        if 'joint_names_emb' in value:
            arr=value['joint_names_emb']
            if arr.ndim!=2 or arr.shape[0]!=joints or arr.shape[1]<1 or arr.dtype.kind!='f':
                raise ValueError('Invalid joint embeddings')
            dimension=arr.shape[1]
            if text_dim is not None and text_dim!=dimension:
                raise ValueError('Inconsistent embedding dimensions')
            text_dim=dimension
        for key,expected in (('offsets',(joints,3)),('tpos_first_frame_parents',(joints,12))):
            if key in value:
                shape(key,expected,'f')
        for key in ('caption_emb','caption_tokens'):
            if key in value:
                arr=value[key]
                if arr.ndim!=(1 if key=='caption_emb' else 2) or any(s<1 for s in arr.shape) or arr.dtype.kind!='f':
                    raise ValueError('Invalid caption embedding')
                dimension=arr.shape[-1]
                if text_dim is not None and text_dim!=dimension:
                    raise ValueError('Inconsistent embedding dimensions')
                text_dim=dimension
                if key=='caption_tokens':
                    max_tokens=max(max_tokens,arr.shape[0])
        if 'caption_tokens' in value and 'caption_emb' not in value:
            raise ValueError('Caption tokens require pooled view')
        for key in ('object_type','caption','split_tag'):
            if key in value and (type(value[key]) is not str or len(value[key].encode('utf-8'))>1024*1024):
                raise ValueError('Invalid sample label')
    capacity,frames=size
    estimate=4*input_bytes+len(batch)*(64*capacity*capacity+64*capacity*(frames*12+72+max_spectral+(text_dim or 0))
                                    +16*max_tokens*(text_dim or 0))
    if estimate>max_workspace_bytes:
        raise ValueError('Collation exceeds workspace budget')
    for value in batch:
        _check(cancel)
        for array in value.values():
            if isinstance(array,np.ndarray) and array.dtype.kind=='u' and np.any(array>np.iinfo(np.int64).max):
                raise ValueError('Integer array exceeds int64 range')
            if isinstance(array,np.ndarray) and (not np.isfinite(array).all() or
                    (array.dtype.kind=='f' and np.any(np.abs(array)>np.finfo(np.float32).max))):
                raise ValueError('Nonfinite or float32-unrepresentable sample')
        if np.any(value['std'].astype(np.float32)<=0):
            raise ValueError('Nonpositive standard deviations')
    return batch


def collate_samples(batch, *, max_workspace_bytes=512*1024*1024, cancel=None):
    """Return released CPU tensors; persisted portable batch encoding is separate."""
    batch=validate_samples(batch,max_workspace_bytes=max_workspace_bytes,cancel=cancel)
    if not batch:
        return None,None
    # Source retains parents/edges by reference. Own input arrays at this boundary.
    owned=[{key:array.copy() if isinstance(array,np.ndarray) else array
            for key,array in value.items()} for value in batch]
    _check(cancel)
    if torch.get_default_dtype()!=torch.float32:
        raise ValueError('Released collator requires float32 default tensor dtype')
    with torch.device('cpu'):
        result=_source_collate(owned)
    _check(cancel)
    return result
