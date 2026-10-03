"""Bind prepared feature clips, statistics and real text views to encoded samples.

Conditioning/statistics assembly adapted from MIT UniMate, copyright (c) 2026
Linzhan Mou. See _vendor/LICENSE-UniMate and _vendor/SOURCES.json.
"""
import hashlib
import io
import zipfile

import numpy as np

from .contracts import _members, decode_arrays
from .dataset_contracts import _label, manifest_bytes, validate_dataset
from .dataset_selection import _identity as dataset_identity
from .statistics import validate_statistics
from .training_augmentation import augment_sample,_validate_sample
from .training_samples import assemble_sample
from .training_sample_contracts import make_training_sample
from .training_text import validate_text_cache
from .training_transforms import _check, _integer
from ._vendor import topology_utils as topology


def text_cache_identity(value):
    return hashlib.sha256(manifest_bytes({key:value[key] for key in
        ('schema','encoder','upstream_revision','texts','sha256')})).hexdigest()


def statistics_identity(value):
    return hashlib.sha256(manifest_bytes({key:item for key,item in value.items() if key!='arrays'})).hexdigest()


def validate_training_conditioning(cond,*,cancel=None,max_workspace_bytes=512*1024*1024):
    """Check the source augmentation topology contract without motion/text IO.

    Reuse the owning augmentation validator with one-frame shape placeholders;
    real motion/statistics/embeddings are validated during sample assembly.
    """
    required={'parents','offsets','tpos_first_frame','spectral_feats','edge_indexs',
        'joint_graph_dists','joint_relations','joint_depths'}
    if type(cond) is not dict or not required<=cond.keys():
        raise ValueError('Prepared conditioning lacks required training topology fields')
    parents,spectral=cond['parents'],cond['spectral_feats']
    if (type(parents) is not np.ndarray or parents.ndim!=1 or not 1<=len(parents)<=4096
        or type(spectral) is not np.ndarray or spectral.ndim!=2 or not 1<=spectral.shape[1]<=4096):
        raise ValueError('Invalid training topology dimensions')
    joints=len(parents)
    if 96*joints**2+512*joints>max_workspace_bytes:
        raise ValueError('Training topology validation exceeds workspace budget')
    _validate_sample(dict(motion=np.zeros((1,joints,12),dtype=np.float32),parents=parents,
        tpos=cond['tpos_first_frame'],offsets=cond['offsets'],spectral_feats=spectral,
        edge_indexs=cond['edge_indexs'],joint_graph_dist=cond['joint_graph_dists'],
        joint_relations=cond['joint_relations'],joint_depths=cond['joint_depths'],
        joint_names_emb=np.zeros((joints,1),dtype=np.float32),
        mean=np.zeros((joints,12),dtype=np.float32),std=np.ones((joints,12),dtype=np.float32)),
        spectral.shape[1],cancel)


def _preflight(dataset,statistics,cache,workspace):
    if (type(dataset) is not dict or type(dataset.get('files')) is not dict
            or type(statistics) is not dict or type(cache) is not dict):
        raise ValueError('Invalid prepared training inputs')
    payloads=list(dataset['files'].values())+[statistics.get('arrays'),cache.get('arrays')]
    if any(type(payload) is not bytes for payload in payloads):
        raise ValueError('Training inputs require numeric bytes')
    limit=workspace//8
    if sum(map(len,payloads))>limit:
        raise ValueError('Training inputs exceed workspace budget')
    expanded=0
    for payload in payloads:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                expanded+=sum(info.file_size for info in _members(archive,limit,limit,128))
        except (OSError,zipfile.BadZipFile) as error:
            raise ValueError('Invalid training numeric archive') from error
        if expanded>limit:
            raise ValueError('Expanded training inputs exceed workspace budget')


def produce_training_sample(dataset,statistics,cache,clip_id,*,mode='tpos',
        max_motion_length,max_joints,augmentation='none',augmentation_seed=0,crop_seed=0,
        start_idx=None,realign_feature=True,ground_rest=True,embedding_policy='cached',
        max_freqs=8,addition_policy='released',enabled=(),cancel=None,
        max_workspace_bytes=512*1024*1024):
    """Use already prepared feature clips; raw feature extraction is separate.

    Augmentation and crop seeds are independent explicit streams. Grounding here
    affects rest conditioning only; input motion features are never re-extracted.
    """
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    _integer(max_freqs,'max_freqs',1)
    if max_freqs>4096:
        raise ValueError('Spectral width exceeds supported bound')
    if type(ground_rest) is not bool or embedding_policy not in ('cached','fresh'):
        raise ValueError('Invalid rest/text policy')
    _label(clip_id)
    _preflight(dataset,statistics,cache,max_workspace_bytes)
    validate_dataset(dataset,cancel=cancel)
    validate_statistics(statistics)
    text_arrays=validate_text_cache(cache)
    if statistics['dataset_sha256']!=dataset_identity(dataset):
        raise ValueError('Statistics source dataset identity mismatch')
    clip=next((entry for entry in dataset['manifest']['clips'] if entry['id']==clip_id),None)
    if clip is None or clip['split']!='train' or not clip['caption']:
        raise ValueError('Select a captioned training clip')
    if clip['dataset_type'] not in statistics['datasets']:
        raise ValueError('Statistics do not cover selected dataset')
    top=next(entry for entry in dataset['manifest']['topologies'] if entry['id']==clip['topology_id'])
    cond=decode_arrays(dataset['files'][top['conditioning']])
    validate_training_conditioning(cond,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    features=decode_arrays(dataset['files'][clip['features']])['features']
    cleaned=cond['clean_joint_names'].tolist()
    clean=all(str(name).strip() for name in cleaned)
    names=cleaned if clean else cond['joint_names'].tolist()
    records={}
    offset=0
    for index,text in enumerate(cache['texts']):
        length=int(text_arrays['lengths'][index])
        tokens=text_arrays['tokens'][offset:offset+length]
        if embedding_policy=='cached':
            pooled=text_arrays['pooled'][index]
        else:
            with np.errstate(over='ignore',invalid='ignore'):
                pooled=tokens.mean(axis=0)
            if not np.isfinite(pooled).all():
                raise ValueError('Nonfinite pooled training text')
        records[text]=(tokens,pooled)
        offset+=length
    if any(text not in records for text in names+[clip['caption']]):
        raise ValueError('Text cache misses joint names or caption')
    parents=cond['parents']
    rest=cond['tpos_first_frame'].copy()
    if ground_rest:
        rest[:,1]-=rest[:,1].min()
    offsets=rest.copy()
    offsets[1:]=rest[1:]-rest[parents[1:]]
    stats_arrays=decode_arrays(statistics['arrays'])
    row=statistics['datasets'].index(clip['dataset_type'])
    mean,std=np.zeros((len(parents),12)),np.zeros((len(parents),12))
    for field,array in (('mean',mean),('std',std)):
        array[0]=stats_arrays[field+'_root'][row]
        array[1:]=stats_arrays[field+'_local'][row]
    spectral=cond.get('spectral_feats')
    if spectral is None:
        raise ValueError('Prepared conditioning lacks spectral features')
    if spectral.shape[-1]!=max_freqs:
        _check(cancel)
        if 64*len(parents)*max_freqs+96*len(parents)**2>max_workspace_bytes:
            raise ValueError('Spectral recomputation exceeds workspace budget')
        spectral,_=topology.compute_laplacian_eigenvectors(parents,max_freqs=max_freqs)
    required={'edge_indexs','joint_graph_dists','joint_relations','joint_depths'}
    if not required<=cond.keys():
        raise ValueError('Prepared conditioning lacks topology fields')
    aug=dict(motion=features,parents=parents,tpos=rest,offsets=offsets,mean=mean,std=std,
        spectral_feats=spectral,joint_names_emb=np.stack([records[name][1] for name in names]),
        edge_indexs=cond['edge_indexs'],joint_graph_dist=cond['joint_graph_dists'],
        joint_relations=cond['joint_relations'],joint_depths=cond['joint_depths'])
    aug,report=augment_sample(aug,augmentation,augmentation_seed,enabled=enabled,
        max_freqs=max_freqs,addition_policy=addition_policy,cancel=cancel,
        max_workspace_bytes=max_workspace_bytes)
    caption_tokens,caption_emb=records[clip['caption']]
    numeric=assemble_sample(aug,object_type=clip['object_type'],caption=clip['caption'],
        caption_tokens=caption_tokens,caption_emb=caption_emb,mode=mode,
        max_motion_length=max_motion_length,max_joints=max_joints,crop_seed=crop_seed,
        start_idx=start_idx,realign_feature=realign_feature,max_freqs=max_freqs,
        cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    report['conditioning']=dict(joint_names='cleaned' if clean else 'raw_fallback',
        ground_rest=ground_rest,embedding_policy=embedding_policy,max_freqs=max_freqs,
        feature_policy='prepared_input_unchanged')
    provenance=dict(dataset_sha256=dataset_identity(dataset),clip_id=clip_id,
        statistics_sha256=statistics_identity(statistics),text_cache_sha256=text_cache_identity(cache))
    options=dict(mode=mode,crop_seed=crop_seed,start_idx=start_idx,
                 realign_feature=realign_feature,augmentation=report)
    return make_training_sample(numeric,provenance,options,cancel=cancel,
                                max_workspace_bytes=max_workspace_bytes)
