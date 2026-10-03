"""Numeric encoded-sample assembly adapted from MIT UniMate.

Copyright (c) 2026 Linzhan Mou. See _vendor/LICENSE-UniMate and SOURCES.json.
Encoder/cache production and portable identities belong to their own boundaries.
"""
import numpy as np

from . import training_transforms as transforms
from .training_augmentation import FIELDS, _validate_sample


def assemble_sample(aug, *, object_type, caption, caption_tokens, caption_emb,
                    mode, max_motion_length, max_joints, crop_seed=0, start_idx=None,
                    realign_feature=True, max_freqs=8, cancel=None,
                    max_workspace_bytes=512*1024*1024):
    """Assemble an already augmented encoded dictionary in released order.

    Caption embeddings are supplied explicitly and never regenerated or pooled.
    This numeric result has no persisted dataset/cache identity on its own.
    """
    transforms._check(cancel)
    for name,value in (('object_type',object_type),('caption',caption)):
        if type(value) is not str or not value or len(value.encode('utf-8')) > 1024*1024:
            raise ValueError(f'Invalid training {name}')
    transforms._integer(max_motion_length,'max_motion_length',1)
    transforms._integer(max_joints,'max_joints',1)
    transforms._integer(max_freqs,'max_freqs',1)
    transforms._integer(max_workspace_bytes,'max_workspace_bytes',1)
    if type(realign_feature) is not bool:
        raise ValueError('realign_feature must be boolean')
    transforms._array(caption_tokens,(2,))
    transforms._array(caption_emb,(1,))
    if caption_tokens.shape[-1] != caption_emb.shape[0]:
        raise ValueError('Caption token and pooled dimensions differ')
    if (type(aug) is not dict or set(aug) != FIELDS
            or any(type(value) is not np.ndarray for value in aug.values())
            or aug['motion'].ndim != 3):
        raise ValueError('Invalid augmented sample arrays')
    joints=aug['motion'].shape[1]
    if joints > max_joints:
        raise ValueError('Augmented skeleton exceeds configured joint capacity')
    estimate=(16*sum(value.nbytes for value in aug.values())
              +12*8*joints*joints+32*max_motion_length*joints*12
              +caption_tokens.nbytes+caption_emb.nbytes)
    if estimate > max_workspace_bytes:
        raise ValueError('Sample exceeds workspace budget')
    _validate_sample(aug,max_freqs,cancel)
    if caption_emb.shape[0] != aug['joint_names_emb'].shape[1]:
        raise ValueError('Caption and joint embedding dimensions differ')
    motion,start=transforms.apply_cropping(aug['motion'],mode,max_motion_length,
        start_idx=start_idx,seed=crop_seed,cancel=cancel)
    if realign_feature and start > 0:
        motion=transforms.realign_clip(motion,aug['parents'],cancel=cancel,
                                       max_workspace_bytes=max_workspace_bytes)
    pad=np.zeros((joints,9))
    pad[:,:6]=[1,0,0,0,1,0]
    rest=np.concatenate([aug['tpos'],pad],axis=-1)
    conditions=transforms.extract_conditions(motion,rest,mode,cancel=cancel)
    motion=transforms.apply_normalization(conditions['motion'],aug['mean'],aug['std'],cancel=cancel)
    rest=transforms.apply_normalization(conditions['tpos_first_frame'],aug['mean'],aug['std'],cancel=cancel)
    motion,length=transforms.apply_padding(motion,max_motion_length,cancel=cancel,
                                           max_workspace_bytes=max_workspace_bytes)
    parent_features=transforms.build_parent_features(rest,aug['parents'],cancel=cancel)
    result={key:aug[key].copy() for key in ('parents','edge_indexs','offsets',
        'joint_graph_dist','joint_relations','joint_depths','spectral_feats',
        'joint_names_emb','mean','std')}
    result.update(motion=motion,max_motion_length=max_motion_length,motion_length=length,
                  max_joints=max_joints,tpos_first_frame=rest,object_type=object_type,
                  start_idx=start,caption=caption,caption_emb=caption_emb.copy(),
                  caption_tokens=caption_tokens.copy(),**parent_features)
    transforms._check(cancel)
    return result
