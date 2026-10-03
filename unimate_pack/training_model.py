"""Validated scratch reconstruction of all released training backbone axes."""
import math

import torch

from ._vendor import denoiser
from .inference import _LOCK

DEFAULTS=dict(attention='graph',text_cond='adaln',feature_len=12,
    max_motion_length=60,max_joints=71,max_depth=19,latent_dim=512,ff_size=2048,
    num_layers=8,num_heads=8,dropout=0.,use_spectral_rope=False,max_freqs=8,
    use_signnet=True,cond_mode='no_cond',cond_mask_prob=0.,text_dim=768,
    use_joint_name_emb=False,use_graph_emb=False,use_depth_emb=False,
    concat_parent_features=False,num_tpos_queries=0,inject_tpos_to_adaln=False,
    use_graph_attn_bias=True,share_graph_attn_bias=False,gradient_checkpointing=False)
_LIMITS=dict(feature_len=(12,12),max_motion_length=(1,4096),max_joints=(1,4096),
    max_depth=(0,4096),latent_dim=(4,8192),ff_size=(1,32768),num_layers=(1,128),
    num_heads=(1,128),max_freqs=(1,4096),text_dim=(1,8192),num_tpos_queries=(0,4096))


def model_options(options=None):
    if options is None:
        options={}
    if type(options) is not dict or options.keys()-DEFAULTS.keys():
        raise ValueError('Invalid training model options')
    result={**DEFAULTS,**options}
    for key,(low,high) in _LIMITS.items():
        if type(result[key]) is not int or not low<=result[key]<=high:
            raise ValueError('Invalid '+key)
    for key,default in DEFAULTS.items():
        if type(default) is bool and type(result[key]) is not bool:
            raise ValueError('Invalid '+key)
    for key in ('dropout','cond_mask_prob'):
        value=result[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid '+key)
    if result['attention'] not in ('full','graph') or result['text_cond'] not in ('adaln','cross_attn'):
        raise ValueError('Invalid model axes')
    if result['cond_mode'] not in ('text','no_cond'):
        raise ValueError('Invalid conditioning mode')
    latent,heads=result['latent_dim'],result['num_heads']
    divisor=4 if result['attention']=='full' else 2
    if latent%heads or (latent//heads)%divisor:
        raise ValueError('Invalid attention head dimensions')
    return result


def _constructor(config):
    cls=getattr(denoiser,'UniMate'+config['attention'].title()+
        ('AdaLN' if config['text_cond']=='adaln' else 'CrossAttn'))
    kwargs={key:value for key,value in config.items() if key not in
        ('attention','text_cond','use_graph_attn_bias','share_graph_attn_bias','gradient_checkpointing')}
    if config['attention']=='graph':
        kwargs.update(share_graph_attn_bias=config['share_graph_attn_bias'],
            gradient_checkpointing=config['gradient_checkpointing'])
        if config['text_cond']=='adaln':
            kwargs['use_graph_attn_bias']=config['use_graph_attn_bias']
    return cls,kwargs


def training_model_layout(options,*,cancel=None,max_model_bytes=2*1024**3):
    config=model_options(options)
    if type(max_model_bytes) is not int or max_model_bytes<=0:
        raise ValueError('Invalid model budget')
    if cancel:
        cancel()
    cls,kwargs=_constructor(config)
    with _LOCK,torch.random.fork_rng(devices=[]),torch.device('meta'):
        probe=cls(**kwargs)
    size=sum(t.numel()*t.element_size() for t in (*probe.parameters(),*probe.buffers()))
    if size>max_model_bytes:
        raise ValueError('Training model exceeds budget')
    state=probe.state_dict(keep_vars=True)
    groups={}
    for name,tensor in state.items():
        groups.setdefault(id(tensor),[]).append(name)
    layout={name:(tuple(tensor.shape),tensor.dtype) for name,tensor in state.items()}
    if cancel:
        cancel()
    return layout,[names for names in groups.values() if len(names)>1]


def create_training_model(options=None,*,seed=0,cancel=None,max_model_bytes=2*1024**3):
    """Create a CPU float32 model without consuming process RNG streams.

    Meta construction measures persistent parameters/buffers before real
    allocation. This budget excludes activations and optimizer state; sessions
    and public job execution must enforce their separate workspace limits.
    text_dim must be bound to the validated text cache by the job owner.
    """
    config=model_options(options)
    if torch.get_default_dtype()!=torch.float32:
        raise ValueError('Training model initialization requires float32 default dtype')
    if type(seed) is not int or not 0<=seed<2**64:
        raise ValueError('Invalid model seed')
    if type(max_model_bytes) is not int or max_model_bytes<=0:
        raise ValueError('Invalid model budget')
    if cancel:
        cancel()
    cls,kwargs=_constructor(config)
    with _LOCK,torch.random.fork_rng(devices=[]):
        with torch.device('meta'):
            probe=cls(**kwargs)
        size=sum(t.numel()*t.element_size() for t in
            (*probe.parameters(),*probe.buffers()))
        del probe
        if size>max_model_bytes:
            raise ValueError('Training model exceeds budget')
        if cancel:
            cancel()
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        with torch.device('cpu'):
            model=cls(**kwargs).float().train()
        if cancel:
            cancel()
        return model
