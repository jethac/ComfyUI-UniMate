"""Explicit installed-bundle initialization; optimizer state is never imported."""
import hashlib
import io
import zipfile
import struct

import torch
from safetensors.torch import load

from .bundle import inspect_bundle,_json
from .contracts import _digest,_name
from .dataset_contracts import _fields
from .training_checkpoint import _header
from .training_model import DEFAULTS,model_options,training_model_layout
from .training_transforms import _check,_integer
from .upstream import validate_config

DEFAULT_WORKSPACE=8*1024**3


def validate_initialization(value):
    _fields(value,{'bundle_sha256','denoiser_sha256','weights'})
    _digest(value['bundle_sha256'],'initialization bundle')
    _digest(value['denoiser_sha256'],'initialization weights')
    if value['weights'] not in ('raw','ema'):
        raise ValueError('Invalid initialization weight selection')


def inspect_initialization(model,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    if type(model) is not dict or type(model.get('bundle')) is not bytes:
        raise ValueError('Initialization requires a portable installed model')
    payload=model['bundle']
    if 2*len(payload)+64*1024**2>max_workspace_bytes:
        raise ValueError('Initialization bundle exceeds workspace budget')
    _fields(model,{'schema','bundle','sha256','name'})
    _digest(model['sha256'],'initialization model')
    _name(model['name'],'.unimate')
    if model['schema']!='unimate.model.v1' or hashlib.sha256(payload).hexdigest()!=model['sha256']:
        raise ValueError('Invalid initialization model identity')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        size=archive.getinfo('denoiser.safetensors').file_size
        if 2*len(payload)+8*size+64*1024**2>max_workspace_bytes:
            raise ValueError('Initialization weights exceed workspace budget')
        manifest=inspect_bundle(payload)
        _check(cancel)
        config=_json(archive.read('config.json'))
        validate_config(config)
        options={key:value for key,value in config['model'].items() if key in DEFAULTS}
        options.update({key:config['dataset'][key] for key in
            ('feature_len','max_motion_length','max_joints','max_depth')},text_dim=768)
        options=model_options(options)
        weights=archive.read('denoiser.safetensors')
    info=_header(weights,max_workspace_bytes-2*len(payload),named=True)
    layout,aliases=training_model_layout(options,cancel=cancel,max_model_bytes=max_workspace_bytes//8)
    if info.keys()!=layout.keys() or any((item.shape,item.dtype)!=layout[name] for name,item in info.items()):
        raise ValueError('Initialization weights differ from selected architecture')
    length=struct.unpack('<Q',weights[:8])[0]
    header=_json(weights[8:8+length])
    data=memoryview(weights)[8+length:]
    for names in aliases:
        hashes=[]
        for name in names:
            _check(cancel)
            start,end=header[name]['data_offsets']
            hashes.append(hashlib.sha256(data[start:end]).digest())
        if len(set(hashes))!=1:
            raise ValueError('Inconsistent shared initialization weight alias')
    descriptor=dict(bundle_sha256=model['sha256'],denoiser_sha256=hashlib.sha256(weights).hexdigest(),
        weights=manifest['weights'])
    _check(cancel)
    return descriptor,options,weights


def validate_weight_state(model,state,*,cancel=None):
    expected=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=expected.keys():
        raise ValueError('Initialization weight inventory mismatch')
    aliases={}
    for name,target in expected.items():
        _check(cancel)
        tensor=state[name]
        if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu'
            or tensor.shape!=target.shape or tensor.dtype!=target.dtype):
            raise ValueError('Initialization weight shape/dtype mismatch: '+name)
        if tensor.is_floating_point():
            flat=tensor.reshape(-1)
            for offset in range(0,flat.numel(),1024**2):
                _check(cancel)
                if not torch.isfinite(flat[offset:offset+1024**2]).all():
                    raise ValueError('Nonfinite initialization weights')
        if target.numel():
            identity=(target.untyped_storage().data_ptr(),target.storage_offset(),tuple(target.shape),tuple(target.stride()))
            if identity in aliases and not torch.equal(tensor,state[aliases[identity]]):
                raise ValueError('Inconsistent shared initialization weight alias')
            aliases[identity]=name
    _check(cancel)


def apply_initialization(model,payload,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _check(cancel)
    info=_header(payload,max_workspace_bytes,named=True)
    expected=model.state_dict()
    if info.keys()!=expected.keys() or any(info[name].shape!=tuple(target.shape) or
            info[name].dtype!=target.dtype for name,target in expected.items()):
        raise ValueError('Initialization weight inventory/shape/dtype mismatch')
    _check(cancel)
    state=load(payload)
    validate_weight_state(model,state,cancel=cancel)
    # The caller owns this private model; all validation precedes mutation.
    model.load_state_dict(state,strict=True)
    _check(cancel)


def select_checkpoint_weights(model,checkpoint,selection):
    if selection not in ('raw','ema') or type(checkpoint) is not dict or 'model_state_dict' not in checkpoint:
        raise ValueError('Select explicit raw or EMA checkpoint weights')
    state=checkpoint['model_state_dict']
    validate_weight_state(model,state)
    if selection=='ema':
        ema=checkpoint.get('ema_state_dict')
        shadows=ema.get('shadow_params') if type(ema) is dict else None
        parameters=list(model.named_parameters())
        if type(shadows) is not list or len(shadows)!=len(parameters):
            raise ValueError('EMA parameter inventory mismatch; no raw fallback')
        state=dict(state)
        # Parameter aliases retain the same EMA value in every state_dict key.
        shadow_by_id={id(parameter):shadow for (_,parameter),shadow in zip(parameters,shadows)}
        for name,parameter in model.named_parameters(remove_duplicate=False):
            state[name]=shadow_by_id[id(parameter)]
        validate_weight_state(model,state)
    model.load_state_dict(state,strict=True)
