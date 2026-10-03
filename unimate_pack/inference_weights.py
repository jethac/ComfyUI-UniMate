"""Selected trained weights; no optimizer, scaler or RNG state in the output."""
from collections import OrderedDict
import hashlib
import struct

from safetensors.torch import load as load_tensors,save as save_tensors

from .bundle import _json
from .contracts import MAX_JSON_BYTES,_digest
from .dataset_contracts import _fields,manifest_bytes
from .statistics import UPSTREAM_REVISION
from .training_checkpoint import MAX_BYTES,_header,_decode_tree,_exact,_json_value
from .training_checkpoint_io import _validate as validate_checkpoint_envelope,_finite_payload
from .training_initialization import apply_initialization
from .training_job import validate_training_job_metadata
from .training_model import training_model_spec,create_training_model
from .training_transforms import _check,_integer

FIELDS={'schema','upstream_revision','job','weights','source_checkpoint_sha256','tensors','sha256','identity_sha256'}
MAGIC=b'UMWEIGHT'
MAX_ARCHIVE_BYTES=16+MAX_JSON_BYTES+MAX_BYTES
DEFAULT_WORKSPACE=8*1024**3


def _budget(size,workspace):
    _integer(workspace,'max_workspace_bytes',1)
    if size*8+64*1024**2>workspace:
        raise ValueError('Inference weights exceed workspace budget')


def _identity(value):
    return hashlib.sha256(manifest_bytes({k:v for k,v in value.items()
        if k not in ('tensors','identity_sha256')})).hexdigest()


def _job(value):
    validate_training_job_metadata(value)


def _layout(state,layout):
    if type(state) not in (dict,OrderedDict) or state.keys()!=layout.keys():
        raise ValueError('Inference model tensor inventory mismatch')
    for name,tensor in state.items():
        shape,dtype=layout[name]
        if tuple(tensor.shape)!=shape or tensor.dtype!=dtype:
            raise ValueError('Inference model tensor shape/dtype mismatch: '+name)


def _alias_integrity(payload,groups,cancel):
    length=struct.unpack('<Q',payload[:8])[0]
    header=_json(payload[8:8+length])
    data=memoryview(payload)[8+length:]
    for names in groups:
        hashes=[]
        for name in names:
            _check(cancel)
            start,end=header[name]['data_offsets']
            hashes.append(hashlib.sha256(data[start:end]).digest())
        if len(set(hashes))!=1:
            raise ValueError('Inconsistent inference weight alias')


def _preflight(checkpoint,selection,workspace,cancel):
    _check(cancel)
    if selection not in ('raw','ema'):
        raise ValueError('Select explicit raw or EMA inference weights')
    if type(checkpoint) is not dict or type(checkpoint.get('tensors')) is not bytes:
        raise ValueError('Expected portable training checkpoint')
    _budget(len(checkpoint['tensors']),workspace)
    validate_checkpoint_envelope(checkpoint,cancel,workspace)
    job=checkpoint['binding']['architecture']['config']
    _job(job)
    spec=training_model_spec(job['model'],cancel=cancel,max_model_bytes=workspace//8)
    architecture=dict(class_name=spec['class_name'],config=job,
        initial_weights_sha256=job.get('initialization',{}).get('denoiser_sha256'))
    expected=dict(architecture=architecture,datasets=job['datasets'],
        sampling=dict(**job['sampling'],job_sha256=job['sha256']))
    if not _exact(checkpoint['binding'],expected):
        raise ValueError('Inference checkpoint/job binding mismatch')
    infos=_header(checkpoint['tensors'],workspace)
    state=_decode_tree(checkpoint['state'],infos)
    if any(not _exact(state[key],job[field]) for key,field in
        (('paradigm','paradigm'),('options','optimizer'),('loss_options','loss'))):
        raise ValueError('Inference checkpoint training configuration mismatch')
    if state['updates']>job['optimizer']['num_steps']:
        raise ValueError('Inference checkpoint update count exceeds job')
    _layout(state['model'],spec['layout'])
    tensor_names={id(info):name for name,info in infos.items()}
    _alias_integrity(checkpoint['tensors'],[[tensor_names[id(state['model'][name])] for name in group]
        for group in spec['aliases']],cancel)
    flags=state['trainable']
    if type(flags) is not dict or not _exact(flags,spec['trainable']):
        raise ValueError('Inference checkpoint trainable parameter inventory mismatch')
    if selection=='ema':
        ema=state['ema']
        if not job['optimizer']['use_ema'] or type(ema) is not dict:
            raise ValueError('EMA is absent; no raw fallback')
        _fields(ema,{'decay','min_decay','optimization_step','update_after_step','use_ema_warmup','shadow_params'})
        expected_ema=dict(decay=job['optimizer']['ema_decay'],min_decay=0.,optimization_step=state['updates'],
            update_after_step=0,use_ema_warmup=True)
        if any(not _exact(ema[key],value) for key,value in expected_ema.items()):
            raise ValueError('Inference EMA configuration/counter mismatch')
        names=[name for name in spec['parameters'] if flags[name]]
        if type(ema['shadow_params']) is not list or len(ema['shadow_params'])!=len(names):
            raise ValueError('Inference EMA parameter inventory mismatch')
        _layout(dict(zip(names,ema['shadow_params'])),{name:spec['layout'][name] for name in names})
    _check(cancel)
    return job,spec


def make_inference_weights(checkpoint,selection,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    job,spec=_preflight(checkpoint,selection,max_workspace_bytes,cancel)
    tensors=load_tensors(checkpoint['tensors'])
    state=_decode_tree(checkpoint['state'],tensors)
    selected=dict(state['model'])
    if selection=='ema':
        names=[name for name in spec['parameters'] if state['trainable'][name]]
        selected.update(zip(names,state['ema']['shadow_params']))
        for aliases in spec['aliases']:
            canonical=next((name for name in aliases if name in spec['parameters']),None)
            if canonical is not None:
                for name in aliases:
                    selected[name]=selected[canonical]
    _check(cancel)
    # Materialize shared aliases for strict safetensors state_dict reconstruction.
    payload=save_tensors({name:tensor.contiguous().clone() for name,tensor in selected.items()})
    value=dict(schema='unimate.inference_weights.v1',upstream_revision=UPSTREAM_REVISION,
        job=_json(manifest_bytes(job)),weights=selection,source_checkpoint_sha256=checkpoint['identity_sha256'],
        tensors=payload,sha256=hashlib.sha256(payload).hexdigest())
    value['identity_sha256']=_identity(value)
    validate_inference_weights(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    return value


def validate_inference_weights(value,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _check(cancel)
    _fields(value,FIELDS)
    if type(value['tensors']) is not bytes:
        raise ValueError('Invalid inference weight bytes')
    _budget(len(value['tensors']),max_workspace_bytes)
    metadata={k:v for k,v in value.items() if k!='tensors'}
    _json_value(metadata)
    if value['schema']!='unimate.inference_weights.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Unsupported inference weights schema/source')
    if value['weights'] not in ('raw','ema'):
        raise ValueError('Invalid inference weight selection')
    for name in ('sha256','identity_sha256','source_checkpoint_sha256'):
        _digest(value[name],name)
    if hashlib.sha256(value['tensors']).hexdigest()!=value['sha256'] or _identity(value)!=value['identity_sha256']:
        raise ValueError('Inference weights content digest mismatch')
    _job(value['job'])
    spec=training_model_spec(value['job']['model'],cancel=cancel,max_model_bytes=max_workspace_bytes//8)
    info=_header(value['tensors'],max_workspace_bytes,named=True)
    _layout(info,spec['layout'])
    _finite_payload(value['tensors'],cancel)
    _alias_integrity(value['tensors'],spec['aliases'],cancel)
    _check(cancel)


def create_inference_model(value,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    validate_inference_weights(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    model=create_training_model(value['job']['model'],cancel=cancel,max_model_bytes=max_workspace_bytes//8)
    apply_initialization(model,value['tensors'],cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    return model.eval().requires_grad_(False)


def dump_inference_weights(value,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    validate_inference_weights(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    metadata=manifest_bytes({k:v for k,v in value.items() if k!='tensors'})
    _check(cancel)
    return MAGIC+struct.pack('<Q',len(metadata))+metadata+value['tensors']


def load_inference_weights(payload,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    _check(cancel)
    if type(payload) is not bytes or not 32<=len(payload)<=MAX_ARCHIVE_BYTES:
        raise ValueError('Invalid inference weights archive')
    _budget(len(payload),max_workspace_bytes)
    length=struct.unpack('<Q',payload[8:16])[0]
    if payload[:8]!=MAGIC or not 1<=length<=MAX_JSON_BYTES or 16+length>=len(payload):
        raise ValueError('Invalid inference weights file header')
    metadata=_json(payload[16:16+length])
    if type(metadata) is not dict or 'tensors' in metadata:
        raise ValueError('Invalid inference weights file metadata')
    value=dict(metadata,tensors=payload[16+length:])
    validate_inference_weights(value,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    return value
