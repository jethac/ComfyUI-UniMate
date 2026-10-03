"""Bounded safetensors/JSON training checkpoints; no object deserialization."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import struct
import sys

import numpy as np
import torch
from safetensors.torch import load as load_tensors,save as save_tensors

from .bundle import _json
from .contracts import _digest,MAX_JSON_BYTES
from .dataset_contracts import _fields,manifest_bytes
from .inference import _LOCK
from .statistics import UPSTREAM_REVISION
from .training_session import TrainingSession,_finite
from .training_transforms import _check,_integer

MAX_BYTES=2*1024**3
MAX_HEADER=4*1024**2
MAX_NODES=100000
DEFAULT_WORKSPACE=8*1024**3
DTYPES={'F64':torch.float64,'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,
    'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}
ITEMSIZE={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
FIELDS={'schema','upstream_revision','binding','position','runtime','state','tensors','sha256','identity_sha256'}


@dataclass(frozen=True)
class TensorInfo:
    shape:tuple
    dtype:torch.dtype


def _exact(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(_exact(a[k],b[k]) for k in a)
    if type(a) in (list,tuple):
        return len(a)==len(b) and all(_exact(x,y) for x,y in zip(a,b))
    return a==b


def _json_value(value,depth=0,counter=None):
    counter=[0,0] if counter is None else counter
    counter[0]+=1
    if depth>32 or counter[0]>MAX_NODES:
        raise ValueError('Checkpoint JSON nesting/count exceeds budget')
    if type(value) is dict:
        counter[1]+=2+max(0,len(value)-1)
        for key,item in value.items():
            if type(key) is not str or len(key)>1024:
                raise ValueError('Invalid checkpoint JSON key')
            counter[1]+=len(json.dumps(key,ensure_ascii=False).encode('utf-8'))+1
            if counter[1]>MAX_JSON_BYTES:
                raise ValueError('Checkpoint JSON exceeds encoded size budget')
            _json_value(item,depth+1,counter)
    elif type(value) is list:
        counter[1]+=2+max(0,len(value)-1)
        for item in value:
            _json_value(item,depth+1,counter)
    elif type(value) is str:
        if len(value)>65536 or len(value.encode('utf-8'))>65536 or '\0' in value:
            raise ValueError('Invalid checkpoint JSON string')
    elif type(value) is int:
        if not -(2**64)<=value<2**64:
            raise ValueError('Checkpoint integer exceeds range')
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite checkpoint JSON')
    elif value is not None and type(value) is not bool:
        raise ValueError('Unsupported checkpoint JSON value')
    if type(value) not in (dict,list):
        counter[1]+=len(json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8'))
    if counter[1]>MAX_JSON_BYTES:
        raise ValueError('Checkpoint JSON exceeds encoded size budget')


def _binding(value,session):
    _fields(value,{'architecture','datasets','sampling'})
    architecture=value['architecture']
    _fields(architecture,{'class_name','config','initial_weights_sha256'})
    cls=type(session.model)
    if architecture['class_name'] not in (cls.__module__+'.'+cls.__qualname__,model_class_identity(session.model)) or type(architecture['config']) is not dict:
        raise ValueError('Mismatched checkpoint architecture')
    if architecture['initial_weights_sha256'] is not None:
        _digest(architecture['initial_weights_sha256'],'initial weights')
    datasets=value['datasets']
    if type(datasets) is not list or not 1<=len(datasets)<=4096:
        raise ValueError('Invalid checkpoint dataset binding')
    for data in datasets:
        _fields(data,{'dataset_sha256','statistics_sha256','text_cache_sha256'})
        for name,digest in data.items():
            _digest(digest,name)
    if type(value['sampling']) is not dict:
        raise ValueError('Invalid checkpoint sampling binding')
    _json_value(value)
    manifest_bytes(value)


def model_class_identity(model):
    """Stable source identity for known backbones; no dynamic class imports."""
    from ._vendor import denoiser
    cls=type(model)
    for name in ('UniMateFullAdaLN','UniMateFullCrossAttn','UniMateGraphAdaLN','UniMateGraphCrossAttn'):
        if cls is getattr(denoiser,name):
            return 'unimate.denoiser.'+name
    return cls.__module__+'.'+cls.__qualname__


def _position(value,batches):
    _fields(value,{'epoch','batch_in_epoch','batches_per_epoch','consumed_batches','epoch_plan_sha256'})
    for name in ('epoch','batch_in_epoch','consumed_batches'):
        _integer(value[name],name)
    _integer(value['batches_per_epoch'],'batches_per_epoch',minimum=1)
    _digest(value['epoch_plan_sha256'],'epoch plan')
    if (value['batch_in_epoch']>=value['batches_per_epoch'] or value['consumed_batches']!=batches
        or value['epoch']*value['batches_per_epoch']+value['batch_in_epoch']!=batches):
        raise ValueError('Mismatched checkpoint data position')


def _runtime(session):
    cuda=session.device.type=='cuda'
    return dict(torch=str(torch.__version__),python=sys.version.split()[0],numpy=np.__version__,
        transformers=version('transformers'),torch_geometric=version('torch-geometric'),
        safetensors=version('safetensors'),default_dtype=str(torch.get_default_dtype()),
        platform=sys.platform,device=str(session.device),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cuda_version=torch.version.cuda if cuda else None,
        device_name=torch.cuda.get_device_name(session.device) if cuda else None,
        capability=list(torch.cuda.get_device_capability(session.device)) if cuda else None,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_version=torch.backends.cudnn.version(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled())


def _encode_tree(state):
    tensors={}
    counter=[0]
    def encode(value,depth=0):
        counter[0]+=1
        if depth>32 or counter[0]>MAX_NODES:
            raise ValueError('Checkpoint state nesting/count exceeds budget')
        if isinstance(value,torch.Tensor):
            if value.dtype not in DTYPES.values() or value.ndim>8:
                raise ValueError('Unsupported checkpoint tensor')
            name=f't{len(tensors):06d}'
            tensors[name]=value.detach().to('cpu').contiguous()
            return dict(kind='tensor',name=name)
        if type(value) in (dict,OrderedDict):
            items=[]
            for key,item in value.items():
                if type(key) not in (str,int):
                    raise ValueError('Invalid checkpoint state key')
                items.append([key,encode(item,depth+1)])
            return dict(kind='dict',ordered=type(value) is OrderedDict,items=items,
                metadata=encode(value._metadata,depth+1) if hasattr(value,'_metadata') else None)
        if type(value) in (tuple,list):
            return dict(kind='tuple' if type(value) is tuple else 'list',items=[encode(v,depth+1) for v in value])
        _json_value(value)
        return dict(kind='scalar',value=value)
    return encode(state),tensors


def _decode_tree(tree,tensors):
    seen=set()
    counter=[0]
    def decode(node,depth=0):
        counter[0]+=1
        if depth>32 or counter[0]>MAX_NODES or type(node) is not dict:
            raise ValueError('Invalid checkpoint state tree')
        kind=node.get('kind')
        if kind=='tensor':
            _fields(node,{'kind','name'})
            name=node['name']
            if type(name) is not str or name not in tensors or name in seen:
                raise ValueError('Invalid or duplicate checkpoint tensor reference')
            seen.add(name)
            return tensors[name]
        if kind=='scalar':
            _fields(node,{'kind','value'})
            if type(node['value']) in (dict,list):
                raise ValueError('Invalid checkpoint scalar')
            _json_value(node['value'])
            return node['value']
        if kind in ('list','tuple','dict'):
            _fields(node,{'kind','items','ordered','metadata'} if kind=='dict' else {'kind','items'})
            if type(node['items']) is not list or len(node['items'])>MAX_NODES:
                raise ValueError('Invalid checkpoint state sequence')
            if kind!='dict':
                values=[decode(v,depth+1) for v in node['items']]
                return tuple(values) if kind=='tuple' else values
            if type(node['ordered']) is not bool:
                raise ValueError('Invalid checkpoint dictionary type')
            result=OrderedDict() if node['ordered'] else {}
            for entry in node['items']:
                if type(entry) is not list or len(entry)!=2 or type(entry[0]) not in (str,int):
                    raise ValueError('Invalid checkpoint dictionary entry')
                key,value=entry
                if key in result:
                    raise ValueError('Duplicate checkpoint state key')
                _json_value(key)
                result[key]=decode(value,depth+1)
            if node['metadata'] is not None:
                if not node['ordered']:
                    raise ValueError('Metadata requires ordered model state')
                result._metadata=decode(node['metadata'],depth+1)
            return result
        raise ValueError('Unsupported checkpoint tree tag')
    result=decode(tree)
    if seen!=set(tensors):
        raise ValueError('Unused checkpoint tensors')
    return result


def _header(payload,workspace):
    if type(payload) is not bytes or not 8<len(payload)<=MAX_BYTES:
        raise ValueError('Invalid checkpoint tensor payload')
    if len(payload)*8>workspace:
        raise ValueError('Checkpoint tensor payload exceeds workspace budget')
    length=struct.unpack('<Q',payload[:8])[0]
    if not 2<=length<=MAX_HEADER or length>len(payload)-8:
        raise ValueError('Invalid safetensors header length')
    try:
        header=_json(payload[8:8+length])
    except (ValueError,UnicodeError,RecursionError) as error:
        raise ValueError('Invalid safetensors header') from error
    if type(header) is not dict or not 1<=len(header)<=MAX_NODES:
        raise ValueError('Invalid safetensors tensor inventory')
    result={}
    intervals=[]
    data_size=len(payload)-8-length
    for name,entry in header.items():
        if type(name) is not str or len(name)!=7 or name[0]!='t' or not name[1:].isascii() or not name[1:].isdecimal():
            raise ValueError('Invalid safetensors tensor name')
        _fields(entry,{'dtype','shape','data_offsets'})
        dtype=entry['dtype']
        shape=entry['shape']
        offsets=entry['data_offsets']
        if type(dtype) is not str or dtype not in DTYPES or type(shape) is not list or len(shape)>8:
            raise ValueError('Invalid safetensors dtype/shape')
        count=1
        for dim in shape:
            if type(dim) is not int or not 0<=dim<=2**31:
                raise ValueError('Invalid safetensors dimension')
            count*=dim
            if count>MAX_BYTES:
                raise ValueError('Excessive safetensors element count')
        size=count*ITEMSIZE[dtype]
        if (type(offsets) is not list or len(offsets)!=2 or any(type(v) is not int for v in offsets)
            or not 0<=offsets[0]<=offsets[1]<=data_size or offsets[1]-offsets[0]!=size):
            raise ValueError('Invalid safetensors offsets')
        if size:
            intervals.append(tuple(offsets))
        result[name]=TensorInfo(tuple(shape),DTYPES[dtype])
    cursor=0
    for start,end in sorted(intervals):
        if start!=cursor:
            raise ValueError('Overlapping or incomplete safetensors data')
        cursor=end
    if cursor!=data_size:
        raise ValueError('Unreferenced safetensors data')
    return result


def _tensor(value,shape,dtype):
    if not isinstance(value,(torch.Tensor,TensorInfo)) or tuple(value.shape)!=tuple(shape) or value.dtype!=dtype:
        raise ValueError('Mismatched checkpoint tensor dtype/shape')


def _state(state,session,*,numeric):
    current=session._raw_state()
    _fields(state,set(current))
    for key in ('paradigm','loss_options','options','device','trainable','modes'):
        if not _exact(state[key],current[key]):
            raise ValueError('Mismatched checkpoint session '+key)
    updates,batches=state['updates'],state['batches']
    _integer(updates,'updates')
    _integer(batches,'batches')
    accumulation=session.options['gradient_accumulation_steps']
    if updates>session.options['num_steps'] or not updates<=batches<=updates*accumulation:
        raise ValueError('Invalid checkpoint update/batch counters')
    model=state['model']
    if type(model) not in (dict,OrderedDict) or model.keys()!=current['model'].keys():
        raise ValueError('Mismatched checkpoint model tensors')
    if not _exact(getattr(model,'_metadata',None),getattr(current['model'],'_metadata',None)):
        raise ValueError('Mismatched checkpoint model metadata')
    for name,tensor in model.items():
        expected=current['model'][name]
        _tensor(tensor,expected.shape,expected.dtype)
    _tensor(state['rng_cpu'],session.rng_cpu.shape,torch.uint8)
    if session.rng_device is None:
        if state['rng_device'] is not None:
            raise ValueError('Unexpected checkpoint CUDA RNG')
    else:
        _tensor(state['rng_device'],session.rng_device.shape,torch.uint8)
    optimizer=state['optimizer']
    _fields(optimizer,{'state','param_groups'})
    groups=optimizer['param_groups']
    if type(groups) is not list or len(groups)!=1:
        raise ValueError('Invalid checkpoint optimizer groups')
    template=current['optimizer']['param_groups'][0]
    _fields(groups[0],set(template))
    lr=session.options['learning_rate']*session.scheduler.lr_lambdas[0](updates)
    for key,value in template.items():
        if not _exact(groups[0][key],lr if key=='lr' else value):
            raise ValueError('Mismatched checkpoint optimizer option')
    params=list(session.model.parameters())
    counts=state['optimizer_updates']
    if type(counts) is not list or len(counts)!=len(params):
        raise ValueError('Invalid checkpoint optimizer coverage')
    for count,param in zip(counts,params):
        if type(count) is not int or not 0<=count<=updates or (not param.requires_grad and count):
            raise ValueError('Invalid checkpoint parameter update count')
    if sum(counts)<updates:
        raise ValueError('Incomplete checkpoint optimizer update history')
    if type(optimizer['state']) is not dict or any(type(k) is not int or not 0<=k<len(params) for k in optimizer['state']):
        raise ValueError('Invalid checkpoint optimizer parameter index')
    if updates==0 and optimizer['state']:
        raise ValueError('Unexpected initial optimizer moments')
    if updates>0 and not optimizer['state']:
        raise ValueError('Missing checkpoint optimizer moments')
    if set(optimizer['state'])!={i for i,count in enumerate(counts) if count}:
        raise ValueError('Mismatched checkpoint optimizer coverage')
    for index,moments in optimizer['state'].items():
        _fields(moments,{'step','exp_avg','exp_avg_sq'})
        for key in ('exp_avg','exp_avg_sq'):
            _tensor(moments[key],params[index].shape,torch.float32)
        _tensor(moments['step'],(),torch.float32)
        if numeric:
            step=moments['step'].item()
            # Source AdamW's float32 scalar counter saturates under repeated +1
            # at 2**24; independent integer update history remains exact.
            if step!=min(counts[index],2**24) or (moments['exp_avg_sq']<0).any():
                raise ValueError('Invalid checkpoint optimizer moment/counter')
    scheduler=dict(current['scheduler'])
    scheduler.update(last_epoch=updates,_step_count=updates+1,_last_lr=[lr])
    if not _exact(state['scheduler'],scheduler):
        raise ValueError('Mismatched checkpoint LR scheduler')
    if session.ema is None:
        if state['ema'] is not None:
            raise ValueError('Unexpected checkpoint EMA')
    else:
        ema=state['ema']
        _fields(ema,set(current['ema']))
        for key,value in current['ema'].items():
            if key not in ('shadow_params','optimization_step') and not _exact(ema[key],value):
                raise ValueError('Mismatched checkpoint EMA configuration')
        if type(ema['optimization_step']) is not int or ema['optimization_step']!=updates:
            raise ValueError('Mismatched checkpoint EMA counter')
        trainable=[p for p in params if p.requires_grad]
        if type(ema['shadow_params']) is not list or len(ema['shadow_params'])!=len(trainable):
            raise ValueError('Mismatched checkpoint EMA parameters')
        for shadow,param in zip(ema['shadow_params'],trainable):
            _tensor(shadow,param.shape,torch.float32)
    scaler=state['scaler']
    if session.options['precision']!='fp16':
        if scaler!={}:
            raise ValueError('Unexpected checkpoint gradient scaler')
    else:
        template=session._new_scaler().state_dict()
        _fields(scaler,set(template))
        for key in ('growth_factor','backoff_factor','growth_interval'):
            if not _exact(scaler[key],template[key]):
                raise ValueError('Mismatched checkpoint scaler configuration')
        scale=scaler['scale']
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0<scale<=np.finfo(np.float32).max:
            raise ValueError('Invalid checkpoint scaler scale')
        if type(scaler['_growth_tracker']) is not int or scaler['_growth_tracker']!=updates%scaler['growth_interval']:
            raise ValueError('Mismatched checkpoint scaler counter')
    if numeric:
        _finite(state)
        try:
            torch.Generator(device='cpu').set_state(state['rng_cpu'])
            if state['rng_device'] is not None:
                torch.Generator(device=session.device).set_state(state['rng_device'])
        except RuntimeError as error:
            raise ValueError('Invalid checkpoint RNG state') from error


def _identity(value):
    metadata={k:v for k,v in value.items() if k not in ('tensors','identity_sha256')}
    _json_value(metadata)
    return hashlib.sha256(manifest_bytes(metadata)).hexdigest()


def _tensor_bytes(value):
    if isinstance(value,torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value,dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if type(value) in (list,tuple):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def make_training_checkpoint(session,binding,position,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    if not isinstance(session,TrainingSession):
        raise ValueError('Expected training session')
    _integer(max_workspace_bytes,'max_workspace_bytes',minimum=1)
    with _LOCK:
        _check(cancel)
        _binding(binding,session)
        _position(position,session.batches)
        size=_tensor_bytes(session._raw_state())
        if size*8>max_workspace_bytes or size>MAX_BYTES-MAX_HEADER:
            raise ValueError('Checkpoint tensor state exceeds workspace budget')
        state=session.snapshot(max_state_bytes=max_workspace_bytes)
        _state(state,session,numeric=True)
        tree,tensors=_encode_tree(state)
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        if size*8>max_workspace_bytes or size>MAX_BYTES-MAX_HEADER:
            raise ValueError('Checkpoint tensor state exceeds workspace budget')
        _check(cancel)
        payload=save_tensors(tensors)
        value=dict(schema='unimate.training_checkpoint.v1',upstream_revision=UPSTREAM_REVISION,
            binding=deepcopy_json(binding),position=deepcopy_json(position),runtime=_runtime(session),
            state=tree,tensors=payload,sha256=hashlib.sha256(payload).hexdigest())
        value['identity_sha256']=_identity(value)
        _header(payload,max_workspace_bytes)
        _check(cancel)
        return value


def deepcopy_json(value):
    return _json(manifest_bytes(value))


def validate_training_checkpoint(value,session,binding,*,expected_position=None,cancel=None,
                                 max_workspace_bytes=DEFAULT_WORKSPACE):
    """Preflight and validate all state before returning owned decoded tensors."""
    if not isinstance(session,TrainingSession):
        raise ValueError('Expected training session')
    _integer(max_workspace_bytes,'max_workspace_bytes',minimum=1)
    with _LOCK:
        _check(cancel)
        _fields(value,FIELDS)
        if value['schema']!='unimate.training_checkpoint.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
            raise ValueError('Unsupported training checkpoint schema/source')
        _binding(binding,session)
        _binding(value['binding'],session)
        if not _exact(value['binding'],binding) or not _exact(value['runtime'],_runtime(session)):
            raise ValueError('Mismatched checkpoint binding/runtime')
        _json_value({k:v for k,v in value.items() if k!='tensors'})
        for key in ('sha256','identity_sha256'):
            _digest(value[key],key)
        if _identity(value)!=value['identity_sha256']:
            raise ValueError('Checkpoint identity digest mismatch')
        infos=_header(value['tensors'],max_workspace_bytes)
        if hashlib.sha256(value['tensors']).hexdigest()!=value['sha256']:
            raise ValueError('Checkpoint tensor digest mismatch')
        preflight=_decode_tree(value['state'],infos)
        _state(preflight,session,numeric=False)
        _position(value['position'],preflight['batches'])
        if expected_position is not None and value['position']!=expected_position:
            raise ValueError('Mismatched expected checkpoint data position')
        session._budget(max_workspace_bytes)
        _check(cancel)
        try:
            tensors=load_tensors(value['tensors'])
        except Exception as error:
            raise ValueError('Invalid checkpoint safetensors data') from error
        state=_decode_tree(value['state'],tensors)
        _state(state,session,numeric=True)
        _check(cancel)
        return state,deepcopy_json(value['position'])


def restore_training_checkpoint(session,value,binding,*,expected_position=None,cancel=None,
                                max_workspace_bytes=DEFAULT_WORKSPACE):
    with _LOCK:
        state,position=validate_training_checkpoint(value,session,binding,
            expected_position=expected_position,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
        _check(cancel)
        session.restore(state,max_state_bytes=max_workspace_bytes,cancel=cancel)
        return position
