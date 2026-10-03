"""Normalized inference from recorded training schedules; live models stay private."""
import math

import torch
from torchdiffeq import odeint

from .inference import _LOCK
from .training_job import validate_training_job_metadata
from .training_loss import create_flow_schedule
from .training_diffusion import create_diffusion_schedule
from ._vendor.flow.transport import Sampler
from .training_transforms import _check,_integer


def sampling_options(paradigm,options=None):
    options={} if options is None else options
    if type(options) is not dict:
        raise ValueError('Expected sampling options object')
    if paradigm=='flow':
        allowed={'method','num_steps','atol','rtol'}
        result=dict(method='dopri5',num_steps=50,atol=1e-6,rtol=1e-3)
        result.update(options)
        # Use the installed integrator's registry, rather than treating an
        # arbitrary import/function name as a solver.
        if type(result['method']) is not str or result['method'] not in odeint.__globals__['SOLVERS']:
            raise ValueError('Unknown ODE sampling method')
        _integer(result['num_steps'],'num_steps',2)
        if result['num_steps']>100000:
            raise ValueError('Excess ODE sampling steps')
        for name in ('atol','rtol'):
            value=result[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:
                raise ValueError('Invalid sampling tolerance')
    elif paradigm=='diffusion':
        allowed={'method','clip_denoised','eta'}
        result=dict(method='ancestral',clip_denoised=False)
        result.update(options)
        if result['method'] not in ('ancestral','ddim') or type(result['clip_denoised']) is not bool:
            raise ValueError('Invalid diffusion sampling options')
        if result['method']=='ddim':
            result.setdefault('eta',0.)
            eta=result['eta']
            if type(eta) not in (int,float) or not math.isfinite(eta) or not 0<=eta<=1:
                raise ValueError('Invalid DDIM eta')
        elif 'eta' in options:
            raise ValueError('Eta requires DDIM sampling')
    else:
        raise ValueError('Unknown sampling paradigm')
    if options.keys()-allowed:
        raise ValueError('Unknown sampling option')
    return result


def _finite(value,shape,device):
    if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shape)
        or value.dtype!=torch.float32 or value.device!=device
        or not torch.isfinite(value).all()):
        raise ValueError('Invalid/nonfinite float32 sampling tensor')


def _preflight(model,cond,job,seed,guidance,options,workspace,cancel):
    _check(cancel)
    _integer(seed,'seed')
    if seed>=2**64:
        raise ValueError('Seed exceeds uint64')
    _integer(workspace,'max_workspace_bytes',1)
    if type(guidance) not in (int,float) or not math.isfinite(guidance) or not 1<=guidance<=10:
        raise ValueError('Guidance must be between one and ten')
    validate_training_job_metadata(job)
    options=sampling_options(job['paradigm'],options)
    if not isinstance(model,torch.nn.Module) or model.training:
        raise ValueError('Expected private eval model')
    params=list(model.parameters())
    if not params or any(p.requires_grad or p.dtype!=torch.float32 or p.device!=params[0].device for p in params):
        raise ValueError('Expected frozen uniform float32 model parameters')
    device=params[0].device
    if device.type not in ('cpu','cuda'):
        raise ValueError('Sampling requires caller-selected CPU or CUDA device')
    if guidance>1 and job['model']['cond_mask_prob']<=0:
        raise ValueError('Guidance requires training with conditional dropout')
    if type(cond) is not dict or not {'mean','std','lengths_mask','motion_length'}<=cond.keys():
        raise ValueError('Missing sampling conditioning fields')
    mean=cond['mean']
    if not isinstance(mean,torch.Tensor) or mean.ndim!=3 or not 1<=mean.shape[0]<=4096:
        raise ValueError('Invalid sampling batch')
    batch,joints,_=mean.shape
    frames=job['model']['max_motion_length']
    if joints!=job['model']['max_joints'] or mean.shape[2]!=12:
        raise ValueError('Sampling capacity does not match trained model')
    mask=cond['lengths_mask']
    if (not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.shape!=(batch,1,1,frames)
        or mask.device!=device or not isinstance(cond['motion_length'],torch.Tensor)
        or cond['motion_length'].shape!=(batch,) or cond['motion_length'].dtype!=torch.int64
        or cond['motion_length'].device!=device or not ((cond['motion_length']>0)&(cond['motion_length']<=frames)).all()):
        raise ValueError('Invalid sampling lengths/mask')
    shape=(batch,joints,12,frames)
    saved=options['num_steps'] if job['paradigm']=='flow' else 1
    condition_bytes=sum(v.numel()*v.element_size() for v in cond.values() if isinstance(v,torch.Tensor))
    estimate=64*1024**2+condition_bytes*8+math.prod(shape)*4*(32+saved*2)
    if estimate>workspace:
        raise ValueError('Sampling exceeds workspace budget')
    _finite(mean,(batch,joints,12),device)
    _finite(cond['std'],mean.shape,device)
    if not (cond['std']>0).all():
        raise ValueError('Expected positive conditioning standard deviations')
    for value in cond.values():
        if isinstance(value,torch.Tensor) and value.is_floating_point():
            _finite(value,value.shape,device)
    _check(cancel)
    return shape,device,options


def sample_trained_model(model,cond,job,*,seed,guidance=1.,options=None,cancel=None,
    progress=None,max_workspace_bytes=512*1024**2):
    """Return normalized (B,J,12,T) features, without modifying model weights.

    CFG=1 forces unconditional inference, matching the standalone source entry.
    Recorded job loss options reconstruct the path/parameterization or diffusion
    schedule. Workspace accounts for solver tensors, not model activations.
    """
    shape,device,options=_preflight(model,cond,job,seed,guidance,options,max_workspace_bytes,cancel)
    variance=job['paradigm']=='diffusion' and job['loss'].get('learn_sigma',False)
    output_shape=(shape[0],shape[1]*2,shape[2],shape[3]) if variance else shape
    def checked(x,t,cond=None):
        _check(cancel)
        if guidance==1:
            output=model(x,t,cond,force_mask=True)
            _finite(output,output_shape,device)
        else:
            conditional=model(x,t,cond)
            _finite(conditional,output_shape,device)
            _check(cancel)
            unconditional=model(x,t,cond,force_mask=True)
            _finite(unconditional,output_shape,device)
            output=unconditional+guidance*(conditional-unconditional)
            _finite(output,output_shape,device)
        _check(cancel)
        return output
    devices=[device.index] if device.type=='cuda' else []
    with _LOCK,torch.random.fork_rng(devices=devices),torch.device('cpu'),torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        noise=torch.randn(shape,device=device,dtype=torch.float32)
        _check(cancel)
        if job['paradigm']=='flow':
            transport=create_flow_schedule(job['loss'])
            sampler=Sampler(transport).sample_ode(sampling_method=options['method'],num_steps=options['num_steps'],
                atol=options['atol'],rtol=options['rtol'])
            def flow_model(x,t,cond=None):
                output=checked(x,t,cond)
                if progress:
                    progress(min(99,max(0,int(float(t[0])*100))))
                return output
            result=sampler(noise,flow_model,cond=cond)[-1]
        else:
            schedule=create_diffusion_schedule(job['loss'])
            loop=schedule.p_sample_loop_progressive if options['method']=='ancestral' else schedule.ddim_sample_loop_progressive
            extra=dict(eta=options['eta']) if options['method']=='ddim' else {}
            result=None
            for index,item in enumerate(loop(checked,shape,noise=noise,device=device,
                clip_denoised=options['clip_denoised'],model_kwargs=dict(cond=cond),**extra)):
                _check(cancel)
                _finite(item['sample'],shape,device)
                result=item['sample']
                if progress:
                    progress(int((index+1)*100/schedule.num_timesteps))
        _check(cancel)
        _finite(result,shape,device)
        if progress:
            progress(100)
        _check(cancel)
        # The final ODE slice otherwise retains the complete trajectory.
        return result.clone()
