"""Validated diffusion training over portable source batches."""
import math

import numpy as np
import torch

from ._vendor.diffusion import gaussian_diffusion as gd
from ._vendor.diffusion.respace import SpacedDiffusion,space_timesteps
from .training_batch_contracts import validate_training_batch
from .training_loss import StableTrainingTransport
from .training_transforms import _check,_integer


class TrainingDiffusion(SpacedDiffusion):
    """Explicit source corrections; pinned sampling and loss math remain intact."""
    def geodesic_loss(self,target,pred,temp_mask,spat_mask,lengths,n_joints):
        if self.geodesic_policy=='stable':
            valid=temp_mask.bool()&spat_mask.bool().transpose(1,3)
            target=StableTrainingTransport._rotations(target,valid)
            pred=StableTrainingTransport._rotations(pred,valid)
        return super().geodesic_loss(target,pred,temp_mask,spat_mask,lengths,n_joints)

    def training_losses(self,model,x_start,t,model_kwargs=None,noise=None):
        if self.timestep_policy=='mapped':
            model=self._wrap_model(model)
        terms=super().training_losses(model,x_start,t,model_kwargs,noise)
        if self.variance_policy=='trained' and 'vb' in terms:
            terms['loss']=terms['loss']+terms['vb']
        return terms


def create_diffusion_schedule(options=None):
    options={} if options is None else options
    keys={'diffusion_steps','timestep_respacing','rescale_timesteps','noise_schedule',
        'scale_beta','predict_xstart','learn_sigma','sigma_small','lambda_geo',
        'geodesic_policy','timestep_policy','variance_policy'}
    if type(options) is not dict or options.keys()-keys:
        raise ValueError('Invalid diffusion training options')
    steps=options.get('diffusion_steps',100)
    if type(steps) is not int or not 2<=steps<=100000:
        raise ValueError('Diffusion steps must be between 2 and 100000')
    for key in ('rescale_timesteps','predict_xstart','learn_sigma','sigma_small'):
        if key in options and type(options[key]) is not bool:
            raise ValueError('Expected boolean diffusion option')
    name=options.get('noise_schedule','cosine')
    if name not in ('linear','cosine'):
        raise ValueError('Unsupported diffusion noise schedule')
    scale=options.get('scale_beta',1.)
    geo=options.get('lambda_geo',.5)
    for value,label,positive in ((scale,'scale_beta',True),(geo,'lambda_geo',False)):
        if type(value) not in (int,float) or not math.isfinite(value) or (value<=0 if positive else value<0):
            raise ValueError('Invalid '+label)
    policies={key:options.get(key,default) for key,default in (
        ('geodesic_policy','stable'),('timestep_policy','mapped'),('variance_policy','trained'))}
    for key,choices in (('geodesic_policy',('stable','released')),
        ('timestep_policy',('mapped','released')),('variance_policy',('trained','released'))):
        if policies[key] not in choices:
            raise ValueError('Invalid '+key)
    respacing=options.get('timestep_respacing','')
    if type(respacing) is str:
        if len(respacing)>1024:
            raise ValueError('Invalid diffusion timestep respacing')
        if respacing:
            counts=respacing[4:] if respacing.startswith('ddim') else respacing
            if any(not part.isascii() or not part.isdecimal() or int(part)<1 for part in counts.split(',')):
                raise ValueError('Invalid diffusion timestep respacing')
    elif type(respacing) is list:
        if not respacing or len(respacing)>steps or any(type(v) is not int or v<1 for v in respacing):
            raise ValueError('Invalid diffusion timestep respacing')
    else:
        raise ValueError('Invalid diffusion timestep respacing')
    betas=gd.get_named_beta_schedule(name,steps,scale)
    if not np.isfinite(betas).all() or not ((betas>0)&(betas<1)).all():
        raise ValueError('Diffusion beta schedule must lie strictly between zero and one')
    retained=space_timesteps(steps,respacing or [steps])
    if len(retained)<2:
        raise ValueError('Diffusion requires at least two retained timesteps')
    cumulative=np.cumprod(1-betas)
    if (cumulative<1/np.finfo(np.float32).max).any():
        raise ValueError('Diffusion cumulative schedule exceeds float32 reciprocal range')
    selected=cumulative[sorted(retained)]
    effective=1-selected/np.concatenate(([1.],selected[:-1]))
    if not ((effective>0)&(effective<1)).all():
        raise ValueError('Respaced diffusion beta schedule is not representable')
    schedule=TrainingDiffusion(use_timesteps=retained,betas=betas,
        model_mean_type=gd.ModelMeanType.START_X if options.get('predict_xstart',True) else gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE if options.get('learn_sigma',False) else (
            gd.ModelVarType.FIXED_SMALL if options.get('sigma_small',True) else gd.ModelVarType.FIXED_LARGE),
        loss_type=gd.LossType.MSE,rescale_timesteps=options.get('rescale_timesteps',False),lambda_geo=geo)
    for key,value in policies.items():
        setattr(schedule,key,value)
    return schedule


def diffusion_training_loss(model,portable_batch,*,options=None,seed=0,cancel=None,
                            max_workspace_bytes=512*1024*1024):
    """Return differentiable mean loss and metrics; caller owns optimizer/device."""
    _check(cancel)
    _integer(seed,'seed')
    if seed>=2**64:
        raise ValueError('Seed exceeds uint64 range')
    schedule=create_diffusion_schedule(options)
    if not isinstance(model,torch.nn.Module) or not model.training:
        raise ValueError('Expected model in training mode')
    params=list(model.parameters())
    if not any(p.requires_grad for p in params):
        raise ValueError('Training kernel requires trainable parameters')
    if not params or any(p.dtype!=torch.float32 or p.device!=params[0].device for p in params):
        raise ValueError('Training kernel requires uniform float32 parameters')
    device=params[0].device
    if device.type not in ('cpu','cuda'):
        raise ValueError('Training device must be CPU or CUDA selected by caller')
    motion,cond=validate_training_batch(portable_batch,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    if (cond['motion_length']<=0).any():
        raise ValueError('Diffusion training requires positive valid motion lengths')
    motion=motion.to(device)
    cond={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in cond.items()}
    devices=[device.index] if device.type=='cuda' else []
    _check(cancel)
    with torch.random.fork_rng(devices=devices),torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        t=torch.randint(schedule.num_timesteps,(len(motion),),device=device)
        terms=schedule.training_losses(model,motion,t,dict(cond=cond))
    _check(cancel)
    if any(not torch.isfinite(term).all() for term in terms.values()):
        raise ValueError('Nonfinite diffusion training loss')
    means={key:value.mean() for key,value in terms.items()}
    if any(not torch.isfinite(value) for value in means.values()):
        raise ValueError('Nonfinite reduced diffusion training loss')
    return means['loss'],{key:value.item() for key,value in means.items()}
