"""Validated differentiable flow losses over portable source batches."""
import math

import torch

from ._vendor.ema import EMAModel as EMAModel
from ._vendor.flow.transport import Transport,ModelType,PathType,WeightType
from .training_batch_contracts import validate_training_batch
from .training_transforms import _check,_integer


class StableTrainingTransport(Transport):
    """Avoid undefined rotation frames before the unchanged masked source loss."""
    @staticmethod
    def _rotations(value,valid):
        rotations=value[:,:,3:9,:].permute(0,1,3,2)
        with torch.no_grad():
            raw=torch.nan_to_num(rotations)+1e-8
            norm=torch.linalg.norm(raw[...,:3],dim=-1,keepdim=True)
            x=raw[...,:3]/norm.clamp_min(1e-8)
            cross_norm=torch.linalg.norm(torch.cross(x,raw[...,3:],dim=-1),dim=-1)
            bad=(norm.squeeze(-1)<1e-8)|(cross_norm<1e-8)|(~torch.isfinite(norm.squeeze(-1)))|(~torch.isfinite(cross_norm))
            bad=bad|(~valid.squeeze(2))
        identity=rotations.new_tensor([1,0,0,0,1,0])
        safe=torch.where(bad[...,None],identity,rotations).permute(0,1,3,2)
        return torch.cat((value[:,:,:3,:],safe,value[:,:,9:,:]),dim=2)

    def geodesic_loss(self,target,pred,temp_mask,spat_mask,lengths,n_joints):
        valid=temp_mask.bool()&spat_mask.bool().transpose(1,3)
        target=self._rotations(target,valid)
        pred=self._rotations(pred,valid)
        return super().geodesic_loss(target,pred,temp_mask,spat_mask,lengths,n_joints)


def create_flow_schedule(options=None):
    options={} if options is None else options
    if type(options) is not dict or options.keys()-{
        'path_type','prediction','loss_weight','train_eps','sample_eps','lambda_geo','lambda_smooth','geodesic_policy'}:
        raise ValueError('Invalid flow training options')
    path=options.get('path_type','Linear')
    prediction=options.get('prediction','velocity')
    weight=options.get('loss_weight')
    policy=options.get('geodesic_policy','stable')
    if policy not in ('stable','released'):
        raise ValueError('Invalid geodesic policy')
    if path not in ('Linear','GVP','VP') or prediction not in ('velocity','noise','score') or weight not in (None,'velocity','likelihood'):
        raise ValueError('Unsupported flow path/prediction/weight')
    geo,smooth=options.get('lambda_geo',.5),options.get('lambda_smooth',.1)
    for value in (geo,smooth):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid auxiliary loss weight')
    if (geo or smooth) and (path!='Linear' or prediction!='velocity'):
        raise ValueError('Auxiliary flow losses require Linear velocity prediction')
    default_train,default_sample=(1e-5,1e-3) if path=='VP' else ((1e-3,1e-3) if prediction!='velocity' else (0.,0.))
    eps=[]
    for key,default in (('train_eps',default_train),('sample_eps',default_sample)):
        value=options.get(key,default)
        if value is None:
            value=default
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<.5:
            raise ValueError('Invalid flow interval epsilon')
        eps.append(value)
    # Released factory always uses zero eps for stable Linear/GVP velocity.
    if path!='VP' and prediction=='velocity':
        eps=[0.,0.]
    transport=StableTrainingTransport if policy=='stable' else Transport
    return transport(model_type=getattr(ModelType,prediction.upper()),
        path_type={'Linear':PathType.LINEAR,'GVP':PathType.GVP,'VP':PathType.VP}[path],
        loss_type={None:WeightType.NONE,'velocity':WeightType.VELOCITY,'likelihood':WeightType.LIKELIHOOD}[weight],
        train_eps=eps[0],sample_eps=eps[1],lambda_geo=geo,lambda_smooth=smooth)


def flow_training_loss(model,portable_batch,*,options=None,seed=0,cancel=None,
                       max_workspace_bytes=512*1024*1024):
    """Return graph-connected mean loss and metrics without modifying weights.

    The caller owns model mode/device and optimizer updates. Source operations
    consume a scoped CPU/selected-CUDA RNG stream; other device RNGs are untouched.
    """
    _check(cancel)
    _integer(seed,'seed')
    if seed>=2**64:
        raise ValueError('Seed exceeds uint64 range')
    schedule=create_flow_schedule(options)
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
        raise ValueError('Flow training requires positive valid motion lengths')
    motion=motion.to(device)
    cond={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in cond.items()}
    devices=[device.index] if device.type=='cuda' else []
    _check(cancel)
    with torch.random.fork_rng(devices=devices),torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        terms=schedule.training_losses(model,motion,dict(cond=cond))
    _check(cancel)
    if any(not torch.isfinite(term).all() for term in terms.values()):
        raise ValueError('Nonfinite flow training loss')
    means={key:value.mean() for key,value in terms.items()}
    if any(not torch.isfinite(value) for value in means.values()):
        raise ValueError('Nonfinite reduced flow training loss')
    return means['loss'],{key:value.item() for key,value in means.items()}
