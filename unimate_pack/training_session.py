"""Private optimizer sessions; portable checkpoint validation is a separate layer."""
from copy import deepcopy
import math

import torch
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from .inference import _LOCK
from .training_batch_contracts import validate_training_batch
from .training_diffusion import create_diffusion_schedule
from .training_loss import create_flow_schedule,EMAModel
from .training_transforms import _check,_integer

DEFAULTS=dict(learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.99,
    max_grad_norm=None,gradient_accumulation_steps=1,warmup_ratio=.02,
    min_lr_ratio=.01,num_steps=100000,use_ema=True,ema_decay=.9999,precision='none')
MAX_STATE_BYTES=2*1024**3


def session_options(options=None):
    if options is None:
        options={}
    if type(options) is not dict or options.keys()-DEFAULTS.keys():
        raise ValueError('Invalid optimizer session options')
    result={**DEFAULTS,**options}
    for key in ('num_steps','gradient_accumulation_steps'):
        _integer(result[key],key,minimum=1)
        if result[key]>100000000:
            raise ValueError('Excessive '+key)
    for key in ('learning_rate','weight_decay','adam_beta1','adam_beta2','warmup_ratio','min_lr_ratio','ema_decay'):
        value=result[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid '+key)
    if result['learning_rate']==0 or any(result[k]>=1 for k in ('adam_beta1','adam_beta2','warmup_ratio')):
        raise ValueError('Invalid learning rate, beta or warmup ratio')
    if result['min_lr_ratio']>1 or result['ema_decay']>1:
        raise ValueError('Invalid minimum LR or EMA decay')
    clip=result['max_grad_norm']
    if clip is not None and (type(clip) not in (int,float) or not math.isfinite(clip) or clip<=0):
        raise ValueError('Invalid gradient clipping norm')
    if type(result['use_ema']) is not bool or result['precision'] not in ('none','bf16','fp16'):
        raise ValueError('Invalid EMA or precision option')
    return result


def _copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):
        result=type(value)((key,_copy(item)) for key,item in value.items())
        if hasattr(value,'_metadata'):
            result._metadata=_copy(value._metadata)
        return result
    if type(value) in (tuple,list):
        return type(value)(_copy(item) for item in value)
    return deepcopy(value)


def _finite(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite session state')
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite session state')
    elif isinstance(value,dict):
        for item in value.values():
            _finite(item)
    elif type(value) in (list,tuple):
        for item in value:
            _finite(item)


class TrainingSession:
    """Single-process transactional accumulation over caller-owned models.

    Snapshots are internal trusted tensor trees, not an untrusted file format.
    Caller owns device selection and ComfyUI model residency. CPU backup budget
    conservatively reserves eight copies of persistent model state for rollback.
    """
    def __init__(self,model,*,paradigm='flow',loss_options=None,options=None,seed=0):
        _integer(seed,'seed')
        if seed>=2**64:
            raise ValueError('Seed exceeds uint64 range')
        self.options=session_options(options)
        self.loss_options=deepcopy({} if loss_options is None else loss_options)
        if paradigm not in ('flow','diffusion'):
            raise ValueError('Unsupported training paradigm')
        self.paradigm=paradigm
        self.schedule=(create_flow_schedule if paradigm=='flow' else create_diffusion_schedule)(self.loss_options)
        if not isinstance(model,torch.nn.Module) or not model.training:
            raise ValueError('Expected model in training mode')
        params=list(model.parameters())
        if not params or not any(p.requires_grad for p in params):
            raise ValueError('Session requires trainable parameters')
        self.device=params[0].device
        if self.device.type not in ('cpu','cuda') or any(p.device!=self.device or p.dtype!=torch.float32 for p in params):
            raise ValueError('Session requires uniform float32 CPU or caller-selected CUDA parameters')
        if self.options['precision']=='fp16' and self.device.type!='cuda':
            raise ValueError('FP16 scaling requires caller-selected CUDA')
        if any(p.grad is not None for p in params):
            raise ValueError('Session requires cleared gradients')
        _finite(model.state_dict())
        self.model=model
        cfg=self.options
        self.optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],
            betas=(cfg['adam_beta1'],cfg['adam_beta2']))
        self.scheduler=get_cosine_with_min_lr_schedule_with_warmup(self.optimizer,
            num_warmup_steps=int(cfg['warmup_ratio']*cfg['num_steps']),
            num_training_steps=cfg['num_steps'],min_lr_rate=cfg['min_lr_ratio'],num_cycles=.5)
        self.ema=EMAModel(params,decay=cfg['ema_decay'],use_ema_warmup=True) if cfg['use_ema'] else None
        self.scaler=self._new_scaler()
        self.rng_cpu=torch.Generator(device='cpu').manual_seed(seed).get_state()
        self.rng_device=torch.Generator(device=self.device).manual_seed(seed).get_state() if self.device.type=='cuda' else None
        self.updates=0
        self.batches=0
        self.optimizer_updates=[0]*len(params)

    def _new_scaler(self):
        return torch.amp.GradScaler('cuda',enabled=self.options['precision']=='fp16')

    def _budget(self,limit):
        _integer(limit,'max_state_bytes',minimum=1)
        count=sum(t.numel()*t.element_size() for t in self.model.state_dict().values())
        needed=count*8+self.rng_cpu.numel()*2+(self.rng_device.numel()*2 if self.rng_device is not None else 0)
        if needed>limit:
            raise ValueError('Session rollback state exceeds budget')

    def _raw_state(self):
        return dict(paradigm=self.paradigm,loss_options=self.loss_options,options=self.options,
            device=str(self.device),model=self.model.state_dict(),
            trainable={name:p.requires_grad for name,p in self.model.named_parameters()},
            modes={name:m.training for name,m in self.model.named_modules()},
            optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
            ema=self.ema.state_dict() if self.ema else None,scaler=self.scaler.state_dict(),
            rng_cpu=self.rng_cpu,rng_device=self.rng_device,updates=self.updates,batches=self.batches,
            optimizer_updates=self.optimizer_updates)

    def snapshot(self,*,max_state_bytes=MAX_STATE_BYTES):
        with _LOCK:
            self._budget(max_state_bytes)
            return _copy(self._raw_state())

    def _apply(self,state):
        self.model.load_state_dict(state['model'],strict=True)
        modules=dict(self.model.named_modules())
        for name,mode in state['modes'].items():
            modules[name].training=mode
        self.optimizer.load_state_dict(_copy(state['optimizer']))
        self.scheduler.load_state_dict(_copy(state['scheduler']))
        if self.ema:
            self.ema.load_state_dict(_copy(state['ema']))
            self.ema.to(self.device,dtype=torch.float32)
            self.ema.cur_decay_value=self.ema.get_decay(self.ema.optimization_step) if self.ema.optimization_step else None
        self.scaler=self._new_scaler()
        if state['scaler']:
            self.scaler.load_state_dict(_copy(state['scaler']))
        self.rng_cpu=state['rng_cpu'].clone()
        self.rng_device=state['rng_device'].clone() if state['rng_device'] is not None else None
        self.updates=state['updates']
        self.batches=state['batches']
        self.optimizer_updates=list(state['optimizer_updates'])
        self.optimizer.zero_grad(set_to_none=True)

    def restore(self,state,*,max_state_bytes=MAX_STATE_BYTES,cancel=None):
        """Restore an internally generated snapshot, with rollback on load errors."""
        with _LOCK:
            _check(cancel)
            self._budget(max_state_bytes)
            current=self._raw_state()
            if type(state) is not dict or state.keys()!=current.keys():
                raise ValueError('Invalid internal session snapshot')
            for key in ('paradigm','options','loss_options','device','trainable'):
                if state[key]!=current[key]:
                    raise ValueError('Mismatched session '+key)
            if state['model'].keys()!=current['model'].keys():
                raise ValueError('Mismatched model state')
            for name,tensor in state['model'].items():
                expected=current['model'][name]
                if not isinstance(tensor,torch.Tensor) or tensor.shape!=expected.shape or tensor.dtype!=expected.dtype:
                    raise ValueError('Mismatched model tensor')
            _finite(state)
            before=_copy(current)
            try:
                _check(cancel)
                self._apply(state)
                _check(cancel)
            except BaseException:
                self._apply(before)
                raise

    def step(self,batches,*,final_group=False,cancel=None,max_state_bytes=MAX_STATE_BYTES,
             max_workspace_bytes=512*1024*1024):
        with _LOCK:
            _check(cancel)
            if type(final_group) is not bool or type(batches) is not list:
                raise ValueError('Expected explicit accumulation group')
            accumulation=self.options['gradient_accumulation_steps']
            if not 1<=len(batches)<=accumulation or (len(batches)!=accumulation and not final_group):
                raise ValueError('Incomplete or excessive accumulation group')
            if self.updates>=self.options['num_steps']:
                raise ValueError('Training step limit reached')
            if not self.model.training or any(p.grad is not None for p in self.model.parameters()):
                raise ValueError('Session requires training mode and cleared gradients')
            self._budget(max_state_bytes)
            before=self.snapshot(max_state_bytes=max_state_bytes)
            devices=[self.device.index] if self.device.type=='cuda' else []
            metrics={}
            try:
                with torch.random.fork_rng(devices=devices),torch.device('cpu'):
                    torch.set_rng_state(self.rng_cpu)
                    if devices:
                        torch.cuda.set_rng_state(self.rng_device,self.device)
                    for portable in batches:
                        _check(cancel)
                        motion,cond=validate_training_batch(portable,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
                        if (cond['motion_length']<=0).any():
                            raise ValueError('Training requires positive valid lengths')
                        motion=motion.to(self.device)
                        cond={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in cond.items()}
                        dtype=torch.bfloat16 if self.options['precision']=='bf16' else torch.float16
                        with torch.autocast(self.device.type,dtype=dtype,enabled=self.options['precision']!='none'):
                            if self.paradigm=='flow':
                                terms=self.schedule.training_losses(self.model,motion,dict(cond=cond))
                            else:
                                t=torch.randint(self.schedule.num_timesteps,(len(motion),),device=self.device)
                                terms=self.schedule.training_losses(self.model,motion,t,dict(cond=cond))
                            _finite(terms)
                            means={key:value.mean() for key,value in terms.items()}
                            _finite(means)
                            loss=means['loss']/accumulation
                        _check(cancel)
                        self.scaler.scale(loss).backward()
                        for key,value in means.items():
                            metrics[key]=metrics.get(key,0.)+value.item()/len(batches)
                    self.scaler.unscale_(self.optimizer)
                    active=[i for i,p in enumerate(self.model.parameters()) if p.grad is not None]
                    gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
                    if not gradients:
                        raise ValueError('Training produced no gradients')
                    _finite(gradients)
                    clip=self.options['max_grad_norm']
                    if clip is not None:
                        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),clip,error_if_nonfinite=True)
                        metrics['grad_norm']=norm.item()
                    _check(cancel)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema:
                        self.ema.step(self.model.parameters())
                    _finite(self._raw_state())
                    _check(cancel)
                    self.rng_cpu=torch.get_rng_state().clone()
                    if devices:
                        self.rng_device=torch.cuda.get_rng_state(self.device).clone()
                    self.updates+=1
                    self.batches+=len(batches)
                    for index in active:
                        self.optimizer_updates[index]+=1
                    metrics.update(lr=self.scheduler.get_last_lr()[0],updates=self.updates,batches=self.batches)
                return metrics
            except BaseException:
                self._apply(before)
                raise
