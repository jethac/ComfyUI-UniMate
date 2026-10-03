"""Optimizer order, scoped streams, atomic groups and exact session resume."""
from copy import deepcopy
import math

import pytest
import torch
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from test_training_loss import Predictor,batch
from unimate_pack.training_batch_contracts import validate_training_batch
from unimate_pack.training_loss import create_flow_schedule,EMAModel
from unimate_pack import training_session as sessions


def options(**kwargs):
    return dict(num_steps=8,warmup_ratio=.25,max_grad_norm=.1,**kwargs)


def make(model=None,**kwargs):
    return sessions.TrainingSession(model or Predictor(),paradigm='flow',
        loss_options=dict(lambda_geo=0.,lambda_smooth=0.),options=options(**kwargs),seed=17)


def assert_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            assert_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_tree(x,y)
    else:
        assert a==b


@pytest.mark.parametrize('accumulation',[1,2,3])
def test_updates_match_released_order(accumulation):
    session=make(gradient_accumulation_steps=accumulation)
    reference=Predictor()
    optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.99))
    scheduler=get_cosine_with_min_lr_schedule_with_warmup(optimizer,
        num_warmup_steps=2,num_training_steps=8,min_lr_rate=.01,num_cycles=.5)
    ema=EMAModel(reference.parameters(),decay=.9999,use_ema_warmup=True)
    schedule=create_flow_schedule(dict(lambda_geo=0.,lambda_smooth=0.))
    motion,cond=validate_training_batch(batch())
    process_state=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        for _ in range(4):
            for _ in range(accumulation):
                terms=schedule.training_losses(reference,motion,dict(cond=cond))
                (terms['loss'].mean()/accumulation).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),.1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.step(reference.parameters())
    for _ in range(4):
        metrics=session.step([batch()]*accumulation)
        assert math.isfinite(metrics['loss'])
    assert torch.equal(process_state,torch.get_rng_state())
    assert_tree(reference.state_dict(),session.model.state_dict())
    assert_tree(optimizer.state_dict(),session.optimizer.state_dict())
    assert_tree(scheduler.state_dict(),session.scheduler.state_dict())
    assert_tree(ema.state_dict(),session.ema.state_dict())
    assert session.updates==4 and session.batches==4*accumulation


@pytest.mark.parametrize('paradigm',['flow','diffusion'])
def test_resume_matches_uninterrupted(paradigm):
    loss_options=dict(lambda_geo=0.)
    if paradigm=='flow':
        loss_options['lambda_smooth']=0.
    def new():
        return sessions.TrainingSession(Predictor(),paradigm=paradigm,loss_options=loss_options,
            options=options(gradient_accumulation_steps=2),seed=17)
    uninterrupted=new()
    for _ in range(5):
        uninterrupted.step([batch(),batch()])
    interrupted=new()
    for _ in range(2):
        interrupted.step([batch(),batch()])
    state=interrupted.snapshot()
    resumed=new()
    resumed.restore(state)
    for _ in range(3):
        resumed.step([batch(),batch()])
    assert_tree(uninterrupted.snapshot(),resumed.snapshot())
    assert_tree(state,interrupted.snapshot())


def test_cancel_after_update_rolls_back_complete_group(monkeypatch):
    session=make()
    session.step([batch()])
    initial=session.snapshot()
    original=session.optimizer.step
    cancelled=False
    def update(*args,**kwargs):
        nonlocal cancelled
        result=original(*args,**kwargs)
        cancelled=True
        return result
    monkeypatch.setattr(session.optimizer,'step',update)
    def cancel():
        if cancelled:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        session.step([batch()],cancel=cancel)
    assert_tree(initial,session.snapshot())
    assert all(p.grad is None for p in session.model.parameters())


def test_second_microbatch_failure_rolls_back_rng_buffers_and_gradients():
    class Failing(Predictor):
        def __init__(self):
            super().__init__()
            self.register_buffer('calls',torch.tensor(0))
        def forward(self,*args,**kwargs):
            self.calls.add_(1)
            if self.calls==2:
                raise RuntimeError('microbatch failed')
            return super().forward(*args,**kwargs)
    session=make(Failing(),gradient_accumulation_steps=2)
    initial=session.snapshot()
    with pytest.raises(RuntimeError,match='microbatch failed'):
        session.step([batch(),batch()])
    assert_tree(initial,session.snapshot())
    assert session.model.weight.grad is None


def test_partial_group_is_explicit_and_uses_configured_divisor():
    session=make(gradient_accumulation_steps=2)
    with pytest.raises(ValueError,match='accumulation'):
        session.step([batch()])
    session.step([batch()],final_group=True)
    assert session.batches==1 and session.updates==1


@pytest.mark.parametrize('change',[
    {'learning_rate':0},{'adam_beta1':1},{'weight_decay':-1},
    {'warmup_ratio':1},{'min_lr_ratio':2},{'num_steps':True},
    {'gradient_accumulation_steps':0},{'precision':'bad'},
    {'use_ema':1},{'ema_decay':float('nan')},{'max_grad_norm':0}])
def test_invalid_options(change):
    with pytest.raises(ValueError):
        sessions.TrainingSession(Predictor(),options={**options(),**change})


def test_snapshot_budget_before_copy_and_update():
    session=make()
    with pytest.raises(ValueError,match='budget'):
        session.step([batch()],max_state_bytes=1)
    assert session.updates==0 and session.optimizer.state=={}


def test_corrupt_internal_snapshot_rejected_without_mutation():
    session=make()
    session.step([batch()])
    initial=session.snapshot()
    bad=deepcopy(initial)
    bad['model']['weight']=torch.tensor(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):
        session.restore(bad)
    assert_tree(initial,session.snapshot())


def test_cpu_bfloat16_session():
    session=make(precision='bf16')
    metrics=session.step([batch()])
    assert math.isfinite(metrics['loss'])
    assert session.updates==1


def test_nonfinite_optimizer_update_rolls_back(monkeypatch):
    session=make()
    initial=session.snapshot()
    original=session.optimizer.step
    def update(*args,**kwargs):
        result=original(*args,**kwargs)
        with torch.no_grad():
            session.model.weight.fill_(float('nan'))
        return result
    monkeypatch.setattr(session.optimizer,'step',update)
    with pytest.raises(ValueError,match='Nonfinite'):
        session.step([batch()])
    assert_tree(initial,session.snapshot())


def test_ema_overflow_rolls_back_complete_state(monkeypatch):
    session=make()
    with torch.no_grad():
        session.model.weight.fill_(3e38)
        session.ema.shadow_params[0].fill_(3e38)
    monkeypatch.setattr(session.schedule,'training_losses',
        lambda model,motion,cond:{'loss':model.weight*0})
    initial=session.snapshot()
    original=session.optimizer.step
    def update(*args,**kwargs):
        result=original(*args,**kwargs)
        with torch.no_grad():
            session.model.weight.fill_(-3e38)
        return result
    monkeypatch.setattr(session.optimizer,'step',update)
    with pytest.raises(ValueError,match='Nonfinite'):
        session.step([batch()])
    assert_tree(initial,session.snapshot())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['none','bf16','fp16'])
def test_cuda_precision_resume_and_global_rng(precision):
    device=torch.device('cuda',0)
    def new():
        return make(Predictor().to(device),precision=precision)
    cpu=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state(device).clone()
    uninterrupted=new()
    uninterrupted.step([batch()])
    state=uninterrupted.snapshot()
    assert all(t.device.type=='cpu' for t in state['model'].values())
    uninterrupted.step([batch()])
    resumed=new()
    resumed.restore(state)
    resumed.step([batch()])
    assert_tree(uninterrupted.snapshot(),resumed.snapshot())
    assert torch.equal(cpu,torch.get_rng_state())
    assert torch.equal(cuda,torch.cuda.get_rng_state(device))


@pytest.mark.parametrize('name',['UniMateGraphAdaLN','UniMateGraphCrossAttn','UniMateFullAdaLN','UniMateFullCrossAttn'])
def test_backbone_dropout_stream_resume(name):
    from unimate_pack._vendor import denoiser
    def new():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model=getattr(denoiser,name)(feature_len=12,max_motion_length=12,max_joints=16,max_depth=32,
                latent_dim=64,ff_size=128,num_layers=1,num_heads=4,dropout=.1,cond_mask_prob=.2,
                cond_mode='text',text_dim=7,use_joint_name_emb=True,use_depth_emb=True,
                concat_parent_features=True,num_tpos_queries=2,inject_tpos_to_adaln=True)
        return make(model)
    uninterrupted=new()
    uninterrupted.step([batch()])
    state=uninterrupted.snapshot()
    uninterrupted.step([batch()])
    resumed=new()
    resumed.restore(state)
    resumed.step([batch()])
    assert_tree(uninterrupted.snapshot(),resumed.snapshot())


@pytest.mark.parametrize('device,precision',[
    ('cpu','none'),('cpu','bf16'),('cuda:0','bf16'),('cuda:0','fp16')])
def test_autocast_linear_dtype_and_resume(device,precision):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(12,12)
            self.output_dtype=None
        def forward(self,x,t,cond):
            output=self.linear(x.permute(0,1,3,2))
            self.output_dtype=output.dtype
            return output.permute(0,1,3,2)
    def new():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model=LinearModel().to(device)
        return make(model,precision=precision)
    session=new()
    session.step([batch()])
    expected={'none':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[precision]
    assert session.model.output_dtype==expected
    state=session.snapshot()
    session.step([batch()])
    resumed=new()
    resumed.restore(state)
    resumed.step([batch()])
    assert_tree(session.snapshot(),resumed.snapshot())
