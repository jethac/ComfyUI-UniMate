"""Diffusion training losses compared with the pinned released implementation."""
import ast
from copy import deepcopy
import enum
import hashlib
import math
import os
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from test_training_loss import Predictor,batch
from unimate_pack.training_batch_contracts import validate_training_batch
from unimate_pack import training_diffusion as kernel
from unimate_pack._vendor import training_math


@pytest.fixture(scope='module')
def reference():
    root=os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned source required')
    root=Path(root)/'unimate/models/diffusion'
    hashes={
        'losses.py':'21f29634e38f6e647248075d4f61e3beb5e7a6f287db1936af1a99a8c3227607',
        'gaussian_diffusion.py':'98be2fd835926c1bde21017cf72b707f641230c1066315938bcd98b0b579b629',
        'respace.py':'16c2d1a477d9859d4e3ea24b57e0601c0669989e9dda49530086224a587cb13e'}
    ctx=dict(np=np,th=torch,torch=torch,enum=enum,math=math,deepcopy=deepcopy,
        mean_flat=training_math.mean_flat,sum_flat=training_math.sum_flat,
        geodesic_distance=training_math.geodesic_distance,
        rotation_6d_to_matrix_safe=training_math.rotation_6d_to_matrix_safe)
    for name,digest in hashes.items():
        source=(root/name).read_bytes()
        assert hashlib.sha256(source).hexdigest()==digest
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        exec(compile(tree,str(root/name),'exec'),ctx)
    return types.SimpleNamespace(**ctx)


def source_schedule(ref,options):
    steps=options.get('diffusion_steps',100)
    return ref.SpacedDiffusion(use_timesteps=ref.space_timesteps(steps,options.get('timestep_respacing') or [steps]),
        betas=ref.get_named_beta_schedule(options.get('noise_schedule','cosine'),steps,options.get('scale_beta',1.)),
        model_mean_type=ref.ModelMeanType.START_X if options.get('predict_xstart',True) else ref.ModelMeanType.EPSILON,
        model_var_type=ref.ModelVarType.LEARNED_RANGE if options.get('learn_sigma',False) else (
            ref.ModelVarType.FIXED_SMALL if options.get('sigma_small',True) else ref.ModelVarType.FIXED_LARGE),
        loss_type=ref.LossType.MSE,rescale_timesteps=options.get('rescale_timesteps',False),lambda_geo=options.get('lambda_geo',0.))


@pytest.mark.parametrize('predict',[True,False])
@pytest.mark.parametrize('small',[True,False])
@pytest.mark.parametrize('respacing',['','10','ddim10'])
@pytest.mark.parametrize('rescale',[True,False])
def test_fixed_losses_and_gradients(reference,predict,small,respacing,rescale):
    options=dict(predict_xstart=predict,sigma_small=small,timestep_respacing=respacing,
        rescale_timesteps=rescale,lambda_geo=0.)
    source=source_schedule(reference,options)
    actual_model,expected_model=Predictor(),Predictor()
    initial=torch.get_rng_state().clone()
    actual,metrics=kernel.diffusion_training_loss(actual_model,batch(),options=options,seed=17)
    assert torch.equal(torch.get_rng_state(),initial)
    motion,cond=validate_training_batch(batch())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        t=torch.randint(source.num_timesteps,(len(motion),),device=motion.device)
        expected=source.training_losses(source._wrap_model(expected_model),motion,t,dict(cond=cond))
    assert torch.equal(actual,expected['loss'].mean())
    assert metrics=={k:v.mean().item() for k,v in expected.items()}
    actual.backward()
    expected['loss'].mean().backward()
    assert torch.equal(actual_model.weight.grad,expected_model.weight.grad)


class VariancePredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.variance=torch.nn.Parameter(torch.tensor(.1))
    def forward(self,x,t,cond):
        return torch.cat((super().forward(x,t,cond),torch.ones_like(x)*self.variance),dim=1)


def test_source_variance_omission_and_corrected_objective(reference):
    options=dict(learn_sigma=True,lambda_geo=0.,timestep_respacing='10')
    source=source_schedule(reference,options)
    expected_model,actual_model=VariancePredictor(),VariancePredictor()
    motion,cond=validate_training_batch(batch())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        t=torch.randint(source.num_timesteps,(len(motion),))
        terms=source.training_losses(source._wrap_model(expected_model),motion,t,dict(cond=cond))
    terms['loss'].mean().backward(retain_graph=True)
    assert expected_model.variance.grad.item()==0
    expected_model.zero_grad()
    expected=(terms['loss']+terms['vb']).mean()
    expected.backward()
    actual,metrics=kernel.diffusion_training_loss(actual_model,batch(),options=options,seed=17)
    assert torch.equal(actual,expected)
    assert metrics['vb']==terms['vb'].mean().item()
    actual.backward()
    assert torch.equal(actual_model.variance.grad,expected_model.variance.grad)
    assert actual_model.variance.grad.abs()>0
    released,_=kernel.diffusion_training_loss(VariancePredictor(),batch(),options={**options,'variance_policy':'released'},seed=17)
    assert torch.equal(released,terms['loss'].mean())


def test_source_unmapped_timesteps_remain_explicit(reference):
    options=dict(timestep_respacing='10',lambda_geo=0.,rescale_timesteps=True)
    model=Predictor()
    corrected,_=kernel.diffusion_training_loss(model,batch(),options=options,seed=17)
    released,_=kernel.diffusion_training_loss(model,batch(),options={**options,'timestep_policy':'released'},seed=17)
    source=source_schedule(reference,options)
    motion,cond=validate_training_batch(batch())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        t=torch.randint(source.num_timesteps,(len(motion),))
        terms=source.training_losses(model,motion,t,dict(cond=cond))
    assert torch.equal(released,terms['loss'].mean())
    assert not torch.equal(corrected,released)


def test_default_stable_geodesic_and_backward():
    model=Predictor()
    loss,metrics=kernel.diffusion_training_loss(model,batch(),seed=4)
    assert math.isfinite(metrics['geodesic_loss'])
    loss.backward()
    assert torch.isfinite(model.weight.grad)


@pytest.mark.parametrize('options',[
    {'diffusion_steps':True},{'diffusion_steps':1},{'diffusion_steps':1000001},
    {'noise_schedule':'bad'},{'noise_schedule':'linear','diffusion_steps':10},
    {'scale_beta':float('nan')},{'scale_beta':0},{'predict_xstart':1},
    {'timestep_respacing':'1'},{'timestep_respacing':[0,4]},
    {'timestep_policy':'bad'},{'variance_policy':'bad'},{'geodesic_policy':'bad'},
    {'lambda_geo':-1},{'typo':1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        kernel.create_diffusion_schedule(options)


def test_model_failure_restores_rng():
    class Broken(Predictor):
        def forward(self,*args,**kwargs):
            raise RuntimeError('forward failed')
    initial=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='forward failed'):
        kernel.diffusion_training_loss(Broken(),batch())
    assert torch.equal(initial,torch.get_rng_state())


def test_underflowing_cumulative_schedule_rejected():
    with pytest.raises(ValueError,match='cumulative'):
        kernel.create_diffusion_schedule(dict(noise_schedule='linear',diffusion_steps=10000,scale_beta=100))


def test_vendor_bodies_unchanged(reference):
    root=Path(os.environ['UNIMATE_DATASET_REFERENCE'])/'unimate/models/diffusion'
    vendor=Path(kernel.__file__).parent/'_vendor/diffusion'
    def bodies(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in ('gaussian_diffusion.py','respace.py','losses.py'):
        assert bodies(root/name)==bodies(vendor/name)


@pytest.mark.parametrize('name',['UniMateGraphAdaLN','UniMateGraphCrossAttn','UniMateFullAdaLN','UniMateFullCrossAttn'])
def test_actual_backbone_diffusion_update(name):
    from unimate_pack._vendor import denoiser
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model=getattr(denoiser,name)(feature_len=12,max_motion_length=12,max_joints=16,max_depth=32,
            latent_dim=64,ff_size=128,num_layers=1,num_heads=4,dropout=0.,cond_mode='text',
            text_dim=7,use_joint_name_emb=True,use_depth_emb=True,concat_parent_features=True,
            num_tpos_queries=2,inject_tpos_to_adaln=True)
    before=[p.detach().clone() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loss,_=kernel.diffusion_training_loss(model,batch(),seed=10)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    assert any(not torch.equal(p,b) for p,b in zip(model.parameters(),before))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required')
def test_cuda_reference_and_rng(reference):
    device=torch.device('cuda',0)
    model=Predictor().to(device)
    options=dict(lambda_geo=0.,timestep_respacing='10',rescale_timesteps=True)
    cpu_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state(device).clone()
    loss,metrics=kernel.diffusion_training_loss(model,batch(),options=options,seed=123)
    assert torch.equal(cpu_before,torch.get_rng_state())
    assert torch.equal(cuda_before,torch.cuda.get_rng_state(device))
    motion,cond=validate_training_batch(batch())
    motion=motion.to(device)
    cond={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cond.items()}
    source=source_schedule(reference,options)
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(123)
        torch.cuda.default_generators[0].manual_seed(123)
        t=torch.randint(source.num_timesteps,(len(motion),),device=device)
        expected=source.training_losses(source._wrap_model(model),motion,t,dict(cond=cond))
    assert torch.equal(loss,expected['loss'].mean())
    assert metrics=={k:v.mean().item() for k,v in expected.items()}
    loss.backward()
    assert torch.isfinite(model.weight.grad)


def test_cancellation_after_forward_restores_rng():
    cancelled=False
    class Cancelling(Predictor):
        def forward(self,*args,**kwargs):
            nonlocal cancelled
            result=super().forward(*args,**kwargs)
            cancelled=True
            return result
    def cancel():
        if cancelled:
            raise RuntimeError('cancelled')
    initial=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='cancelled'):
        kernel.diffusion_training_loss(Cancelling(),batch(),cancel=cancel)
    assert torch.equal(initial,torch.get_rng_state())
