"""Pinned differentiable flow training and EMA comparisons."""
import ast
import hashlib
import os
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from test_training_batch_contracts import values
from unimate_pack.training_batch_contracts import collate_training_samples,validate_training_batch
from unimate_pack import training_loss as kernel
from unimate_pack.training_sample_contracts import make_training_sample,validate_training_sample
from test_training_sample_contracts import PROVENANCE,OPTIONS

HASHES={
 'models/flow/transport.py':'5e2d4aa7dd472689d984d6473d353582c5386327593d3da64e4f427b5c75af40',
 'models/flow/path.py':'ea4995a973b902a7339fb4b9d05b903b353db99c12ed48f4dae7420e7cb44692',
 'training/ema.py':'d6d9c46d9e60549a6ee16cb60fc304fc6dd01152ae5e12f8e6e86dbbf0889c3e',
 'models/diffusion/nn.py':'e180198678ad9c0f4c1a32ed63996ff3605480b26160dfe043a02b0cc801e2de',
 'utils/rotation_conversions.py':'0b3df0c50126f2298de3d9512936e404504e2089bf2a0629c1cd2ade6eed012f',
 'utils/geo_utils.py':'45fe62097f22d1d2f58f0114a2995d9fbb6b0514440b6fee1106c915c38f69d2',
}


@pytest.fixture(scope='module')
def reference():
    root=os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned training source required')
    root=Path(root)/'unimate'
    import subprocess
    assert subprocess.check_output(['git','-C',str(root.parent),'rev-parse','HEAD'],text=True).strip()=='2c5b384715aa63d8639b1ed7eb74bfe614570c7a'
    def namespace(path,context):
        source=(root/path).read_bytes()
        if path in HASHES:
            assert hashlib.sha256(source).hexdigest()==HASHES[path]
        tree=ast.parse(source)
        tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
        exec(compile(tree,str(root/path),'exec'),context)
        return types.SimpleNamespace(**context)
    path=namespace('models/flow/path.py',dict(np=np,th=torch))
    helpers=namespace('models/diffusion/nn.py',{})
    rotations=namespace('utils/rotation_conversions.py',dict(np=np,torch=torch))
    geo=namespace('utils/geo_utils.py',dict(th=torch))
    import enum
    transport=namespace('models/flow/transport.py',dict(enum=enum,np=np,th=torch,path=path,
        mean_flat=helpers.mean_flat,sum_flat=helpers.sum_flat,
        rotation_6d_to_matrix_safe=rotations.rotation_6d_to_matrix_safe,
        geodesic_distance=geo.geodesic_distance,ode=None,sde=None))
    from typing import Iterator
    ema=namespace('training/ema.py',dict(torch=torch,Iterator=Iterator))
    return transport,ema


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.2))
    def forward(self,x,t,cond):
        return x*self.weight+t[:,None,None,None]*.01


def batch():
    # Source divides by valid lengths, so choose the nonempty samples here.
    return collate_training_samples(values()[:2])


def unpadded_rotation_batch():
    sample=validate_training_sample(values()[0])
    sample['max_joints']=len(sample['parents'])
    sample['max_motion_length']=sample['motion_length']
    sample['motion']=sample['motion'][:sample['motion_length']].copy()
    sample['motion'][:,:,3:9]=[1,0,0,0,1,0]
    sample['mean'][:]=0
    sample['std'][:]=1
    return collate_training_samples([make_training_sample(sample,PROVENANCE,OPTIONS)])


@pytest.mark.parametrize('path',['Linear','GVP','VP'])
@pytest.mark.parametrize('prediction',['velocity','noise','score'])
@pytest.mark.parametrize('weight',[None,'velocity','likelihood'])
def test_flow_losses_and_gradients_match_source(reference,path,prediction,weight):
    ref,_=reference
    options=dict(path_type=path,prediction=prediction,loss_weight=weight,lambda_geo=0.,lambda_smooth=0.)
    schedule=kernel.create_flow_schedule(options)
    source=ref.Transport(model_type=getattr(ref.ModelType,prediction.upper()),
        path_type={'Linear':ref.PathType.LINEAR,'GVP':ref.PathType.GVP,'VP':ref.PathType.VP}[path],
        loss_type={None:ref.WeightType.NONE,'velocity':ref.WeightType.VELOCITY,'likelihood':ref.WeightType.LIKELIHOOD}[weight],
        train_eps=schedule.train_eps,sample_eps=schedule.sample_eps)
    portable=batch()
    model,expected_model=Predictor(),Predictor()
    initial=torch.get_rng_state().clone()
    actual,metrics=kernel.flow_training_loss(model,portable,options=options,seed=17)
    assert torch.equal(torch.get_rng_state(),initial)
    motion,cond=validate_training_batch(portable)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        terms=source.training_losses(expected_model,motion,dict(cond=cond))
    expected=terms['loss'].mean()
    assert torch.equal(actual,expected)
    assert metrics=={k:v.mean().item() for k,v in terms.items()}
    actual.backward()
    expected.backward()
    assert torch.equal(model.weight.grad,expected_model.weight.grad)


@pytest.mark.parametrize('geo,smooth',[(.5,0.),(0.,.1),(.5,.1)])
def test_auxiliary_losses_match_source(reference,geo,smooth):
    ref,_=reference
    portable=unpadded_rotation_batch()
    model=Predictor()
    actual,metrics=kernel.flow_training_loss(model,portable,options=dict(lambda_geo=geo,lambda_smooth=smooth),seed=3)
    source=ref.Transport(model_type=ref.ModelType.VELOCITY,path_type=ref.PathType.LINEAR,
        loss_type=ref.WeightType.NONE,train_eps=0.,sample_eps=0.,lambda_geo=geo,lambda_smooth=smooth)
    motion,cond=validate_training_batch(portable)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        terms=source.training_losses(model,motion,dict(cond=cond))
    assert torch.equal(actual,terms['loss'].mean())
    assert metrics=={k:v.mean().item() for k,v in terms.items()}
    actual.backward()
    assert torch.isfinite(model.weight.grad)


def test_source_degenerate_geodesic_reproduced_and_stable_adapter(reference):
    ref,_=reference
    portable=batch()
    model=Predictor()
    motion,cond=validate_training_batch(portable)
    source=ref.Transport(model_type=ref.ModelType.VELOCITY,path_type=ref.PathType.LINEAR,
        loss_type=ref.WeightType.NONE,train_eps=0.,sample_eps=0.,lambda_geo=.5,lambda_smooth=.1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        terms=source.training_losses(model,motion,dict(cond=cond))
    assert not torch.isfinite(terms['geo_loss']).all()
    loss,metrics=kernel.flow_training_loss(model,portable,seed=3)
    assert torch.isfinite(loss) and np.isfinite(metrics['geo_loss'])
    loss.backward()
    assert torch.isfinite(model.weight.grad)


def test_ema_matches_source(reference):
    _,ref=reference
    a,b=Predictor(),Predictor()
    actual=kernel.EMAModel(a.parameters(),use_ema_warmup=True)
    expected=ref.EMAModel(b.parameters(),use_ema_warmup=True)
    for index in range(12):
        with torch.no_grad():
            a.weight.add_(index*.01)
            b.weight.add_(index*.01)
        actual.step(a.parameters())
        expected.step(b.parameters())
        assert actual.cur_decay_value==expected.cur_decay_value
        assert torch.equal(actual.shadow_params[0],expected.shadow_params[0])


@pytest.mark.parametrize('options',[
    {'path_type':'unknown'}, {'prediction':'unknown'}, {'loss_weight':'unknown'},
    {'lambda_geo':-1}, {'lambda_smooth':True}, {'lambda_geo':float('inf')},
    {'train_eps':.5}, {'sample_eps':True}, {'unexpected':1},
    {'path_type':'GVP'}, {'prediction':'noise'}, {'geodesic_policy':'unknown'},
])
def test_invalid_flow_configuration(options):
    with pytest.raises(ValueError):
        kernel.create_flow_schedule(options)


def test_zero_length_and_eval_model_rejected():
    with pytest.raises(ValueError,match='positive'):
        kernel.flow_training_loss(Predictor(),collate_training_samples(values()))
    with pytest.raises(ValueError,match='training mode'):
        kernel.flow_training_loss(Predictor().eval(),batch())


def test_failed_model_and_cancel_restore_rng():
    class Failure(Predictor):
        def forward(self,*args,**kwargs):
            torch.rand(10)
            raise RuntimeError('model failed')
    initial=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='model failed'):
        kernel.flow_training_loss(Failure(),batch(),seed=123)
    assert torch.equal(torch.get_rng_state(),initial)
    class Called(Predictor):
        called=False
        def forward(self,*args,**kwargs):
            self.called=True
            return super().forward(*args,**kwargs)
    model=Called()
    def cancel():
        if model.called:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        kernel.flow_training_loss(model,batch(),seed=456,cancel=cancel)
    assert torch.equal(torch.get_rng_state(),initial)
    assert model.weight.grad is None


def test_released_geodesic_mode_stays_explicit(reference):
    ref,_=reference
    schedule=kernel.create_flow_schedule({'geodesic_policy':'released'})
    motion,cond=validate_training_batch(batch())
    terms=schedule.training_losses(Predictor(),motion,dict(cond=cond))
    assert not torch.isfinite(terms['geo_loss']).all()
    with pytest.raises(ValueError,match='Nonfinite'):
        kernel.flow_training_loss(Predictor(),batch(),options={'geodesic_policy':'released'})


def test_frozen_model_rejected_before_batch_decode(monkeypatch):
    model=Predictor()
    model.requires_grad_(False)
    def forbidden(*args,**kwargs):
        raise AssertionError('decoder reached for frozen model')
    monkeypatch.setattr(kernel,'validate_training_batch',forbidden)
    with pytest.raises(ValueError,match='trainable'):
        kernel.flow_training_loss(model,{},seed=1)


def test_nonfinite_reduction_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        kernel.flow_training_loss(Predictor(),batch(),options=dict(lambda_geo=3e38,lambda_smooth=0.),seed=1)


def test_vendored_training_bodies_unchanged(reference):
    root=Path(os.environ['UNIMATE_DATASET_REFERENCE'])/'unimate'
    vendor=Path(kernel.__file__).parent/'_vendor'
    def bodies(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    for src,dst in [('models/flow/transport.py','flow/transport.py'),('models/flow/path.py','flow/path.py'),
                    ('models/flow/integrators.py','flow/integrators.py'),('training/ema.py','ema.py')]:
        assert bodies(root/src)==bodies(vendor/dst)
    helpers=bodies(vendor/'training_math.py')
    for name,src in [('mean_flat','models/diffusion/nn.py'),('sum_flat','models/diffusion/nn.py'),
                     ('rotation_6d_to_matrix_safe','utils/rotation_conversions.py'),('geodesic_distance','utils/geo_utils.py')]:
        assert helpers[name]==bodies(root/src)[name]


@pytest.mark.parametrize('name',['UniMateGraphAdaLN','UniMateGraphCrossAttn','UniMateFullAdaLN','UniMateFullCrossAttn'])
def test_actual_backbone_backprop_adamw_and_ema(name):
    from unimate_pack._vendor import denoiser
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model=getattr(denoiser,name)(feature_len=12,max_motion_length=12,max_joints=16,max_depth=32,
            latent_dim=64,ff_size=128,num_layers=1,num_heads=4,dropout=0.,cond_mode='text',
            text_dim=7,use_joint_name_emb=True,use_depth_emb=True,concat_parent_features=True,
            num_tpos_queries=2,inject_tpos_to_adaln=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.99))
    ema=kernel.EMAModel(model.parameters(),use_ema_warmup=True)
    before=[p.detach().clone() for p in model.parameters()]
    loss,metrics=kernel.flow_training_loss(model,batch(),seed=10)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    optimizer.zero_grad()
    ema.step(model.parameters())
    assert any(not torch.equal(p,b) for p,b in zip(model.parameters(),before))
    assert all(torch.equal(p,s) for p,s in zip(model.parameters(),ema.shadow_params))
    assert np.isfinite(metrics['loss'])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device required')
def test_caller_selected_cuda_loss_preserves_rng(reference):
    ref,_=reference
    device=torch.device('cuda',0)
    model=Predictor().to(device)
    portable=batch()
    cpu_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state(device).clone()
    options=dict(lambda_geo=0.,lambda_smooth=0.)
    loss,metrics=kernel.flow_training_loss(model,portable,options=options,seed=123)
    assert loss.device==device
    assert torch.equal(cpu_before,torch.get_rng_state())
    assert torch.equal(cuda_before,torch.cuda.get_rng_state(device))
    motion,cond=validate_training_batch(portable)
    motion=motion.to(device)
    cond={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cond.items()}
    source=ref.Transport(model_type=ref.ModelType.VELOCITY,path_type=ref.PathType.LINEAR,
        loss_type=ref.WeightType.NONE,train_eps=0.,sample_eps=0.)
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(123)
        torch.cuda.default_generators[0].manual_seed(123)
        expected=source.training_losses(model,motion,dict(cond=cond))
    assert torch.equal(loss,expected['loss'].mean())
    assert metrics=={k:v.mean().item() for k,v in expected.items()}
    loss.backward()
    assert torch.isfinite(model.weight.grad)
