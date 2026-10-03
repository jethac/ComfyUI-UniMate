"""Training factory axes, source initialization and allocation boundaries."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unimate_pack import training_model as factory
from unimate_pack._vendor import denoiser


def tiny(**changes):
    return dict(feature_len=12,max_motion_length=12,max_joints=16,max_depth=32,
        latent_dim=64,ff_size=128,num_layers=1,num_heads=4,text_dim=7,**changes)


@pytest.mark.parametrize('attention,text_cond',[
    ('full','adaln'),('full','cross_attn'),('graph','adaln'),('graph','cross_attn')])
@pytest.mark.parametrize('spectral',[False,True])
def test_source_initialization_and_process_rng(attention,text_cond,spectral):
    root=os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned source required')
    source=Path(root)/'unimate/models/factory.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='51fb3959d2efb3f80e93cd8270d07a65de9312605675b554228219fea0a00e3a'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_model']
    classes={(a,t):getattr(denoiser,'UniMate'+a.title()+('AdaLN' if t=='adaln' else 'CrossAttn'))
        for a in ('full','graph') for t in ('adaln','cross_attn')}
    ctx=dict(_MODEL_BY_AXES=classes,get_text_encoder_dim=lambda *_:7,
        UniMateGraphAdaLN=denoiser.UniMateGraphAdaLN,UniMateGraphCrossAttn=denoiser.UniMateGraphCrossAttn)
    exec(compile(tree,str(source),'exec'),ctx)
    config=factory.model_options(tiny(attention=attention,text_cond=text_cond,use_spectral_rope=spectral,
        use_joint_name_emb=True,use_depth_emb=True,concat_parent_features=True,num_tpos_queries=2,
        inject_tpos_to_adaln=True,dropout=.1,cond_mode='text',cond_mask_prob=.1))
    dataset=SimpleNamespace(**config)
    model_cfg=SimpleNamespace(**config,text_encoder_type='t5',text_encoder_version='fixture')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference=ctx['create_model'](dataset,model_cfg)
    rng=torch.get_rng_state().clone()
    actual=factory.create_training_model(config,seed=17)
    assert torch.equal(rng,torch.get_rng_state())
    assert actual.training and next(actual.parameters()).device.type=='cpu'
    assert type(actual) is type(reference)
    assert actual.state_dict().keys()==reference.state_dict().keys()
    for name,value in actual.state_dict().items():
        assert torch.equal(value,reference.state_dict()[name]),name


@pytest.mark.parametrize('changes',[
    dict(unknown=True),dict(latent_dim=True),dict(latent_dim=63),dict(dropout=float('nan')),
    dict(cond_mode='bogus'),dict(attention='bogus'),dict(text_cond='bogus'),
    dict(max_motion_length=0),dict(text_dim=0),dict(num_layers=1000),dict(num_heads=0),
    dict(use_signnet=1),dict(cond_mask_prob=1.1),dict(num_tpos_queries=-1)])
def test_invalid_configuration(changes):
    config=tiny()
    config.update(changes)
    with pytest.raises(ValueError):
        factory.create_training_model(config)


def test_budget_and_cancel_precede_real_allocation(monkeypatch):
    calls=[]
    original=denoiser.UniMateGraphAdaLN.__init__
    def init(self,*args,**kwargs):
        calls.append(torch.empty(0).device.type)
        original(self,*args,**kwargs)
    monkeypatch.setattr(denoiser.UniMateGraphAdaLN,'__init__',init)
    with pytest.raises(ValueError,match='budget'):
        factory.create_training_model(tiny(),max_model_bytes=1)
    assert calls==['meta']
    calls.clear()
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        factory.create_training_model(tiny(),cancel=cancel)
    assert calls==[]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_factory_does_not_seed_cuda_streams():
    before=[state.clone() for state in torch.cuda.get_rng_state_all()]
    factory.create_training_model(tiny(),seed=31)
    assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))


def test_reject_non_float32_default_before_construction():
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError,match='float32'):
            factory.create_training_model(tiny())
    finally:
        torch.set_default_dtype(old)


def test_all_vendored_backbone_bodies_match_training_source():
    root=os.environ.get('UNIMATE_DATASET_REFERENCE')
    if not root:
        pytest.skip('Explicit pinned source required')
    class Bodies(ast.NodeTransformer):
        def visit_Import(self,node):
            return None
        def visit_ImportFrom(self,node):
            return None
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Constant) and type(node.value.value) is str:
                return None
            return self.generic_visit(node)
    local=Path(denoiser.__file__).parent
    for path in local.rglob('*.py'):
        source=Path(root)/'unimate/models/denoiser'/path.relative_to(local)
        trees=[ast.dump(Bodies().visit(ast.parse(p.read_text(encoding='utf-8'))),include_attributes=False)
            for p in (path,source)]
        assert trees[0]==trees[1],str(path.relative_to(local))
