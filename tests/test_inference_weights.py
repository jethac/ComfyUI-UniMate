from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import load

from test_training_execution import job,cpu_residency
from test_training_checkpoint import rehash
from test_training_session import assert_tree
from unimate_pack import inference_weights as weights
from unimate_pack.training_checkpoint import _decode_tree
from unimate_pack.training_execution import run_training_job
from unimate_pack._vendor.ema import EMAModel


@pytest.mark.parametrize('attention,text_cond',[
    ('graph','adaln'),('graph','cross_attn'),('full','adaln'),('full','cross_attn')])
@pytest.mark.parametrize('selection',['raw','ema'])
@pytest.mark.parametrize('paradigm',['flow','diffusion'])
def test_all_backbones_selected_state_and_forward(attention,text_cond,selection,paradigm):
    value,dataset,stats,cache=job()
    from unimate_pack.training_job import make_training_job
    options={k:value[k] for k in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    options['model'].update(attention=attention,text_cond=text_cond,use_spectral_rope=True,
        use_joint_name_emb=True,use_depth_emb=True,concat_parent_features=True,
        num_tpos_queries=2,inject_tpos_to_adaln=True)
    options['paradigm']=paradigm
    if paradigm=='diffusion':
        options['loss']={'lambda_geo':0.}
    value=make_training_job(dataset,stats,cache,options)
    captured=[]
    @contextmanager
    def residency(model):
        captured.append(model)
        yield model
    checkpoint,_=run_training_job(value,dataset,stats,cache,updates=2,residency=residency)
    reference=captured[0].eval()
    state=_decode_tree(checkpoint['state'],load(checkpoint['tensors']))
    if selection=='ema':
        ema=EMAModel(reference.parameters())
        ema.load_state_dict(state['ema'])
        ema.copy_to(reference.parameters())
    reference.requires_grad_(False)
    exported=weights.make_inference_weights(checkpoint,selection)
    assert exported['weights']==selection
    assert exported['job']==value
    actual=weights.create_inference_model(exported)
    assert not actual.training and not any(p.requires_grad for p in actual.parameters())
    assert_tree(actual.state_dict(),reference.state_dict())
    from unimate_pack.training_dataset_samples import produce_training_sample
    from unimate_pack.training_batch_contracts import collate_training_samples,validate_training_batch
    sample=produce_training_sample(dataset,stats,cache,'a',max_motion_length=8,max_joints=16)
    motion,condition=validate_training_batch(collate_training_samples([sample]))
    with torch.no_grad():
        expected=reference(motion,torch.tensor([.5]),condition)
        result=actual(motion,torch.tensor([.5]),condition)
    assert torch.equal(result,expected) and torch.isfinite(result).all()
    payload=weights.dump_inference_weights(exported)
    assert weights.load_inference_weights(payload)==exported
    assert weights.dump_inference_weights(weights.load_inference_weights(payload))==payload


def checkpoint():
    return run_training_job(*job(),updates=2,residency=cpu_residency)[0]


@pytest.mark.parametrize('selection',['raw','ema'])
@pytest.mark.parametrize('freeze_all',[True,False])
def test_forged_trainability_rejected_before_decode(selection,freeze_all,monkeypatch):
    source=checkpoint()
    from unimate_pack.training_checkpoint import _encode_tree
    from safetensors.torch import save
    state=_decode_tree(source['state'],load(source['tensors']))
    flags=state['trainable']
    names=list(flags) if freeze_all else [next(iter(flags))]
    for name in names:
        flags[name]=False
    if freeze_all:
        state['ema']['shadow_params']=[]
    source['state'],tensors=_encode_tree(state)
    source['tensors']=save(tensors)
    rehash(source)
    decoded=[]
    original=weights.load_tensors
    def observed(*args,**kwargs):
        decoded.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(weights,'load_tensors',observed)
    with pytest.raises(ValueError,match='trainable'):
        weights.make_inference_weights(source,selection)
    assert not decoded


@pytest.mark.parametrize('kind',['job','class','model','ema','selection'])
def test_corrupt_metadata_rejected_before_numeric_decode(kind,monkeypatch):
    source=deepcopy(checkpoint())
    selection='ema'
    if kind=='job':
        source['binding']['architecture']['config']['seed']+=1
    elif kind=='class':
        source['binding']['architecture']['class_name']='unimate.denoiser.Unknown'
    elif kind in ('model','ema'):
        # Change the shape of a referenced tensor in the header, not its digest.
        from safetensors.torch import save
        tensors=load(source['tensors'])
        state=_decode_tree(source['state'],tensors)
        target=next(iter(state['model'].values())) if kind=='model' else state['ema']['shadow_params'][0]
        name=next(k for k,v in tensors.items() if v is target)
        tensors[name]=target.reshape(-1)
        source['tensors']=save(tensors)
    else:
        selection='automatic'
    rehash(source)
    def forbidden(*args,**kwargs):
        raise AssertionError('numeric decode before preflight')
    monkeypatch.setattr(weights,'load_tensors',forbidden)
    with pytest.raises(ValueError):
        weights.make_inference_weights(source,selection)


def test_no_ema_fallback_and_predecode_budget_cancel(monkeypatch):
    value,*inputs=job(optimizer={'num_steps':4,'use_ema':False})
    source=run_training_job(value,*inputs,residency=cpu_residency)[0]
    with pytest.raises(ValueError,match='EMA'):
        weights.make_inference_weights(source,'ema')
    assert weights.make_inference_weights(source,'raw')['weights']=='raw'
    def forbidden(*args,**kwargs):
        raise AssertionError('decode before budget/cancel')
    monkeypatch.setattr(weights,'load_tensors',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        weights.make_inference_weights(source,'raw',max_workspace_bytes=1)
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        weights.make_inference_weights(source,'raw',cancel=cancel)


@pytest.mark.parametrize('change',[{'batch_size':True},{'seed':-1},{'drop_last':1},
    {'sample':{}},{'sampling':{'alpha':True,'dataset_alpha':None}}])
def test_rehashed_invalid_job_metadata_rejected(change):
    source=checkpoint()
    from unimate_pack.training_job import job_identity
    value=source['binding']['architecture']['config']
    value.update(change)
    value['sha256']=job_identity(value)
    source['binding']['sampling']['job_sha256']=value['sha256']
    rehash(source)
    with pytest.raises(ValueError):
        weights.make_inference_weights(source,'raw')


def test_inconsistent_checkpoint_alias_rejected_before_decode(monkeypatch):
    from safetensors.torch import save
    from unimate_pack.training_job import make_training_job
    from unimate_pack.training_model import training_model_spec
    value,dataset,stats,cache=job()
    options={k:value[k] for k in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    options['model'].update(use_spectral_rope=True,num_tpos_queries=2,inject_tpos_to_adaln=True)
    value=make_training_job(dataset,stats,cache,options)
    source=run_training_job(value,dataset,stats,cache,residency=cpu_residency)[0]
    tensors=load(source['tensors'])
    state=_decode_tree(source['state'],tensors)
    names=training_model_spec(value['model'])['aliases'][0]
    state['model'][names[0]].reshape(-1)[0]+=1
    source['tensors']=save(tensors)
    rehash(source)
    def forbidden(*args,**kwargs):
        raise AssertionError('numeric decode before alias preflight')
    monkeypatch.setattr(weights,'load_tensors',forbidden)
    with pytest.raises(ValueError,match='alias'):
        weights.make_inference_weights(source,'ema')
