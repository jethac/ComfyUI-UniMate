"""Complete trained bundles bind metadata, numeric state and encoder assets."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from test_training_execution import job,cpu_residency
from test_bundle import archive
from unimate_pack import trained_bundle as bundles
from unimate_pack.bundle import inspect_bundle,extract_bundle
from unimate_pack.contracts import make_model,validate_model,encode_arrays
from unimate_pack.training_execution import run_training_job
from unimate_pack.training_job import make_training_job
from unimate_pack.inference_weights import make_inference_weights,create_inference_model
from unimate_pack.training_text import build_text_cache
from unimate_pack.training_text_model import installed_encoder_identity


def donor():
    payload=archive({'stats.npz':encode_arrays(x=np.zeros(1,dtype=np.float32))})
    return make_model(payload,'encoder.unimate')


def inputs(*,selection='ema',attention='graph',text_cond='adaln',conditioned=False,names=False):
    value,dataset,stats,cache=job()
    encoder_model=None
    if conditioned or names:
        encoder_model=donor()
        identity=installed_encoder_identity(inspect_bundle(encoder_model['bundle']))
        def encode(texts):
            tokens=[np.zeros((1,768),dtype=np.float32) for text in texts]
            return tokens,np.zeros((len(texts),768),dtype=np.float32)
        cache=build_text_cache(cache['texts'],encode,identity)
    options={key:value[key] for key in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    options['model'].update(attention=attention,text_cond=text_cond,
        cond_mode='text' if conditioned else 'no_cond',use_joint_name_emb=names)
    options['model'].pop('text_dim')
    value=make_training_job(dataset,stats,cache,options)
    checkpoint,_=run_training_job(value,dataset,stats,cache,updates=1,residency=cpu_residency)
    return make_inference_weights(checkpoint,selection),stats,cache,encoder_model


@pytest.mark.parametrize('attention,text_cond',[
    ('graph','adaln'),('graph','cross_attn'),('full','adaln'),('full','cross_attn')])
@pytest.mark.parametrize('selection',['raw','ema'])
def test_encoder_free_complete_bundle_roundtrip(attention,text_cond,selection,tmp_path):
    weights,stats,cache,encoder_model=inputs(selection=selection,attention=attention,text_cond=text_cond)
    model=bundles.assemble_trained_bundle(weights,stats,cache,encoder_model,
        sampling=dict(method='euler',num_steps=7))
    validate_model(model)
    manifest=inspect_bundle(model['bundle'])
    assert manifest['schema']=='unimate.bundle.v2'
    assert manifest['weights']==selection and manifest['model_revision']==weights['source_checkpoint_sha256']
    assert manifest['text_encoder'] is None
    assert not any(name.startswith('text_encoder/') for name in manifest['files'])
    root=tmp_path/'extract'
    root.mkdir()
    assert extract_bundle(model['bundle'],root)==manifest
    loaded,loaded_stats=bundles.read_trained_components(model['bundle'])
    assert loaded==weights and loaded_stats==stats
    rebuilt=create_inference_model(loaded)
    original=create_inference_model(weights)
    assert all(torch.equal(t,rebuilt.state_dict()[name]) for name,t in original.state_dict().items())
    from test_trained_sampling import inputs as sampling_inputs
    _,cond=sampling_inputs()
    motion=torch.zeros(1,16,12,8)
    with torch.inference_mode():
        expected=rebuilt(motion,torch.zeros(1),cond)
        for name in ('caption_emb','caption_tokens','joint_names_emb'):
            if name in cond:
                cond[name].fill_(float('nan'))
        ignored=rebuilt(motion,torch.zeros(1),cond)
    assert torch.equal(expected,ignored) and torch.isfinite(ignored).all()


@pytest.mark.parametrize('conditioned,names',[(True,False),(False,True),(True,True)])
def test_required_encoder_inventory_matches_cache(conditioned,names):
    weights,stats,cache,encoder_model=inputs(conditioned=conditioned,names=names)
    model=bundles.assemble_trained_bundle(weights,stats,cache,encoder_model)
    manifest=inspect_bundle(model['bundle'])
    assert installed_encoder_identity(manifest)==cache['encoder']
    with pytest.raises(ValueError,match='encoder'):
        bundles.assemble_trained_bundle(weights,stats,cache,None)
    other=deepcopy(encoder_model)
    other['bundle']=archive({'stats.npz':encode_arrays(x=np.zeros(1,dtype=np.float32)),
        'text_encoder/NOTICE':b'different encoder inventory'})
    other['sha256']=hashlib.sha256(other['bundle']).hexdigest()
    with pytest.raises(ValueError,match='encoder'):
        bundles.assemble_trained_bundle(weights,stats,cache,other)


@pytest.mark.parametrize('field',['statistics','cache'])
def test_rehashed_metadata_does_not_match_job(field):
    weights,stats,cache,encoder_model=inputs()
    if field=='statistics':
        stats['options']['balanced']=not stats['options']['balanced']
    else:
        cache['encoder']['encoder_version']='different'
    with pytest.raises(ValueError,match='binding'):
        bundles.assemble_trained_bundle(weights,stats,cache,encoder_model)


def rewrite(model,member=None,change=None):
    output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(model['bundle'])) as source,zipfile.ZipFile(output,'w') as target:
        manifest=json.loads(source.read('manifest.json'))
        files={name:source.read(name) for name in manifest['files']}
        if member:
            files[member]=b'unsafe'
        if change:
            change(manifest,files)
        manifest['files']={name:dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw)) for name,raw in files.items()}
        target.writestr('manifest.json',json.dumps(manifest))
        for name,raw in files.items():
            target.writestr(name,raw)
    return output.getvalue()


@pytest.mark.parametrize('member',['../escape','code.py','text_encoder/config.json','extra.json'])
def test_unsafe_unknown_or_incomplete_inventory(member):
    model=bundles.assemble_trained_bundle(*inputs())
    with pytest.raises(ValueError):
        inspect_bundle(rewrite(model,member))


@pytest.mark.parametrize('field',['weights','model_revision','statistics_identity_sha256','text_cache_identity_sha256','solver','fps'])
def test_rehashed_manifest_mismatch(field):
    model=bundles.assemble_trained_bundle(*inputs())
    def change(manifest,files):
        manifest[field]=None
    with pytest.raises(ValueError):
        inspect_bundle(rewrite(model,change=change))


def test_budget_and_cancel_before_weights_validation(monkeypatch):
    values=inputs()
    def forbidden(*args,**kwargs):
        raise AssertionError('decoded before budget/cancel')
    monkeypatch.setattr(bundles,'validate_inference_weights',forbidden)
    with pytest.raises(ValueError,match='budget'):
        bundles.assemble_trained_bundle(*values,max_workspace_bytes=1)
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        bundles.assemble_trained_bundle(*values,cancel=cancel)


@pytest.mark.parametrize('index,key',[(0,'tensors'),(1,'arrays'),(2,'arrays')])
def test_invalid_numeric_payload_is_value_error(index,key):
    values=list(inputs())
    values[index][key]=None
    with pytest.raises(ValueError):
        bundles.assemble_trained_bundle(*values)


def test_cancellation_after_final_model_validation(monkeypatch):
    values=inputs()
    finished=[]
    original=bundles.make_model
    def make(*args,**kwargs):
        result=original(*args,**kwargs)
        finished.append(True)
        return result
    monkeypatch.setattr(bundles,'make_model',make)
    def cancel():
        if finished:
            raise InterruptedError('cancelled after final validation')
    with pytest.raises(InterruptedError):
        bundles.assemble_trained_bundle(*values,cancel=cancel)


def test_bundle_bytes_do_not_depend_on_wall_clock(monkeypatch):
    values=inputs()
    monkeypatch.setattr(zipfile.time,'localtime',lambda *args:(2026,10,4,1,2,4,0,0,0))
    first=bundles.assemble_trained_bundle(*values)
    monkeypatch.setattr(zipfile.time,'localtime',lambda *args:(2026,10,4,2,3,6,0,0,0))
    second=bundles.assemble_trained_bundle(*values)
    assert first==second


def test_selected_budget_and_cancel_reach_final_contract_and_loader(monkeypatch,tmp_path):
    from unimate_pack import bundle
    from unimate_pack.inference import load_model_bundle
    values=inputs()
    budget=32*1024**3
    def cancel():
        pass
    original=bundle.inspect_bundle
    calls=[]
    def inspect(payload,**kwargs):
        assert kwargs.get('max_workspace_bytes')==budget
        assert kwargs.get('cancel') is cancel
        calls.append(True)
        return original(payload,**kwargs)
    monkeypatch.setattr(bundle,'inspect_bundle',inspect)
    model=bundles.assemble_trained_bundle(*values,cancel=cancel,max_workspace_bytes=budget)
    path=tmp_path/'trained.unimate'
    path.write_bytes(model['bundle'])
    reloaded=load_model_bundle(path,cancel=cancel,max_workspace_bytes=budget)
    assert reloaded['sha256']==model['sha256'] and len(calls)>=2


def test_loader_budget_rejects_before_file_read(monkeypatch,tmp_path):
    from pathlib import Path
    from unimate_pack.inference import load_model_bundle
    path=tmp_path/'trained.unimate'
    path.write_bytes(bundles.assemble_trained_bundle(*inputs())['bundle'])
    def forbidden(*args,**kwargs):
        raise AssertionError('read before budget')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    with pytest.raises(ValueError,match='budget'):
        load_model_bundle(path,max_workspace_bytes=1)
