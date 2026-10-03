from copy import deepcopy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import torch
import numpy as np
from safetensors.torch import save

from test_bundle import archive
from unimate_pack.contracts import make_model,encode_arrays
from unimate_pack import training_initialization as initialization


def config():
    return json.loads((Path(__file__).parent/'fixtures/model_configs/unimate_uniml3d_f60_v2.json').read_text(encoding='utf-8-sig'))


def bundle(weights='ema'):
    raw=archive(manifest_update={'weights':weights})
    output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        contents={name:source.read(name) for name in source.namelist() if name!='manifest.json'}
        manifest=json.loads(source.read('manifest.json'))
    contents['config.json']=json.dumps(config()).encode()
    contents['stats.npz']=encode_arrays(fixture=np.zeros(12))
    contents['denoiser.safetensors']=save({'weight':torch.ones(2,3)})
    manifest['files']={name:dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data)) for name,data in contents.items()}
    with zipfile.ZipFile(output,'w') as target:
        target.writestr('manifest.json',json.dumps(manifest))
        for name,data in contents.items():
            target.writestr(name,data)
    return make_model(output.getvalue(),'selected.unimate')


@pytest.mark.parametrize('weights',['raw','ema'])
def test_bundle_descriptor_and_architecture_metadata(weights,monkeypatch):
    monkeypatch.setattr(initialization,'training_model_layout',lambda *a,**k:({'weight':((2,3),torch.float32)},[]))
    selected=bundle(weights)
    descriptor,options,payload=initialization.inspect_initialization(selected)
    assert descriptor['bundle_sha256']==selected['sha256']
    assert descriptor['weights']==weights
    assert descriptor['denoiser_sha256']==hashlib.sha256(payload).hexdigest()
    assert options['text_dim']==768 and options['max_motion_length']==60
    assert options['attention']=='graph' and options['text_cond']=='adaln'
    model=torch.nn.Linear(3,2,bias=False)
    initialization.apply_initialization(model,payload)
    assert torch.equal(model.weight,torch.ones(2,3))


def test_validation_precedes_mutation_and_aliases():
    model=torch.nn.Linear(3,2,bias=False)
    before=deepcopy(model.state_dict())
    for state in ({'weight':torch.ones(2,4)}, {'weight':torch.full((2,3),float('nan'))},
                  {'weight':torch.ones(2,3,dtype=torch.float64)}, {'wrong':torch.ones(2,3)}):
        with pytest.raises(ValueError):
            initialization.apply_initialization(model,save(state))
        assert torch.equal(model.weight,before['weight'])
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Linear(3,2,bias=False)
            self.b=self.a
    shared=Shared()
    with pytest.raises(ValueError,match='alias'):
        initialization.apply_initialization(shared,save({'a.weight':torch.ones(2,3),'b.weight':torch.zeros(2,3)}))


def test_budget_and_cancel_before_bundle_read(monkeypatch):
    selected=bundle()
    def forbidden(*args,**kwargs):
        raise AssertionError('read before budget/cancel')
    monkeypatch.setattr(initialization,'inspect_bundle',forbidden)
    with pytest.raises(ValueError,match='workspace'):
        initialization.inspect_initialization(selected,max_workspace_bytes=1)
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        initialization.inspect_initialization(selected,cancel=cancel)


def test_raw_and_ema_checkpoint_selection_without_fallback():
    model=torch.nn.Linear(3,2,bias=False)
    checkpoint={'model_state_dict':{'weight':torch.ones(2,3)},
        'ema_state_dict':{'shadow_params':[torch.full((2,3),2.)]}}
    initialization.select_checkpoint_weights(model,checkpoint,'raw')
    assert torch.equal(model.weight,torch.ones(2,3))
    initialization.select_checkpoint_weights(model,checkpoint,'ema')
    assert torch.equal(model.weight,torch.full((2,3),2.))
    initialization.select_checkpoint_weights(model,{'model_state_dict':checkpoint['model_state_dict']},'raw')
    with pytest.raises(ValueError,match='no raw fallback'):
        initialization.select_checkpoint_weights(model,{'model_state_dict':checkpoint['model_state_dict']},'ema')


def test_job_derives_selected_architecture_and_rejects_overrides(monkeypatch):
    monkeypatch.setattr(initialization,'training_model_layout',lambda *a,**k:({'weight':((2,3),torch.float32)},[]))
    from test_training_dataset_samples import inputs,IDENTITY
    from unimate_pack.training_text import build_text_cache
    from unimate_pack.training_job import make_training_job,validate_training_job
    from unimate_pack import training_execution
    dataset,stats,old=inputs()
    def encode(texts):
        tokens=[np.zeros((2,768),np.float32) for _ in texts]
        return tokens,np.stack([t.mean(0) for t in tokens])
    cache=build_text_cache(old['texts'],encode,IDENTITY)
    selected=bundle('raw')
    job=make_training_job(dataset,stats,cache,initialization=selected,max_workspace_bytes=8*1024**3)
    assert job['initialization']['weights']=='raw'
    assert job['model']['num_layers']==10 and job['model']['use_spectral_rope']
    assert validate_training_job(job,dataset,stats,cache)==job
    with pytest.raises(ValueError,match='architecture'):
        make_training_job(dataset,stats,cache,{'model':{'num_layers':1}},initialization=selected)
    def forbidden(*args,**kwargs):
        raise AssertionError('model allocated before initialization validation')
    monkeypatch.setattr(training_execution,'create_training_model',forbidden)
    with pytest.raises(ValueError,match='initialization'):
        training_execution.run_training_job(job,dataset,stats,cache,residency=lambda model:None)
    with pytest.raises(ValueError,match='initialization'):
        training_execution.run_training_job(job,dataset,stats,cache,initialization=bundle('ema'),residency=lambda model:None)


def test_released_architecture_inventory_rejected_before_real_allocation(monkeypatch):
    from unimate_pack import training_model
    def forbidden(*args,**kwargs):
        raise AssertionError('real model allocated before inventory rejection')
    monkeypatch.setattr(training_model,'create_training_model',forbidden)
    with pytest.raises(ValueError,match='architecture'):
        initialization.inspect_initialization(bundle())
