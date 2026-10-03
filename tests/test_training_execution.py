"""Actual prepared-data optimization, deterministic chunks and data resume."""
from contextlib import contextmanager

import pytest
import torch

from test_training_dataset_samples import inputs
from test_training_job import config
from unimate_pack.training_job import make_training_job
from unimate_pack import training_execution as execution


@contextmanager
def cpu_residency(model):
    yield model


def job(**changes):
    dataset,stats,cache=inputs()
    options=config()
    options['model'].update(cond_mask_prob=.1,dropout=.1)
    options.update(changes)
    return make_training_job(dataset,stats,cache,options),dataset,stats,cache


@pytest.mark.parametrize('accumulation',[1,2])
@pytest.mark.parametrize('paradigm',['flow','diffusion'])
def test_chunked_execution_matches_uninterrupted_actual_dataset(accumulation,paradigm):
    value,dataset,stats,cache=job(optimizer=dict(num_steps=4,gradient_accumulation_steps=accumulation),
        paradigm=paradigm,loss=dict(lambda_geo=0.) if paradigm=='diffusion' else dict(lambda_geo=0.,lambda_smooth=0.))
    before=torch.get_rng_state().clone()
    full,progress=execution.run_training_job(value,dataset,stats,cache,updates=3,residency=cpu_residency)
    first,_=execution.run_training_job(value,dataset,stats,cache,updates=1,residency=cpu_residency)
    resumed,report=execution.run_training_job(value,dataset,stats,cache,updates=2,
        checkpoint=first,residency=cpu_residency)
    assert full==resumed
    assert progress['updates']==report['updates']==3
    assert report['consumed_batches']==3
    assert report['epoch']==3 and report['batch_in_epoch']==0
    assert torch.equal(before,torch.get_rng_state())
    assert all(isinstance(m['loss'],float) for m in report['metrics'])


def test_residency_released_on_failure_and_progress_exception():
    events=[]
    @contextmanager
    def residency(model):
        events.append('loaded')
        try:
            yield model
        finally:
            events.append('released')
    def progress(_):
        raise RuntimeError('progress failure')
    with pytest.raises(RuntimeError,match='progress failure'):
        execution.run_training_job(*job(),updates=1,residency=residency,progress=progress)
    assert events==['loaded','released']


def test_wrong_job_and_epoch_identity_rejected_before_optimizer_step(monkeypatch):
    from copy import deepcopy
    from unimate_pack.training_checkpoint import _identity
    value,dataset,stats,cache=job()
    checkpoint,_=execution.run_training_job(value,dataset,stats,cache,updates=1,residency=cpu_residency)
    checkpoint=deepcopy(checkpoint)
    checkpoint['position']['epoch_plan_sha256']='f'*64
    checkpoint['identity_sha256']=_identity(checkpoint)
    def forbidden(*args,**kwargs):
        raise AssertionError('optimizer executed before resume validation')
    monkeypatch.setattr(execution.TrainingSession,'step',forbidden)
    with pytest.raises(ValueError,match='position'):
        execution.run_training_job(value,dataset,stats,cache,checkpoint=checkpoint,residency=cpu_residency)


def test_zero_chunk_and_budget_rejected_before_model_allocation(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('model allocation')
    monkeypatch.setattr(execution,'create_training_model',forbidden)
    with pytest.raises(ValueError):
        execution.run_training_job(*job(),updates=0,residency=cpu_residency)
    with pytest.raises(ValueError,match='workspace'):
        execution.run_training_job(*job(),residency=cpu_residency,max_workspace_bytes=1)


def test_checkpoint_uses_source_class_identity():
    checkpoint,_=execution.run_training_job(*job(),residency=cpu_residency)
    assert checkpoint['binding']['architecture']['class_name']=='unimate.denoiser.UniMateGraphAdaLN'


def test_training_inside_comfy_inference_context():
    with torch.inference_mode(),torch.no_grad():
        checkpoint,report=execution.run_training_job(*job(),residency=cpu_residency)
        assert report['updates']==1
    assert checkpoint['position']['consumed_batches']==1


def test_full_attention_activation_budget_precedes_model_allocation(monkeypatch):
    dataset,stats,cache=inputs()
    options=config()
    options['model'].update(attention='full',max_motion_length=512,max_joints=512)
    value=make_training_job(dataset,stats,cache,options)
    def forbidden(*args,**kwargs):
        raise AssertionError('model constructed before activation preflight')
    monkeypatch.setattr(execution,'create_training_model',forbidden)
    with pytest.raises(ValueError,match='activation.*workspace'):
        execution.run_training_job(value,dataset,stats,cache,residency=cpu_residency)


@pytest.mark.parametrize('paradigm',['flow','diffusion'])
def test_null_loss_executes_default_objective(paradigm):
    checkpoint,report=execution.run_training_job(*job(paradigm=paradigm,loss=None),residency=cpu_residency)
    assert checkpoint['position']['consumed_batches']==report['consumed_batches']==1


def test_mid_epoch_resume_and_partial_group_cross_epoch_boundary():
    from copy import deepcopy
    from unimate_pack.dataset_contracts import make_dataset
    from unimate_pack.statistics import dataset_statistics
    dataset,_,cache=inputs()
    manifest=deepcopy(dataset['manifest'])
    manifest['clips'][1].update(split='train',caption=manifest['clips'][0]['caption'])
    third=deepcopy(manifest['clips'][0])
    third['id']='c'
    manifest['clips'].append(third)
    dataset=make_dataset(manifest,dataset['files'])
    stats=dataset_statistics(dataset)
    options=config()
    options['optimizer']=dict(num_steps=8,gradient_accumulation_steps=2)
    value=make_training_job(dataset,stats,cache,options)
    full,_=execution.run_training_job(value,dataset,stats,cache,updates=4,residency=cpu_residency)
    first,_=execution.run_training_job(value,dataset,stats,cache,updates=1,residency=cpu_residency)
    assert first['position']['epoch']==0 and first['position']['batch_in_epoch']==2
    assert first['position']['batches_per_epoch']==3
    resumed,report=execution.run_training_job(value,dataset,stats,cache,updates=3,
        checkpoint=first,residency=cpu_residency)
    assert resumed==full
    assert report['epoch']==2 and report['consumed_batches']==6
