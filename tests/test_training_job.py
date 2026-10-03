"""Jobs bind actual prepared artifacts and deterministic source epoch plans."""
from copy import deepcopy

import pytest

from test_training_dataset_samples import inputs
from unimate_pack import training_job as jobs
from unimate_pack.training_dataset_samples import dataset_identity,statistics_identity,text_cache_identity


def config():
    return dict(model=dict(latent_dim=64,ff_size=128,num_layers=1,num_heads=4,
        max_motion_length=8,max_joints=16,max_depth=32,cond_mode='text'),
        optimizer=dict(num_steps=8),loss=dict(lambda_geo=0.,lambda_smooth=0.),batch_size=1)


def test_actual_bindings_and_epoch_batch():
    dataset,stats,cache=inputs()
    original=deepcopy((dataset,stats,cache))
    job=jobs.make_training_job(dataset,stats,cache,config())
    assert job['model']['text_dim']==7
    assert job['datasets']==[dict(dataset_sha256=dataset_identity(dataset),
        statistics_sha256=statistics_identity(stats),text_cache_sha256=text_cache_identity(cache))]
    checked=jobs.validate_training_job(job,dataset,stats,cache)
    assert checked==job
    plan=jobs.plan_training_epoch(job,dataset,stats,cache,epoch=0)
    assert len(plan['batches'])==plan['batches_per_epoch']==1
    assert plan['batches']==[['a']]
    assert plan==jobs.plan_training_epoch(job,dataset,stats,cache,epoch=0)
    assert (dataset,stats,cache)==original


@pytest.mark.parametrize('paradigm',['flow','diffusion'])
def test_null_loss_normalizes_to_canonical_default_options(paradigm):
    job=jobs.make_training_job(*inputs(),{**config(),'loss':None,'paradigm':paradigm})
    assert job['loss']=={}


@pytest.mark.parametrize('change',[
    dict(batch_size=True),dict(batch_size=0),dict(seed=-1),dict(extra='path'),
    dict(paradigm='invalid'),dict(drop_last=1),dict(model=dict(text_dim=768)),
    dict(sample=dict(mode='invalid')),dict(sampling=dict(alpha=float('nan')))])
def test_invalid_job_options(change):
    with pytest.raises(ValueError):
        jobs.make_training_job(*inputs(),{**config(),**change})


def test_resume_binding_rejects_rehashed_configuration_and_changed_artifacts():
    dataset,stats,cache=inputs()
    job=jobs.make_training_job(dataset,stats,cache,config())
    corrupt=deepcopy(job)
    corrupt['model']['text_dim']=8
    corrupt['sha256']=jobs.job_identity(corrupt)
    with pytest.raises(ValueError,match='dimension'):
        jobs.validate_training_job(corrupt,dataset,stats,cache)
    changed=deepcopy(cache)
    changed['encoder']['artifact_sha256']='f'*64
    with pytest.raises(ValueError,match='binding'):
        jobs.validate_training_job(job,dataset,stats,changed)


def test_empty_drop_last_epoch_rejected_and_partial_explicit():
    dataset,stats,cache=inputs()
    job=jobs.make_training_job(dataset,stats,cache,{**config(), 'batch_size':2})
    with pytest.raises(ValueError,match='empty'):
        jobs.plan_training_epoch(job,dataset,stats,cache,epoch=0)
    job=jobs.make_training_job(dataset,stats,cache,{**config(), 'batch_size':2,'drop_last':False})
    assert jobs.plan_training_epoch(job,dataset,stats,cache,epoch=0)['batches']==[['a']]


def test_workspace_precedes_decoding_and_cancel():
    with pytest.raises(ValueError,match='workspace'):
        jobs.make_training_job(*inputs(),config(),max_workspace_bytes=1)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        jobs.make_training_job(*inputs(),config(),cancel=cancel)


def test_mixed_dataset_labels_use_source_epoch_weights():
    from unimate_pack.dataset_contracts import make_dataset
    from unimate_pack.statistics import dataset_statistics
    from unimate_pack.dataset_selection import sampling_plan,validate_sampling
    from unimate_pack.contracts import decode_arrays
    dataset,_,cache=inputs()
    manifest=deepcopy(dataset['manifest'])
    manifest['clips'][1].update(split='train',dataset_type='truebones',caption=manifest['clips'][0]['caption'])
    dataset=make_dataset(manifest,dataset['files'])
    stats=dataset_statistics(dataset)
    job=jobs.make_training_job(dataset,stats,cache,{**config(),'sampling':dict(alpha=.3,dataset_alpha=.7)})
    actual=jobs.plan_training_epoch(job,dataset,stats,cache,epoch=3)
    reference=sampling_plan(dataset,alpha=.3,dataset_alpha=.7,epoch=3)
    validate_sampling(reference,dataset)
    indices=decode_arrays(reference['arrays'])['indices']
    assert actual['batches']==[[reference['clip_ids'][int(i)]] for i in indices]
    assert set(stats['datasets'])=={'objaverse','truebones'}


@pytest.mark.parametrize('augmentation,enabled',[
    ('addition',[]),('addition_linear',[]),('random',['addition'])])
def test_augmentation_capacity_rejected_before_job_publication(augmentation,enabled):
    dataset,stats,cache=inputs()
    from unimate_pack.contracts import decode_arrays
    joints=len(decode_arrays(next(iter(dataset['files'].values())))['parents'])
    options=config()
    options['model']['max_joints']=joints
    options['sample']=dict(augmentation=augmentation,enabled=enabled)
    with pytest.raises(ValueError,match='augmentation.*capacity'):
        jobs.make_training_job(dataset,stats,cache,options)


@pytest.mark.parametrize('field,kind',[
    ('spectral_feats','missing'),('edge_indexs','missing'),('joint_graph_dists','missing'),
    ('joint_relations','missing'),('joint_depths','missing'),
    ('spectral_feats','shape'),('edge_indexs','wrong'),('joint_depths','wrong'),
    ('joint_relations','wrong'),('joint_graph_dists','wrong')])
def test_unusable_training_topology_rejected(field,kind):
    import hashlib
    from unimate_pack.contracts import decode_arrays,encode_arrays
    from unimate_pack.dataset_contracts import make_dataset
    from unimate_pack.statistics import dataset_statistics
    dataset,_,cache=inputs()
    manifest=deepcopy(dataset['manifest'])
    old=manifest['topologies'][0]['conditioning']
    arrays=decode_arrays(dataset['files'][old])
    if kind=='missing':
        del arrays[field]
    elif kind=='shape':
        arrays[field]=arrays[field][:1]
    else:
        arrays[field]=arrays[field].copy()
        arrays[field].flat[-1]=999
    payload=encode_arrays(**arrays)
    digest=hashlib.sha256(payload).hexdigest()
    files={key:value for key,value in dataset['files'].items() if key!=old}
    files[digest]=payload
    manifest['topologies'][0]['conditioning']=digest
    old_id=manifest['topologies'][0]['id']
    manifest['topologies'][0]['id']=digest
    for clip in manifest['clips']:
        if clip['topology_id']==old_id:
            clip['topology_id']=digest
    dataset=make_dataset(manifest,files)
    stats=dataset_statistics(dataset)
    with pytest.raises(ValueError):
        jobs.make_training_job(dataset,stats,cache,config())
