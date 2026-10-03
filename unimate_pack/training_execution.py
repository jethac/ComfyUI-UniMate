"""Node-owned training chunks over validated prepared datasets."""
import hashlib

import torch

from .dataset_contracts import _fields
from .inference import _LOCK
from .training_batch_contracts import collate_training_samples
from .training_checkpoint import make_training_checkpoint,restore_training_checkpoint,_position,model_class_identity
from .training_dataset_samples import produce_training_sample
from .training_job import validate_training_job,plan_training_epoch
from .training_model import create_training_model
from .training_session import TrainingSession
from .training_transforms import _check,_integer
from .training_initialization import inspect_initialization,apply_initialization
from .training_checkpoint import _exact


def _sample_seed(seed,epoch,batch,index,stream):
    # Independent local augmentation/crop streams are an explicit adapter policy;
    # unlike source DataLoader globals they survive process/worker scheduling.
    text=f'unimate.training.v1/{seed}/{epoch}/{batch}/{index}/{stream}'
    return int.from_bytes(hashlib.sha256(text.encode('ascii')).digest()[:8],'little')


def _position_for(plan,batch,consumed):
    return dict(epoch=plan['epoch'],batch_in_epoch=batch,batches_per_epoch=plan['batches_per_epoch'],
        consumed_batches=consumed,epoch_plan_sha256=plan['sha256'])


def _activation_bytes(config,batch_size):
    """Conservative float32 estimate, including a math-SDP fallback.

    Fused CUDA SDP can use less memory. Budgeting its unfused fallback keeps
    sequence acceptance independent of kernel availability. This is a workspace
    guard, not a prediction of hardware peak use or a guarantee against OOM.
    """
    frames,joints=config['max_motion_length']+1,config['max_joints']
    tokens=frames*joints
    heads,layers=config['num_heads'],config['num_layers']
    saved=4*batch_size*tokens*(12*config['latent_dim']+4*config['ff_size'])
    if config['attention']=='full':
        attention=8*batch_size*heads*tokens*tokens
    else:
        attention=8*batch_size*heads*(frames*joints*joints+joints*frames*frames)
    if config['text_cond']=='cross_attn' and config['cond_mode']=='text':
        attention+=8*batch_size*heads*tokens*512
    pooling=0
    if config['inject_tpos_to_adaln']:
        pooling=8*batch_size*4*config['num_tpos_queries']*joints
    return layers*(saved+attention)+pooling


def run_training_job(job,dataset,statistics,cache,*,residency,updates=1,checkpoint=None,initialization=None,
        cancel=None,progress=None,max_workspace_bytes=8*1024**3):
    """Execute complete optimizer groups and return portable state and progress.

    residency is a caller-owned context manager loading the CPU model
    onto its selected device and releasing only that model at exit. Nothing live
    is returned. Selected weights initialize a fresh optimizer/EMA session;
    checkpoint resume restores complete state. Execution is single-process.
    """
    _check(cancel)
    _integer(updates,'updates',1)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    if updates>10000 or not callable(residency):
        raise ValueError('Invalid training chunk or residency owner')
    job=validate_training_job(job,dataset,statistics,cache,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    initial_payload=None
    if job.get('initialization') is not None:
        if initialization is None:
            raise ValueError('Training job requires its selected initialization model')
        descriptor,options,initial_payload=inspect_initialization(initialization,cancel=cancel,
            max_workspace_bytes=max_workspace_bytes)
        if not _exact(descriptor,job['initialization']) or not _exact(options,job['model']):
            raise ValueError('Training initialization artifact or architecture mismatch')
    elif initialization is not None:
        raise ValueError('Scratch training job does not bind an initialization model')
    if job['paradigm']=='diffusion' and job['loss'].get('learn_sigma',False):
        raise ValueError('Learned variance requires a variance-output backbone integration')
    epoch,batch,consumed=0,0,0
    if checkpoint is not None:
        if type(checkpoint) is not dict:
            raise ValueError('Invalid training checkpoint')
        pos=checkpoint.get('position')
        _fields(pos,{'epoch','batch_in_epoch','batches_per_epoch','consumed_batches','epoch_plan_sha256'})
        _position(pos,pos['consumed_batches'])
        epoch,batch,consumed=(pos[k] for k in ('epoch','batch_in_epoch','consumed_batches'))
    plan=plan_training_epoch(job,dataset,statistics,cache,epoch=epoch,cancel=cancel,
        max_workspace_bytes=max_workspace_bytes)
    accumulation=job['optimizer']['gradient_accumulation_steps']
    if batch%accumulation:
        raise ValueError('Checkpoint position is not an optimizer group boundary')
    expected=_position_for(plan,batch,consumed)
    if checkpoint is not None and checkpoint['position']!=expected:
        raise ValueError('Checkpoint position differs from actual epoch plan')
    # Bound encoded groups and worst-case unfused attention before constructing
    # the model. Persistent rollback/checkpoint state has separate session checks.
    cfg=job['model']
    if _activation_bytes(cfg,job['batch_size'])>max_workspace_bytes//2:
        raise ValueError('Training activation estimate exceeds workspace budget')
    sample_bytes=64*cfg['max_motion_length']*cfg['max_joints']*12+256*cfg['max_joints']**2
    group_count=min(accumulation,plan['batches_per_epoch'])*job['batch_size']
    if sample_bytes*group_count>max_workspace_bytes//4:
        raise ValueError('Training accumulation group exceeds workspace budget')
    metrics=[]
    with _LOCK,torch.inference_mode(False),torch.enable_grad():
        model=create_training_model(cfg,seed=job['seed'],cancel=cancel,max_model_bytes=max_workspace_bytes//8)
        if initial_payload is not None:
            apply_initialization(model,initial_payload,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
            del initial_payload
        with residency(model) as resident:
            if resident is not model:
                raise ValueError('Residency must preserve the owned training model')
            session=None
            try:
                model.train()
                session=TrainingSession(model,paradigm=job['paradigm'],loss_options=job['loss'],
                    options=job['optimizer'],seed=job['seed'])
                binding=dict(architecture=dict(class_name=model_class_identity(model),
                    config=job,initial_weights_sha256=job.get('initialization',{}).get('denoiser_sha256')),datasets=job['datasets'],
                    sampling=dict(**job['sampling'],job_sha256=job['sha256']))
                if checkpoint is not None:
                    restore_training_checkpoint(session,checkpoint,binding,expected_position=expected,
                        cancel=cancel,max_workspace_bytes=max_workspace_bytes)
                for _ in range(min(updates,session.options['num_steps']-session.updates)):
                    _check(cancel)
                    end=min(batch+accumulation,plan['batches_per_epoch'])
                    batches=[]
                    for offset in range(batch,end):
                        samples=[]
                        for index,clip_id in enumerate(plan['batches'][offset]):
                            samples.append(produce_training_sample(dataset,statistics,cache,clip_id,
                                **{k:v for k,v in job['sample'].items() if k!='enabled'},
                                enabled=tuple(job['sample']['enabled']),
                                max_motion_length=cfg['max_motion_length'],max_joints=cfg['max_joints'],
                                max_freqs=cfg['max_freqs'],
                                augmentation_seed=_sample_seed(job['seed'],epoch,offset,index,'augmentation')%(2**32),
                                crop_seed=_sample_seed(job['seed'],epoch,offset,index,'crop'),
                                cancel=cancel,max_workspace_bytes=max_workspace_bytes))
                        batches.append(collate_training_samples(samples,cancel=cancel,
                            max_workspace_bytes=max_workspace_bytes))
                    report=session.step(batches,final_group=end-batch<accumulation,cancel=cancel,
                        max_state_bytes=max_workspace_bytes,max_workspace_bytes=max_workspace_bytes)
                    metrics.append(report)
                    consumed+=end-batch
                    batch=end
                    if batch==plan['batches_per_epoch']:
                        epoch+=1
                        batch=0
                        plan=plan_training_epoch(job,dataset,statistics,cache,epoch=epoch,cancel=cancel,
                            max_workspace_bytes=max_workspace_bytes)
                    if progress:
                        progress(dict(report,epoch=epoch,batch_in_epoch=batch))
                value=make_training_checkpoint(session,binding,_position_for(plan,batch,consumed),
                    cancel=cancel,max_workspace_bytes=max_workspace_bytes)
                result=dict(updates=session.updates,consumed_batches=consumed,epoch=epoch,
                    batch_in_epoch=batch,complete=session.updates==session.options['num_steps'],metrics=metrics,
                    job_sha256=job['sha256'])
                _check(cancel)
                return value,result
            finally:
                model.zero_grad(set_to_none=True)
                del session
