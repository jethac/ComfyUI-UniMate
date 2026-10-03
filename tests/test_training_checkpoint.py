"""Bounded numeric checkpoints and validation before session mutation."""
from copy import deepcopy
import hashlib
import json
import struct
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from test_training_loss import batch,Predictor
from test_training_session import make,assert_tree
from unimate_pack import training_checkpoint as contract
from unimate_pack.dataset_contracts import manifest_bytes
from unimate_pack.training_session import TrainingSession


def binding(session):
    cls=type(session.model)
    return dict(architecture=dict(class_name=cls.__module__+'.'+cls.__qualname__,
        config={'fixture':True},initial_weights_sha256=None),
        datasets=[dict(dataset_sha256='1'*64,statistics_sha256='2'*64,text_cache_sha256='3'*64)],
        sampling=dict(seed=10,balanced=False))


def position(session):
    return dict(epoch=session.batches//4,batch_in_epoch=session.batches%4,
        batches_per_epoch=4,consumed_batches=session.batches,epoch_plan_sha256='4'*64)


def checkpoint(session):
    return contract.make_training_checkpoint(session,binding(session),position(session))


def rehash(value):
    value['sha256']=hashlib.sha256(value['tensors']).hexdigest()
    value['identity_sha256']=hashlib.sha256(manifest_bytes({k:v for k,v in value.items()
        if k not in ('tensors','identity_sha256')})).hexdigest()
    return value


def test_roundtrip_resumes_exactly():
    source=make(gradient_accumulation_steps=2)
    source.step([batch(),batch()])
    value=checkpoint(source)
    resumed=make(gradient_accumulation_steps=2)
    state,pos=contract.validate_training_checkpoint(value,resumed,binding(resumed))
    assert_tree(state,source.snapshot())
    assert pos==position(source)
    contract.restore_training_checkpoint(resumed,value,binding(resumed),expected_position=pos)
    source.step([batch(),batch()])
    resumed.step([batch(),batch()])
    assert_tree(source.snapshot(),resumed.snapshot())
    assert type(value['tensors']) is bytes
    assert 'pickle' not in value['schema']


def test_initial_state_without_ema_roundtrip():
    source=make(use_ema=False)
    state,pos=contract.validate_training_checkpoint(checkpoint(source),source,binding(source))
    assert_tree(state,source.snapshot())
    assert pos==position(source)


@pytest.mark.parametrize('kind',['binding','position','runtime','identity','bytes'])
def test_envelope_rejection_before_decode_and_mutation(kind,monkeypatch):
    source=make()
    source.step([batch()])
    value=deepcopy(checkpoint(source))
    target=make()
    before=target.snapshot()
    if kind=='binding':
        value['binding']['datasets'][0]['dataset_sha256']='9'*64
        rehash(value)
    elif kind=='position':
        value['position']['consumed_batches']+=1
        rehash(value)
    elif kind=='runtime':
        value['runtime']['torch']='unknown'
        rehash(value)
    elif kind=='identity':
        value['identity_sha256']='0'*64
    else:
        value['tensors']=value['tensors'][:-1]+bytes([value['tensors'][-1]^1])
    monkeypatch.setattr(contract,'load_tensors',lambda raw:pytest.fail('decoded invalid envelope'))
    with pytest.raises(ValueError):
        contract.restore_training_checkpoint(target,value,binding(target))
    assert_tree(before,target.snapshot())


def mutate_state(value,edit):
    # Helpers are deliberately exercised with rehashed corruption: digest checks
    # alone cannot establish semantic optimizer/resume validity.
    state=contract._decode_tree(value['state'],contract.load_tensors(value['tensors']))
    edit(state)
    tree,tensors=contract._encode_tree(state)
    value['state']=tree
    value['tensors']=save(tensors)
    return rehash(value)


@pytest.mark.parametrize('edit',[
    lambda s:s.update(updates=99),
    lambda s:s.update(batches=0),
    lambda s:s['options'].update(learning_rate=1.),
    lambda s:s['optimizer']['param_groups'][0].update(lr=9.),
    lambda s:s['optimizer']['state'][0].update(exp_avg=torch.zeros(2)),
    lambda s:s['optimizer']['state'][0].update(exp_avg_sq=torch.tensor(-1.)),
    lambda s:s['optimizer']['state'][0].update(step=torch.tensor(99.)),
    lambda s:s['scheduler'].update(last_epoch=0),
    lambda s:s['ema'].update(optimization_step=0),
    lambda s:s['ema'].update(decay=.5),
    lambda s:s['ema'].update(shadow_params=[torch.ones(2)]),
    lambda s:s.update(rng_cpu=torch.zeros(2,dtype=torch.uint8)),
    lambda s:s.update(rng_cpu=torch.zeros_like(s['rng_cpu'])),
    lambda s:s['modes'].update({'':False}),
    lambda s:s['trainable'].update(weight=False),
    lambda s:s['scaler'].update(scale=1.),
    lambda s:s['model'].update(weight=torch.tensor(float('nan'))),
    lambda s:s['optimizer'].update(state={}),
    lambda s:s['options'].update(use_ema=1),
    lambda s:s['modes'].update({'':1}),
    lambda s:s['optimizer']['param_groups'][0].update(amsgrad=0),
])
def test_rehashed_semantic_corruption_rejected(edit):
    source=make()
    source.step([batch()])
    value=mutate_state(checkpoint(source),edit)
    target=make()
    before=target.snapshot()
    with pytest.raises(ValueError):
        contract.restore_training_checkpoint(target,value,binding(target))
    assert_tree(before,target.snapshot())


def test_workspace_preflight_before_decode(monkeypatch):
    source=make()
    value=checkpoint(source)
    monkeypatch.setattr(contract,'load_tensors',lambda raw:pytest.fail('decoded over budget'))
    with pytest.raises(ValueError,match='budget'):
        contract.validate_training_checkpoint(value,source,binding(source),max_workspace_bytes=1)


def test_header_shape_rejected_before_decode(monkeypatch):
    source=make()
    value=checkpoint(source)
    length=struct.unpack('<Q',value['tensors'][:8])[0]
    header=json.loads(value['tensors'][8:8+length])
    first=next(iter(header))
    header[first]['shape']=[2**63]
    raw=manifest_bytes(header)
    value['tensors']=struct.pack('<Q',len(raw))+raw+value['tensors'][8+length:]
    rehash(value)
    monkeypatch.setattr(contract,'load_tensors',lambda raw:pytest.fail('decoded invalid header'))
    with pytest.raises(ValueError):
        contract.validate_training_checkpoint(value,source,binding(source))


def test_unknown_tree_tag_rejected_before_decode(monkeypatch):
    source=make()
    value=checkpoint(source)
    value['state']['kind']='python_object'
    rehash(value)
    monkeypatch.setattr(contract,'load_tensors',lambda raw:pytest.fail('decoded invalid tree'))
    with pytest.raises(ValueError):
        contract.validate_training_checkpoint(value,source,binding(source))


def test_cancelled_restore_leaves_session_untouched():
    source=make()
    source.step([batch()])
    target=make()
    before=target.snapshot()
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls==3:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        contract.restore_training_checkpoint(target,checkpoint(source),binding(target),cancel=cancel)
    assert_tree(before,target.snapshot())


def test_diffusion_portable_resume():
    def new():
        return TrainingSession(Predictor(),paradigm='diffusion',loss_options=dict(lambda_geo=0.),
            options=dict(num_steps=8,warmup_ratio=.25),seed=17)
    source=new()
    source.step([batch()])
    value=checkpoint(source)
    resumed=new()
    contract.restore_training_checkpoint(resumed,value,binding(resumed))
    source.step([batch()])
    resumed.step([batch()])
    assert_tree(source.snapshot(),resumed.snapshot())


@pytest.mark.parametrize('name',['UniMateGraphAdaLN','UniMateGraphCrossAttn','UniMateFullAdaLN','UniMateFullCrossAttn'])
def test_backbone_portable_resume(name):
    from unimate_pack._vendor import denoiser
    def new():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model=getattr(denoiser,name)(feature_len=12,max_motion_length=12,max_joints=16,max_depth=32,
                latent_dim=64,ff_size=128,num_layers=1,num_heads=4,dropout=.1,cond_mask_prob=.2,
                cond_mode='text',text_dim=7,use_joint_name_emb=True,use_depth_emb=True,
                concat_parent_features=True,num_tpos_queries=2,inject_tpos_to_adaln=True)
        return make(model)
    source=new()
    source.step([batch()])
    value=checkpoint(source)
    resumed=new()
    contract.restore_training_checkpoint(resumed,value,binding(resumed))
    source.step([batch()])
    resumed.step([batch()])
    assert_tree(source.snapshot(),resumed.snapshot())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['none','bf16','fp16'])
def test_cuda_portable_resume_and_scaler_validation(precision):
    def new():
        return make(Predictor().to('cuda:0'),precision=precision)
    source=new()
    source.step([batch()])
    value=checkpoint(source)
    resumed=new()
    contract.restore_training_checkpoint(resumed,value,binding(resumed))
    source.step([batch()])
    resumed.step([batch()])
    assert_tree(source.snapshot(),resumed.snapshot())
    if precision=='fp16':
        bad=mutate_state(deepcopy(value),lambda s:s['scaler'].update(_growth_tracker=0))
        before=resumed.snapshot()
        with pytest.raises(ValueError):
            contract.restore_training_checkpoint(resumed,bad,binding(resumed))
        assert_tree(before,resumed.snapshot())


def test_actual_cloud_codecs_roundtrip_checkpoint(tmp_path):
    from test_dataset_transport import codec
    root=Path(__file__).resolve().parents[2]
    client=codec(Path(os.environ.get('CLOUD_CLIENT_ROOT',root/'ComfyUI-Cloud-Offload'))/'partition_protocol.py','checkpoint_client_codec')
    runner=codec(Path(os.environ.get('CLOUD_RUNNER_ROOT',root/'cloud-offload'))/'cloud_offload/partition_protocol.py','checkpoint_runner_codec')
    source=make()
    source.step([batch()])
    value=checkpoint(source)
    kind='UNIMATE_TRAINING_CHECKPOINT'
    client.validate_boundary_type(kind)
    runner.validate_boundary_type(kind)
    first=tmp_path/'client checkpoint.partition'
    client.dump_bundle(client.pack_execution_values([value]),first)
    restored=runner.unpack_execution_values(runner.load_bundle(first))
    assert restored==[value]
    second=tmp_path/'worker checkpoint.partition'
    runner.dump_bundle(runner.pack_execution_values(restored),second)
    result=client.unpack_execution_values(client.load_bundle(second))[0]
    target=make()
    contract.restore_training_checkpoint(target,result,binding(target))
    assert_tree(source.snapshot(),target.snapshot())


def test_make_budget_rejected_before_snapshot_copy(monkeypatch):
    source=make()
    monkeypatch.setattr(source,'snapshot',lambda **kw:pytest.fail('copied over-budget state'))
    with pytest.raises(ValueError,match='budget'):
        contract.make_training_checkpoint(source,binding(source),position(source),max_workspace_bytes=32768)


def test_restore_cancellation_after_live_load_is_atomic():
    source=make()
    source.step([batch()])
    value=checkpoint(source)
    target=make()
    before=target.snapshot()
    def cancel():
        if target.updates==1:
            raise RuntimeError('cancelled after load')
    with pytest.raises(RuntimeError,match='cancelled after load'):
        contract.restore_training_checkpoint(target,value,binding(target),cancel=cancel)
    assert_tree(before,target.snapshot())


@pytest.mark.parametrize('text,count',[('x'*65536,65),('\x01'*65536,12)],ids=['aggregate','escaping'])
def test_json_aggregate_and_escape_budget_before_serialization(text,count,monkeypatch):
    source=make()
    config=binding(source)
    config['sampling']['values']=[text]*count
    monkeypatch.setattr(contract,'manifest_bytes',lambda value:pytest.fail('serialized oversized JSON'))
    with pytest.raises(ValueError,match='budget'):
        contract.make_training_checkpoint(source,config,position(source))


@pytest.mark.parametrize('setting',['matmul_precision','cudnn_benchmark','cudnn_deterministic'])
def test_runtime_numeric_settings_must_match(setting,monkeypatch):
    source=make()
    if setting=='matmul_precision':
        original=torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('high')
    value=checkpoint(source)
    try:
        if setting=='matmul_precision':
            torch.set_float32_matmul_precision('medium')
        elif setting=='cudnn_benchmark':
            monkeypatch.setattr(torch.backends.cudnn,'benchmark',not torch.backends.cudnn.benchmark)
        else:
            monkeypatch.setattr(torch.backends.cudnn,'deterministic',not torch.backends.cudnn.deterministic)
        with pytest.raises(ValueError,match='runtime'):
            contract.validate_training_checkpoint(value,source,binding(source))
    finally:
        if setting=='matmul_precision':
            torch.set_float32_matmul_precision(original)


@pytest.mark.parametrize('corruption',['delete','lower'])
def test_partial_adam_history_corruption_rejected(corruption):
    class TwoParameters(Predictor):
        def __init__(self):
            super().__init__()
            self.offset=torch.nn.Parameter(torch.tensor(.1))
        def forward(self,x,t,cond):
            return super().forward(x,t,cond)+self.offset
    source=make(TwoParameters())
    source.step([batch()])
    source.step([batch()])
    def edit(state):
        if corruption=='delete':
            del state['optimizer']['state'][1]
        else:
            state['optimizer']['state'][1]['step'].fill_(1)
    bad=mutate_state(checkpoint(source),edit)
    target=make(TwoParameters())
    before=target.snapshot()
    with pytest.raises(ValueError):
        contract.restore_training_checkpoint(target,bad,binding(target))
    assert_tree(before,target.snapshot())


def test_optimizer_coverage_accounts_for_unused_and_frozen_parameters():
    class SparseParameters(Predictor):
        def __init__(self):
            super().__init__()
            self.unused=torch.nn.Parameter(torch.tensor(.1))
            self.frozen=torch.nn.Parameter(torch.tensor(.3),requires_grad=False)
    source=make(SparseParameters())
    source.step([batch()])
    assert source.optimizer_updates==[1,0,0]
    target=make(SparseParameters())
    contract.restore_training_checkpoint(target,checkpoint(source),binding(target))
    source.step([batch()])
    target.step([batch()])
    assert_tree(source.snapshot(),target.snapshot())
