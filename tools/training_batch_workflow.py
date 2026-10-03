"""Headless ComfyUI execution-list collation and partition batch transport."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check_sample_cache(samples,cache):
    from unimate_pack.training_dataset_samples import text_cache_identity
    identity=text_cache_identity(cache)
    if any(sample['provenance']['text_cache_sha256']!=identity for sample in samples):
        raise ValueError('Training samples differ from source text cache identity')


def workflow_graph(mode):
    if mode not in ('build','restore','reload'):
        raise ValueError('Unknown batch workflow mode')
    key,kind=('batch','UNIMATE_TRAINING_BATCH') if mode=='restore' else ('samples','UNIMATE_TRAINING_SAMPLE')
    graph={'in_'+key:dict(class_type='CloudPartitionInput',inputs=dict(
        boundary_key=key,type_name=kind,artifact_path=''))}
    if mode!='restore':
        graph['collate']=dict(class_type='UniMateCollateTrainingSamples',inputs=dict(
            samples=['in_samples',0],workspace_mib=512))
    graph['out_batch']=dict(class_type='CloudPartitionOutput',inputs=dict(
        value=['in_batch' if mode=='restore' else 'collate',0],boundary_key='batch',
        type_name='UNIMATE_TRAINING_BATCH',output_path=''))
    return graph,[dict(key=key,type_name=kind)],[dict(key='batch',type_name='UNIMATE_TRAINING_BATCH')]


def run_checks(base,args,workspace):
    import torch
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.partition_protocol import load_bundle,unpack_execution_values
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker
    from tools.training_workflow import CASES
    from tools.verify_workflow import submit
    from unimate_pack.training_batch_contracts import collate_training_samples,validate_training_batch
    from unimate_pack.training_collation import collate_samples
    from unimate_pack.training_sample_contracts import validate_training_sample
    from unimate_pack.training_text import validate_text_cache

    spec=importlib.util.spec_from_file_location('batch_client_codec',args.client_root/'partition_protocol.py')
    client=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.validate_boundary_type('UNIMATE_TRAINING_BATCH')
    config=CloudConfig(storage_path=str(workspace/'worker-store'),queue_db_path=str(workspace/'worker-queue.db'))
    worker=Worker.__new__(Worker)
    worker.queue,worker.storage=JobQueue(config.queue_db_path),LocalStorage(config.storage_path)
    executor=ComfyUIWorkflowExecutor(base_url=base)
    sources=[args.samples_dir/f'direct-sample_{i}.part' for i in range(len(CASES))]
    def read(path):
        return client.unpack_execution_values(client.load_bundle(path))
    samples=[]
    for source in sources:
        values=read(source)
        assert len(values)==1
        validate_training_sample(values[0])
        samples.append(values[0])
    cache=read(args.samples_dir/'direct-cache.part')[0]
    validate_text_cache(cache)
    check_sample_cache(samples,cache)
    expected={mode:collate_training_samples(samples if mode!='reload' else samples[::-1])
              for mode in ('build','reload')}
    expected['restore']=expected['build']
    original=collate_samples([validate_training_sample(s) for s in samples])
    def check(value,mode):
        assert value==expected[mode]
        motion,cond=validate_training_batch(value)
        if mode!='reload':
            assert torch.equal(motion,original[0])
            for key,item in original[1].items():
                if isinstance(item,torch.Tensor):
                    assert torch.equal(cond[key],item),key
        order=samples[::-1] if mode=='reload' else samples
        assert [ref['identity_sha256'] for ref in value['samples']]==[s['identity_sha256'] for s in order]
        assert motion.shape==(len(samples),16,12,60)
        return dict(identity_sha256=value['identity_sha256'],motion_shape=list(motion.shape),
                    n_joints=cond['n_joints'].tolist(),motion_lengths=cond['motion_length'].tolist(),
                    caption_mask_shape=list(cond['caption_mask'].shape))
    artifacts={}
    for mode,values in (('build',samples),('reload',samples[::-1])):
        path=workspace/'partition'/('source-'+mode+'.part')
        client.dump_bundle(client.pack_execution_values(values),path)
        artifacts[mode]=worker._upload_partition_artifact(path)['artifact_id']
    graph,_,_=workflow_graph('build')
    direct_path=workspace/'partition/direct-batch.part'
    graph['in_samples']['inputs']['artifact_path']=str(workspace/'partition/source-build.part')
    graph['out_batch']['inputs']['output_path']=str(direct_path)
    direct=submit(base,graph,timeout=300)
    values=read(direct_path)
    assert len(values)==1
    direct_check=check(values[0],'build')
    jobs=[]
    first=None
    for mode in ('build','restore','reload'):
        graph,inputs,outputs=workflow_graph(mode)
        boundary={'batch':first['output_artifacts']['batch']} if mode=='restore' else {'samples':artifacts[mode]}
        partition=dict(schema='comfy.partition.job.v1',partition_id='training-batch-'+mode,
                       workflow=graph,inputs=inputs,outputs=outputs)
        job=worker.queue.create(model='comfyui-partition-v1',input_path='artifacts://partition',
            request=dict(kind='comfyui-partition',partition=partition,input_artifacts=boundary,timeout_seconds=300))
        result=worker._run_comfyui_partition(job,executor)
        assert not result['files']
        path=workspace/'partition'/(mode+'-batch.part')
        worker._download_partition_artifact(result['output_artifacts']['batch'],path)
        values=read(path)
        assert len(values)==1
        checks=check(values[0],mode)
        # The client envelope can return the restored scalar without losing its batch dimension.
        roundtrip=workspace/'partition'/(mode+'-client-roundtrip.part')
        client.dump_bundle(client.pack_execution_values(values),roundtrip)
        assert unpack_execution_values(load_bundle(roundtrip))==values
        jobs.append(dict(job_id=job.id,prompt_id=result['prompt_id'],mode=mode,files_restored=0,
                         batch_artifacts_restored=1,checks=checks))
        if mode=='build':
            first=result
    return dict(direct_status=direct['status'],direct=direct_check,jobs=jobs,
        sample_count=len(samples),cache_encoder=cache['encoder'],cache_sha256=cache['sha256'],
        source_artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        source_samples=[s['identity_sha256'] for s in samples],
        scope='Actual CPU ComfyUI list collection and partition handlers, batch capture/restore/reordered sample collection and client codec retrieval; no training/provider/container/coordinator discovery or injected worker cancellation')


def verify(args):
    from tools.dataset_workflow import verify as server_verify
    previous=os.environ.get('COMFY_PARTITION_ROOT')
    os.environ['COMFY_PARTITION_ROOT']=str(args.workdir.resolve()/'partition')
    try:
        server_verify(args,graph_builder=workflow_graph,check_runner=run_checks)
    finally:
        if previous is None:
            os.environ.pop('COMFY_PARTITION_ROOT',None)
        else:
            os.environ['COMFY_PARTITION_ROOT']=previous


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comfy-root','cloud-root','client-root','python','samples-dir','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    verify(parser.parse_args())
