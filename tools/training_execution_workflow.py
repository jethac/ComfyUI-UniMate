"""Headless training, checkpoint file staging and real partition retrieval."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def workflow_graph(mode,initialized=False):
    if mode not in ('build','restore','reload'):
        raise ValueError('Unknown training execution mode')
    graph,inputs={},[]
    def boundary(key,kind):
        inputs.append(dict(key=key,type_name=kind))
        graph['in_'+key]=dict(class_type='CloudPartitionInput',inputs=dict(
            boundary_key=key,type_name=kind,artifact_path=''))
    for key,kind in (('dataset','UNIMATE_DATASET'),('statistics','UNIMATE_STATISTICS'),('cache','UNIMATE_TEXT_CACHE')):
        boundary(key,kind)
    options=dict(model=dict(latent_dim=64,ff_size=128,num_layers=1,num_heads=4,
        max_motion_length=8,max_joints=16,max_depth=32,cond_mode='text',dropout=.1,cond_mask_prob=.1),
        optimizer=dict(num_steps=4,gradient_accumulation_steps=2),
        loss=dict(lambda_geo=0.,lambda_smooth=0.),seed=17,batch_size=1)
    if initialized:
        options.pop('model')
        boundary('initialization','UNIMATE_MODEL')
    graph['job']=dict(class_type='UniMateTrainingJob',inputs=dict(dataset=['in_dataset',0],
        statistics=['in_statistics',0],text_cache=['in_cache',0],options=json.dumps(options),workspace_mib=512))
    train=dict(job=['job',0],dataset=['in_dataset',0],statistics=['in_statistics',0],
        text_cache=['in_cache',0],updates=1 if mode=='build' else 2,workspace_mib=8192)
    if initialized:
        graph['job']['inputs'].update(initialization=['in_initialization',0],workspace_mib=32768)
        train.update(initialization=['in_initialization',0],workspace_mib=32768)
    if mode=='restore':
        boundary('checkpoint','UNIMATE_TRAINING_CHECKPOINT')
        train['checkpoint']=['in_checkpoint',0]
    elif mode=='reload':
        graph['load']=dict(class_type='UniMateLoadTrainingCheckpoint',inputs=dict(archive='resume.unimatetrain'))
        train['checkpoint']=['load',0]
    graph['train']=dict(class_type='UniMateTrain',inputs=train)
    graph['save']=dict(class_type='UniMateSaveTrainingCheckpoint',inputs=dict(
        checkpoint=['train',0],filename_prefix='training/'+mode))
    outputs=[dict(key='checkpoint',type_name='UNIMATE_TRAINING_CHECKPOINT'),dict(key='progress',type_name='STRING')]
    for index,item in enumerate(outputs):
        graph['out_'+item['key']]=dict(class_type='CloudPartitionOutput',inputs=dict(
            value=['train',index],boundary_key=item['key'],type_name=item['type_name'],output_path=''))
    return graph,inputs,outputs


def run_checks(base,args,workspace):
    import folder_paths
    from cloud_offload.assets import resolve_partition_assets
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker
    from tools.verify_workflow import submit
    from unimate_pack.training_checkpoint_io import load_training_checkpoint
    from unimate_pack.training_text import validate_text_cache

    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        result=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    codec=module('training_execution_codec',args.client_root/'partition_protocol.py')
    client=module('training_execution_client',args.client_root/'client.py')
    def read(path):
        items=codec.unpack_execution_values(codec.load_bundle(path))
        assert len(items)==1
        return items[0]
    source={key:read(args.source_dir/('direct-'+key+'.part')) for key in ('dataset','statistics','cache')}
    initialized=getattr(args,'initialization_bundle',None) is not None
    if initialized:
        from unimate_pack.inference import load_model_bundle
        source['initialization']=load_model_bundle(args.initialization_bundle)
    validate_text_cache(source['cache'])
    config=CloudConfig(storage_path=str(workspace/'worker-store'),queue_db_path=str(workspace/'worker-queue.db'))
    worker=Worker.__new__(Worker)
    worker.queue,worker.storage=JobQueue(config.queue_db_path),LocalStorage(config.storage_path)
    executor=ComfyUIWorkflowExecutor(base_url=base)
    artifacts={}
    for key,value in source.items():
        path=workspace/'partition'/('source-'+key+'.part')
        codec.dump_bundle(codec.pack_execution_values([value]),path)
        artifacts[key]=worker._upload_partition_artifact(path)['artifact_id']
    def direct(updates,label):
        graph,_,outputs=workflow_graph('build',initialized)
        graph['train']['inputs']['updates']=updates
        graph['save']['inputs']['filename_prefix']='training/direct-'+label
        for key in source:
            graph['in_'+key]['inputs']['artifact_path']=str(workspace/'partition'/('source-'+key+'.part'))
        for item in outputs:
            graph['out_'+item['key']]['inputs']['output_path']=str(workspace/'partition'/('direct-'+label+'-'+item['key']+'.part'))
        history=submit(base,graph,timeout=600)
        checkpoint=read(workspace/'partition'/('direct-'+label+'-checkpoint.part'))
        progress=json.loads(read(workspace/'partition'/('direct-'+label+'-progress.part')))
        assert progress['updates']==updates
        return checkpoint,history['status']
    first,direct_status=direct(1,'first')
    full,_=direct(3,'full')
    previous=folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(workspace/'client-restored'))
    jobs=[]
    try:
        initial=None
        for mode in ('build','restore','reload'):
            graph,inputs,outputs=workflow_graph(mode,initialized)
            boundary=dict(artifacts)
            if mode=='restore':
                boundary['checkpoint']=initial['output_artifacts']['checkpoint']
            partition=dict(schema='comfy.partition.job.v1',partition_id='training-execution-'+mode,
                workflow=graph,inputs=inputs,outputs=outputs)
            queued=worker.queue.create(model='comfyui-partition-v1',input_path='artifacts://partition',
                request=dict(kind='comfyui-partition',partition=partition,input_artifacts=boundary,timeout_seconds=600))
            result=worker._run_comfyui_partition(queued,executor)
            ui=client.restore_partition_files({**result,'job_id':queued.id})
            assert len(result['files'])==len(ui['files'])==1
            captured={}
            for key,artifact in result['output_artifacts'].items():
                path=workspace/'partition'/(mode+'-'+key+'.part')
                worker._download_partition_artifact(artifact,path)
                captured[key]=read(path)
            expected=first if mode=='build' else full
            assert captured['checkpoint']==expected
            progress=json.loads(captured['progress'])
            assert progress['updates']==(1 if mode=='build' else 3)
            descriptor=result['files'][0]
            data=base64.b64decode(descriptor['data'],validate=True)
            restored=Path(folder_paths.get_output_directory())/'cloud_offload'/queued.id/descriptor.get('subfolder','')/descriptor['filename']
            assert restored.read_bytes()==data
            assert load_training_checkpoint(data)==expected
            if mode=='build':
                initial=result
                path=workspace/'input'/'resume.unimatetrain'
                path.write_bytes(data)
                uploaded=worker._upload_partition_artifact(path)
                declaration=dict(category='__input__',filename=path.name,sha256=uploaded['artifact_id'],
                    size=len(data),format='other')
                resolved,missing=resolve_partition_assets(config,[declaration],{},worker.storage)
                assert not missing and len(resolved)==1
                path.unlink()
                worker._stage_declared_asset(resolved[0],workspace/'models',None)
                assert path.read_bytes()==data
            jobs.append(dict(job_id=queued.id,prompt_id=result['prompt_id'],mode=mode,
                files_restored=1,checkpoint_sha256=expected['identity_sha256'],progress=progress))
        from unimate_pack.bundle import inspect_bundle
        selection=dict(bundle_sha256=source['initialization']['sha256'],
            weights=inspect_bundle(source['initialization']['bundle'])['weights']) if initialized else None
        return dict(direct_status=direct_status,jobs=jobs,cache_encoder=source['cache']['encoder'],
            cache_sha256=source['cache']['sha256'],initialization=selection,
            scope=('Actual CPU ComfyUI server, prepared-data '+('installed-weight' if initialized else 'scratch')+
                ' flow training, checkpoint socket/file resume, declared input staging and actual partition handler/client artifact retrieval. '+
                'No provider/container/live coordinator deployment, trained inference export, distributed/unbalanced loader or injected server cancellation.'))
    finally:
        folder_paths.set_output_directory(previous)


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
    for name in ('comfy-root','cloud-root','client-root','python','source-dir','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--initialization-bundle',type=Path)
    verify(parser.parse_args())
