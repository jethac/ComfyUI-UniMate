"""Headless trained-bundle assembly, partition retrieval and staged model reload."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def workflow_graph(mode):
    if mode not in ('build','restore','reload'):
        raise ValueError('Unknown trained bundle workflow mode')
    graph,inputs={},[]
    if mode=='reload':
        graph['model']=dict(class_type='UniMateModelLoader',inputs=dict(bundle='trained.unimate'))
    else:
        for key,kind in [('weights','UNIMATE_INFERENCE_WEIGHTS'),('statistics','UNIMATE_STATISTICS'),
            ('cache','UNIMATE_TEXT_CACHE'),('encoder','UNIMATE_MODEL')]:
            inputs.append(dict(key=key,type_name=kind))
            graph['in_'+key]=dict(class_type='CloudPartitionInput',inputs=dict(
                boundary_key=key,type_name=kind,artifact_path=''))
        graph['model']=dict(class_type='UniMateAssembleModel',inputs=dict(weights=['in_weights',0],
            statistics=['in_statistics',0],text_cache=['in_cache',0],encoder_model=['in_encoder',0],
            sampling='{}',filename_prefix='bundles/'+mode,workspace_mib=32768))
    outputs=[dict(key='model',type_name='UNIMATE_MODEL')]
    graph['out_model']=dict(class_type='CloudPartitionOutput',inputs=dict(
        value=['model',0],boundary_key='model',type_name='UNIMATE_MODEL',output_path=''))
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
    from unimate_pack.inference import load_model_bundle
    from unimate_pack.bundle import inspect_bundle
    from unimate_pack.trained_bundle import read_trained_components

    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    codec=module('trained_bundle_codec',args.client_root/'partition_protocol.py')
    client=module('trained_bundle_client',args.client_root/'client.py')
    def read(path):
        values=codec.unpack_execution_values(codec.load_bundle(path))
        assert len(values)==1
        return values[0]
    source={'weights':read(args.raw_weights),'statistics':read(args.statistics),
        'cache':read(args.cache),'encoder':load_model_bundle(args.encoder)}
    selected={'build':source['weights'],'restore':read(args.ema_weights)}
    assert selected['build']['weights']=='raw' and selected['restore']['weights']=='ema'
    config=CloudConfig(storage_path=str(workspace/'worker-store'),queue_db_path=str(workspace/'worker-queue.db'))
    worker=Worker.__new__(Worker)
    worker.queue,worker.storage=JobQueue(config.queue_db_path),LocalStorage(config.storage_path)
    executor=ComfyUIWorkflowExecutor(base_url=base)
    sources,artifacts={},{}
    for key,value in source.items():
        path=workspace/'partition'/('source-'+key+'.part')
        codec.dump_bundle(codec.pack_execution_values([value]),path)
        sources[key]=path
        artifacts[key]=worker._upload_partition_artifact(path)['artifact_id']
    path=workspace/'partition'/'ema-weights.part'
    codec.dump_bundle(codec.pack_execution_values([selected['restore']]),path)
    ema_artifact=worker._upload_partition_artifact(path)['artifact_id']
    baselines={}
    for mode in ('build','restore'):
        graph,_,_=workflow_graph(mode)
        for key,path in sources.items():
            graph['in_'+key]['inputs']['artifact_path']=str(path)
        if mode=='restore':
            graph['in_weights']['inputs']['artifact_path']=str(workspace/'partition'/'ema-weights.part')
        path=workspace/'partition'/('direct-'+mode+'.part')
        graph['out_model']['inputs']['output_path']=str(path)
        submit(base,graph,timeout=900)
        baselines[mode]=read(path)
        weights,statistics=read_trained_components(baselines[mode]['bundle'],max_workspace_bytes=32*1024**3)
        assert weights==selected[mode] and statistics==source['statistics']
    previous=folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(workspace/'client-restored'))
    jobs=[]
    try:
        for mode in ('build','restore','reload'):
            graph,inputs,outputs=workflow_graph(mode)
            partition=dict(schema='comfy.partition.job.v1',partition_id='trained-bundle-'+mode,
                workflow=graph,inputs=inputs,outputs=outputs)
            boundary_artifacts=dict(artifacts)
            if mode=='restore':
                boundary_artifacts['weights']=ema_artifact
            queued=worker.queue.create(model='comfyui-partition-v1',input_path='artifacts://partition',
                request=dict(kind='comfyui-partition',partition=partition,
                    input_artifacts={} if mode=='reload' else boundary_artifacts,timeout_seconds=900))
            result=worker._run_comfyui_partition(queued,executor)
            ui=client.restore_partition_files({**result,'job_id':queued.id})
            count=0 if mode=='reload' else 1
            assert len(result['files'])==len(ui.get('files',[]))==count
            path=workspace/'partition'/(mode+'-model.part')
            worker._download_partition_artifact(result['output_artifacts']['model'],path)
            value=read(path)
            expected=baselines['restore' if mode=='restore' else 'build']
            assert value['bundle']==expected['bundle'] and value['sha256']==expected['sha256']
            manifest=inspect_bundle(value['bundle'],max_workspace_bytes=32*1024**3)
            assert manifest['weights']==selected['restore' if mode=='restore' else 'build']['weights']
            if count:
                descriptor=result['files'][0]
                data=base64.b64decode(descriptor['data'],validate=True)
                restored=Path(folder_paths.get_output_directory())/'cloud_offload'/queued.id/descriptor.get('subfolder','')/descriptor['filename']
                assert restored.read_bytes()==data==expected['bundle']
                if mode=='build':
                    staged=workspace/'models'/'unimate'/'trained.unimate'
                    staged.parent.mkdir(parents=True,exist_ok=True)
                    staged.write_bytes(data)
                    uploaded=worker._upload_partition_artifact(staged)
                    declaration=dict(category='unimate',filename=staged.name,
                        sha256=uploaded['artifact_id'],size=len(data),format='other')
                    resolved,missing=resolve_partition_assets(config,[declaration],{},worker.storage)
                    assert not missing and len(resolved)==1
                    staged.unlink()
                    worker._stage_declared_asset(resolved[0],workspace/'models',None)
                    assert staged.read_bytes()==data
            jobs.append(dict(job_id=queued.id,prompt_id=result['prompt_id'],mode=mode,
                weights=manifest['weights'],bundle_sha256=value['sha256'],
                checkpoint_sha256=manifest['model_revision'],files_restored=count))
        return dict(jobs=jobs,scope='Actual CPU ComfyUI raw/EMA trained bundle assembly, numeric component equality, portable worker outputs, model staging/reload and client file retrieval. No generation/playback claim.')
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
    for name in ('comfy-root','cloud-root','client-root','python','raw-weights','ema-weights',
        'statistics','cache','encoder','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    verify(parser.parse_args())
