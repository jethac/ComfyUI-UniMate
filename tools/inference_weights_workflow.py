"""Headless selected-weight export, managed reload and partition retrieval."""
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
        raise ValueError('Unknown weight export mode')
    graph,inputs={},[]
    if mode=='reload':
        graph['weights']=dict(class_type='UniMateLoadInferenceWeights',inputs=dict(
            archive='selected.unimateweights',workspace_mib=32768))
    else:
        inputs=[dict(key='checkpoint',type_name='UNIMATE_TRAINING_CHECKPOINT')]
        graph['in_checkpoint']=dict(class_type='CloudPartitionInput',inputs=dict(
            boundary_key='checkpoint',type_name='UNIMATE_TRAINING_CHECKPOINT',artifact_path=''))
        graph['weights']=dict(class_type='UniMateExportInferenceWeights',inputs=dict(
            checkpoint=['in_checkpoint',0],weights='raw' if mode=='build' else 'ema',
            filename_prefix='weights/'+mode,workspace_mib=32768))
    outputs=[dict(key='weights',type_name='UNIMATE_INFERENCE_WEIGHTS')]
    graph['out_weights']=dict(class_type='CloudPartitionOutput',inputs=dict(
        value=['weights',0],boundary_key='weights',type_name='UNIMATE_INFERENCE_WEIGHTS',output_path=''))
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
    from unimate_pack.inference_weights import load_inference_weights

    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    codec=module('weights_workflow_codec',args.client_root/'partition_protocol.py')
    client=module('weights_workflow_client',args.client_root/'client.py')
    def read(path):
        items=codec.unpack_execution_values(codec.load_bundle(path))
        assert len(items)==1
        return items[0]
    checkpoint=read(args.checkpoint)
    config=CloudConfig(storage_path=str(workspace/'worker-store'),queue_db_path=str(workspace/'worker-queue.db'))
    worker=Worker.__new__(Worker)
    worker.queue,worker.storage=JobQueue(config.queue_db_path),LocalStorage(config.storage_path)
    executor=ComfyUIWorkflowExecutor(base_url=base)
    source=workspace/'partition'/'source-checkpoint.part'
    codec.dump_bundle(codec.pack_execution_values([checkpoint]),source)
    artifact=worker._upload_partition_artifact(source)['artifact_id']
    baselines={}
    for mode in ('build','restore'):
        graph,_,_=workflow_graph(mode)
        graph['in_checkpoint']['inputs']['artifact_path']=str(source)
        path=workspace/'partition'/('direct-'+mode+'.part')
        graph['out_weights']['inputs']['output_path']=str(path)
        submit(base,graph,timeout=600)
        baselines[mode]=read(path)
        assert baselines[mode]['weights']==('raw' if mode=='build' else 'ema')
        assert baselines[mode]['source_checkpoint_sha256']==checkpoint['identity_sha256']
    previous=folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(workspace/'client-restored'))
    jobs=[]
    try:
        for mode in ('build','restore','reload'):
            graph,inputs,outputs=workflow_graph(mode)
            partition=dict(schema='comfy.partition.job.v1',partition_id='inference-weights-'+mode,
                workflow=graph,inputs=inputs,outputs=outputs)
            queued=worker.queue.create(model='comfyui-partition-v1',input_path='artifacts://partition',
                request=dict(kind='comfyui-partition',partition=partition,
                    input_artifacts={} if mode=='reload' else {'checkpoint':artifact},timeout_seconds=600))
            result=worker._run_comfyui_partition(queued,executor)
            ui=client.restore_partition_files({**result,'job_id':queued.id})
            count=0 if mode=='reload' else 1
            assert len(result['files'])==len(ui.get('files',[]))==count
            path=workspace/'partition'/(mode+'-weights.part')
            worker._download_partition_artifact(result['output_artifacts']['weights'],path)
            value=read(path)
            expected=baselines['restore' if mode=='restore' else 'build']
            assert value==expected
            if count:
                descriptor=result['files'][0]
                data=base64.b64decode(descriptor['data'],validate=True)
                restored=Path(folder_paths.get_output_directory())/'cloud_offload'/queued.id/descriptor.get('subfolder','')/descriptor['filename']
                assert restored.read_bytes()==data
                assert load_inference_weights(data,max_workspace_bytes=32*1024**3)==expected
                if mode=='build':
                    staged=workspace/'input'/'selected.unimateweights'
                    staged.write_bytes(data)
                    uploaded=worker._upload_partition_artifact(staged)
                    declaration=dict(category='__input__',filename=staged.name,
                        sha256=uploaded['artifact_id'],size=len(data),format='other')
                    resolved,missing=resolve_partition_assets(config,[declaration],{},worker.storage)
                    assert not missing and len(resolved)==1
                    staged.unlink()
                    worker._stage_declared_asset(resolved[0],workspace/'models',None)
                    assert staged.read_bytes()==data
            jobs.append(dict(job_id=queued.id,prompt_id=result['prompt_id'],mode=mode,
                weights=value['weights'],identity_sha256=value['identity_sha256'],files_restored=count))
        return dict(jobs=jobs,source_checkpoint_sha256=checkpoint['identity_sha256'],
            scope='Actual CPU ComfyUI raw/EMA export, portable partition outputs, declared input reload and client file retrieval. No trained bundle assembly or sampling claim.')
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
    for name in ('comfy-root','cloud-root','client-root','python','checkpoint','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    verify(parser.parse_args())
