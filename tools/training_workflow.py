"""Actual headless text/cache/sample nodes and worker capture/restore."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

CASES=[dict(augmentation=operation,mode=mode,addition_policy='released')
       for operation in ('none','addition','addition_linear','removal','pooling','perturbation','random')
       for mode in ('tpos','first_frame')]
CASES += [dict(augmentation=operation,mode=mode,addition_policy='neutral_fk')
          for operation in ('addition','addition_linear') for mode in ('tpos','first_frame')]
BUNDLE_NAME='training-test.unimate'


def workflow_graph(mode):
    if mode not in ('build','restore','reload'):
        raise ValueError('Unknown training workflow mode')
    graph,inputs,outputs={},[],[]
    def boundary(key,kind):
        inputs.append(dict(key=key,type_name=kind))
        graph['in_'+key]=dict(class_type='CloudPartitionInput',inputs=dict(
            boundary_key=key,type_name=kind,artifact_path=''))
    boundary('dataset','UNIMATE_DATASET')
    boundary('statistics','UNIMATE_STATISTICS')
    if mode=='build':
        graph['model']=dict(class_type='UniMateModelLoader',inputs=dict(bundle=BUNDLE_NAME))
        graph['cache']=dict(class_type='UniMateBuildTextCache',inputs=dict(
            model=['model',0],dataset=['in_dataset',0],chunk_size=256))
    else:
        boundary('cache','UNIMATE_TEXT_CACHE')
    for key,kind in (('dataset','UNIMATE_DATASET'),('statistics','UNIMATE_STATISTICS'),('cache','UNIMATE_TEXT_CACHE')):
        outputs.append(dict(key=key,type_name=kind))
    for index,case in enumerate(CASES):
        key=f'sample_{index}'
        if mode=='reload':
            boundary(key,'UNIMATE_TRAINING_SAMPLE')
        else:
            graph[key]=dict(class_type='UniMatePrepareTrainingSample',inputs=dict(
                dataset=['in_dataset',0],statistics=['in_statistics',0],
                text_cache=['cache' if mode=='build' else 'in_cache',0],clip_id='train',
                max_motion_length=60,max_joints=16,augmentation_seed=17,crop_seed=11,
                start_idx=-1,realign_feature=True,ground_rest=True,
                embedding_policy='cached',max_freqs=8,workspace_mib=512,
                enable_addition=True,enable_removal=True,enable_pooling=True,enable_perturbation=True,
                **case))
        outputs.append(dict(key=key,type_name='UNIMATE_TRAINING_SAMPLE'))
    for item in outputs:
        key=item['key']
        source=('cache' if mode=='build' else 'in_cache') if key=='cache' else (
            'in_'+key if key in ('dataset','statistics') or mode=='reload' else key)
        graph['out_'+key]=dict(class_type='CloudPartitionOutput',inputs=dict(
            value=[source,0],boundary_key=key,type_name=item['type_name'],output_path=''))
    for key,node in (('dataset','UniMateSaveDataset'),('statistics','UniMateSaveStatistics')):
        graph['save_'+key]=dict(class_type=node,inputs={key:['in_'+key,0],
            'filename_prefix':'training/'+mode+'-'+key})
    return graph,inputs,outputs


def run_checks(base,args,workspace):
    import folder_paths
    from cloud_offload.assets import resolve_partition_assets
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.partition_protocol import dump_bundle,load_bundle,pack_execution_values,unpack_execution_values
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker
    from tools.verify_workflow import submit
    from unimate_pack.dataset_builder import build_dataset
    from unimate_pack.dataset_io import load_dataset
    from unimate_pack.statistics import dataset_statistics
    from unimate_pack.statistics_io import load_statistics
    from unimate_pack.training_dataset_samples import produce_training_sample
    from unimate_pack.training_sample_contracts import validate_training_sample
    from unimate_pack.training_text import validate_text_cache

    def read(path):
        return unpack_execution_values(load_bundle(path))
    rig,motion=read(args.rig)[0],read(args.motion)[0]
    dataset=build_dataset([rig],[motion],json.dumps([dict(id='train',caption='A character walks forward.')]),'objaverse')
    stats=dataset_statistics(dataset)
    config=CloudConfig(storage_path=str(workspace/'worker-store'),queue_db_path=str(workspace/'worker-queue.db'))
    worker=Worker.__new__(Worker)
    worker.queue,worker.storage=JobQueue(config.queue_db_path),LocalStorage(config.storage_path)
    executor=ComfyUIWorkflowExecutor(base_url=base)
    spec=importlib.util.spec_from_file_location('training_cloud_client',args.client_root/'client.py')
    client=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    previous_output=folder_paths.get_output_directory()
    previous_partition=os.environ.get('COMFY_PARTITION_ROOT')
    folder_paths.set_output_directory(str(workspace/'client-restored'))
    os.environ['COMFY_PARTITION_ROOT']=str(workspace/'partition')
    expected_samples=None
    direct_values=None
    jobs=[]

    def verify_values(values):
        nonlocal expected_samples
        assert values['dataset']==dataset and values['statistics']==stats
        validate_text_cache(values['cache'])
        if expected_samples is None:
            expected_samples=[produce_training_sample(dataset,stats,values['cache'],'train',
                max_motion_length=60,max_joints=16,augmentation_seed=17,crop_seed=11,
                enabled=('addition','removal','pooling','perturbation'),**case) for case in CASES]
        for index,expected in enumerate(expected_samples):
            key=f'sample_{index}'
            numeric=validate_training_sample(values[key],expected_provenance=expected['provenance'])
            assert numeric['motion'].shape[0]==60
            assert values[key]==expected
        if direct_values is not None:
            assert values==direct_values

    def execute(mode,artifacts):
        graph,inputs,outputs=workflow_graph(mode)
        partition=dict(schema='comfy.partition.job.v1',partition_id='training-'+mode,
                       workflow=graph,inputs=inputs,outputs=outputs)
        job=worker.queue.create(model='comfyui-partition-v1',input_path='artifacts://partition',
            request=dict(kind='comfyui-partition',partition=partition,input_artifacts=artifacts,timeout_seconds=600))
        result=worker._run_comfyui_partition(job,executor)
        ui=client.restore_partition_files({**result,'job_id':job.id})
        assert len(result['files'])==2 and len(ui['files'])==2
        values={}
        for key,artifact in result['output_artifacts'].items():
            path=workspace/'partition'/(mode+'-'+key+'.part')
            worker._download_partition_artifact(artifact,path)
            items=read(path)
            assert len(items)==1
            values[key]=items[0]
        verify_values(values)
        for file in result['files']:
            payload=base64.b64decode(file['data'],validate=True)
            path=Path(folder_paths.get_output_directory())/'cloud_offload'/job.id/file.get('subfolder','')/file['filename']
            assert path.read_bytes()==payload
            assert (load_dataset(payload) if path.suffix=='.unimatedata' else load_statistics(payload))==(
                dataset if path.suffix=='.unimatedata' else stats)
        jobs.append(dict(job_id=job.id,prompt_id=result['prompt_id'],mode=mode,
                         files_restored=len(result['files']),sample_count=len(CASES)))
        return result

    try:
        artifacts={}
        for key,value in (('dataset',dataset),('statistics',stats)):
            path=workspace/'partition'/('source-'+key+'.part')
            dump_bundle(pack_execution_values([value]),path)
            artifacts[key]=worker._upload_partition_artifact(path)['artifact_id']
        graph,_,outputs=workflow_graph('build')
        for key in ('dataset','statistics'):
            graph['in_'+key]['inputs']['artifact_path']=str(workspace/'partition'/('source-'+key+'.part'))
        for item in outputs:
            graph['out_'+item['key']]['inputs']['output_path']=str(workspace/'partition'/('direct-'+item['key']+'.part'))
        direct=submit(base,graph,timeout=600)
        direct_values={item['key']:read(workspace/'partition'/('direct-'+item['key']+'.part'))[0] for item in outputs}
        verify_values(direct_values)
        # Stage the actual model through runner asset resolution before build job.
        uploaded=worker._upload_partition_artifact(args.bundle)
        declared=dict(category='unimate',filename=BUNDLE_NAME,sha256=uploaded['artifact_id'],
                      size=args.bundle.stat().st_size,format='other')
        resolved,missing=resolve_partition_assets(config,[declared],{},worker.storage)
        assert not missing and len(resolved)==1
        target=workspace/'models/unimate'/BUNDLE_NAME
        target.unlink()
        worker._stage_declared_asset(resolved[0],workspace/'models',None)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==declared['sha256']
        first=execute('build',artifacts)
        portable={key:value for key,value in first['output_artifacts'].items() if key in ('dataset','statistics','cache')}
        execute('restore',portable)
        execute('reload',first['output_artifacts'])
        return dict(direct_status=direct['status'],jobs=jobs,cases=CASES,
            staged_model=declared,cache_encoder=direct_values['cache']['encoder'],
            cache_sha256=direct_values['cache']['sha256'],
            sample_identities=[value['identity_sha256'] for value in expected_samples],
            scope='Actual ComfyUI and runner handlers; prepared features, text/sample capture/restore, model staging and file retrieval; no training/provider/container execution')
    finally:
        folder_paths.set_output_directory(previous_output)
        if previous_partition is None:
            os.environ.pop('COMFY_PARTITION_ROOT',None)
        else:
            os.environ['COMFY_PARTITION_ROOT']=previous_partition


def verify(args):
    from tools.dataset_workflow import verify as server_verify
    def prepare(workspace):
        target=workspace/'models/unimate'/BUNDLE_NAME
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.bundle,target)
    server_verify(args,graph_builder=workflow_graph,check_runner=run_checks,prepare_workspace=prepare)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comfy-root','cloud-root','client-root','python','rig','motion','bundle','workdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    verify(parser.parse_args())
