"""Real local Cloud Offload partition execution for preprocessing workflows."""

import base64
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys


BOUNDARIES = {
    'asset': ('canonical', 'UNIMATE_ASSET'), 'rig': ('2', 'UNIMATE_RIG'),
    'motion': ('4', 'UNIMATE_MOTION'), 'conditioning': ('conditioning', 'UNIMATE_CONDITIONING'),
    'skeleton_fk': ('recover_fk', 'UNIMATE_SKELETON'),
    'skeleton_ric': ('recover_ric', 'UNIMATE_SKELETON'),
    'image_fk': ('preview_fk', 'IMAGE'), 'image_ric': ('preview_ric', 'IMAGE'),
}


def add_capture_nodes(graph):
    graph['export_fbx'] = {'class_type': 'UniMateExportFBX', 'inputs': {
        'rig': ['2', 0], 'motion': ['4', 0], 'filename_prefix': 'verified/capture-fbx'}}
    graph['canonical'] = {'class_type': 'UniMateCanonicalAsset', 'inputs': {'rig': ['2', 0]}}
    graph['conditioning'] = {'class_type': 'UniMateRigConditioning', 'inputs': {'rig': ['2', 0]}}
    outputs = []
    for key, (node, kind) in BOUNDARIES.items():
        graph['out_' + key] = {'class_type': 'CloudPartitionOutput', 'inputs': {
            'value': [node, 0], 'boundary_key': key, 'output_path': '', 'type_name': kind}}
        outputs.append({'key': key, 'type_name': kind})
    return outputs


def restored_graph(boundaries):
    graph = {}
    for boundary in boundaries:
        key, kind = boundary['key'], boundary['type_name']
        graph['in_' + key] = {'class_type': 'CloudPartitionInput', 'inputs': {
            'boundary_key': key, 'artifact_path': '', 'type_name': kind}}
        graph['out_' + key] = {'class_type': 'CloudPartitionOutput', 'inputs': {
            'boundary_key': key, 'output_path': '', 'type_name': kind, 'value': ['in_' + key, 0]}}
    graph['canonical_prepare'] = {'class_type': 'UniMatePrepareRig',
        'inputs': {'asset': ['in_asset', 0], 'facing': '+Z'}}
    graph['canonical_conditioning'] = {'class_type': 'UniMateRigConditioning',
        'inputs': {'rig': ['canonical_prepare', 0]}}
    graph['out_canonical_conditioning'] = {'class_type': 'CloudPartitionOutput', 'inputs': {
        'value': ['canonical_conditioning', 0], 'boundary_key': 'canonical_conditioning',
        'output_path': '', 'type_name': 'UNIMATE_CONDITIONING'}}
    graph['animated_load'] = {'class_type': 'UniMateLoadRig',
        'inputs': {'asset': 'animated reference.glb'}}
    graph['animated_prepare'] = {'class_type': 'UniMatePrepareRig',
        'inputs': {'asset': ['animated_load', 0], 'facing': '+Z'}}
    graph['extract'] = {'class_type': 'UniMateExtractMotion',
        'inputs': {'rig': ['animated_prepare', 0], 'clip_index': 0}}
    graph['out_extracted'] = {'class_type': 'CloudPartitionOutput', 'inputs': {
        'value': ['extract', 0], 'boundary_key': 'extracted',
        'output_path': '', 'type_name': 'UNIMATE_MOTION'}}
    for key, source in (('canonical_rig', 'canonical_prepare'), ('extracted_rig', 'animated_prepare')):
        graph['out_' + key] = {'class_type': 'CloudPartitionOutput', 'inputs': {
            'value': [source, 0], 'boundary_key': key, 'output_path': '', 'type_name': 'UNIMATE_RIG'}}
    for method in ('fk', 'ric'):
        graph['render_' + method] = {'class_type': 'UniMatePreviewSkeleton', 'inputs': {
            'skeleton': ['in_skeleton_' + method, 0], 'projection': 'front', 'resolution': 128}}
        for source in ('render', 'in_image'):
            graph[f'save_{source}_{method}'] = {'class_type': 'SaveImage', 'inputs': {
                'images': [source + '_' + method, 0],
                'filename_prefix': f'verified/worker-{source}-{method}'}}
    graph['export'] = {'class_type': 'UniMateExportGLB', 'inputs': {
        'rig': ['in_rig', 0], 'motion': ['in_motion', 0], 'filename_prefix': 'verified/worker-restored'}}
    graph['export_fbx'] = {'class_type': 'UniMateExportFBX', 'inputs': {
        'rig': ['in_rig', 0], 'motion': ['in_motion', 0], 'filename_prefix': 'verified/restored-fbx'}}
    return graph


def run_worker_workflows(base, graph, workspace, cloud_root, client_root):
    sys.path.insert(0, str(cloud_root.resolve()))
    from cloud_offload.assets import resolve_partition_assets
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker
    from cloud_offload.partition_protocol import load_bundle
    from tools.verify_workflow import bundle_inventory
    from unimate_pack.contracts import validate_asset, validate_rig, validate_motion, validate_skeleton
    import folder_paths

    spec = importlib.util.spec_from_file_location('unimate_cloud_client', client_root / 'client.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    config = CloudConfig(storage_path=str(workspace / 'worker-store'),
                         queue_db_path=str(workspace / 'worker-queue.db'))
    worker = Worker.__new__(Worker)
    worker.queue, worker.storage = JobQueue(config.queue_db_path), LocalStorage(config.storage_path)
    executor = ComfyUIWorkflowExecutor(base_url=base)
    previous_root = os.environ.get('COMFY_PARTITION_ROOT')
    previous_output = folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(workspace / 'client-restored'))
    os.environ['COMFY_PARTITION_ROOT'] = str(workspace / 'partition')
    jobs, entries = [], []

    def stage(path):
        data = path.read_bytes()
        artifact = worker._upload_partition_artifact(path)
        declared = {'category': '__input__', 'filename': path.name,
                    'sha256': artifact['artifact_id'], 'size': len(data), 'format': 'other'}
        resolved, missing = resolve_partition_assets(config, [declared], {}, worker.storage)
        assert not missing and len(resolved) == 1
        assert path.resolve().is_relative_to((workspace / 'input').resolve())
        path.unlink()
        worker._stage_declared_asset(resolved[0], workspace / 'models', None)
        assert path.read_bytes() == data
        return declared

    def execute(workflow, inputs, outputs, artifacts):
        partition = {'schema': 'comfy.partition.job.v1', 'partition_id': f'unimate-{len(jobs)}',
                     'workflow': workflow, 'inputs': inputs, 'outputs': outputs}
        job = worker.queue.create(model='comfyui-partition-v1', input_path='artifacts://partition',
            request={'kind': 'comfyui-partition', 'partition': partition,
                     'input_artifacts': artifacts, 'timeout_seconds': 300})
        result = worker._run_comfyui_partition(job, executor)
        history = executor._request('GET', '/history/' + result['prompt_id']).json()[result['prompt_id']]
        assert history['status']['status_str'] == 'success' and history['status']['completed']
        ui = client.restore_partition_files({**result, 'job_id': job.id})
        restored_root = Path(folder_paths.get_output_directory())
        for file in result['files']:
            path = restored_root / 'cloud_offload' / job.id / file.get('subfolder', '') / file['filename']
            assert path.read_bytes() == base64.b64decode(file['data'], validate=True)
        assert ui.get('images'), 'Partition image outputs were lost during worker/client handoff'
        jobs.append({'job_id': job.id, 'partition': partition, 'prompt_id': result['prompt_id'],
                     'output_artifacts': result['output_artifacts'], 'files_restored': len(result['files']),
                     'status': history['status']})
        entries.append(history)
        return result

    try:
        staged = [stage(path) for path in sorted((workspace / 'input').iterdir()) if path.is_file()]
        capture = copy.deepcopy(graph)
        boundaries = add_capture_nodes(capture)
        first = execute(capture, [], boundaries, {})
        paths = {}
        for key, artifact in first['output_artifacts'].items():
            paths[key] = workspace / 'partition' / f'captured-{key}.part'
            worker._download_partition_artifact(artifact, paths[key])
        values = {key: load_bundle(path) for key, path in paths.items()}
        validate_asset(values['asset'])
        validate_rig(values['rig'])
        validate_motion(values['motion'], values['rig'])
        for method in ('fk', 'ric'):
            validate_skeleton(values['skeleton_' + method], values['rig']['rig_id'])
        assert values['conditioning']['arrays'] == values['rig']['conditioning']
        generated = next(file for file in first['files'] if file['filename'].endswith('.glb'))
        animated = workspace / 'input/animated reference.glb'
        animated.write_bytes(base64.b64decode(generated['data'], validate=True))
        staged.append(stage(animated))
        restored = restored_graph(boundaries)
        second_outputs = [*boundaries, {'key': 'canonical_conditioning', 'type_name': 'UNIMATE_CONDITIONING'},
                          {'key': 'extracted', 'type_name': 'UNIMATE_MOTION'},
                          {'key': 'canonical_rig', 'type_name': 'UNIMATE_RIG'},
                          {'key': 'extracted_rig', 'type_name': 'UNIMATE_RIG'}]
        second = execute(restored, boundaries, second_outputs, first['output_artifacts'])
        for key in first['output_artifacts']:
            path = workspace / 'partition' / f'round-{key}.part'
            worker._download_partition_artifact(second['output_artifacts'][key], path)
            assert bundle_inventory(path) == bundle_inventory(paths[key]), key
        extracted_path = workspace / 'partition/extracted.part'
        worker._download_partition_artifact(second['output_artifacts']['extracted'], extracted_path)
        extracted = load_bundle(extracted_path)
        rigs = {}
        for key in ('canonical_rig', 'extracted_rig', 'canonical_conditioning'):
            path = workspace / 'partition' / f'{key}.part'
            worker._download_partition_artifact(second['output_artifacts'][key], path)
            rigs[key] = load_bundle(path)
        validate_rig(rigs['canonical_rig'])
        validate_rig(rigs['extracted_rig'])
        validate_motion(extracted, rigs['extracted_rig'])
        assert rigs['canonical_rig']['asset']['sha256'] == values['asset']['sha256']
        assert rigs['canonical_conditioning']['rig_id'] == rigs['canonical_rig']['rig_id']
        assert rigs['canonical_conditioning']['arrays'] == rigs['canonical_rig']['conditioning']
        assert rigs['extracted_rig']['asset']['sha256'] == hashlib.sha256(animated.read_bytes()).hexdigest()
        from unimate_pack.contracts import decode_arrays
        assert len(decode_arrays(extracted['features'])['features']) == len(values['image_fk']) - 1
        return entries, {'jobs': jobs, 'staged_assets': staged,
                         'boundary_types_verified': sorted({item['type_name'] for item in boundaries}),
                         'scope': 'Real partition handler, staging, execution, artifact storage and client restoration; no provider scheduling or container deployment'}
    finally:
        folder_paths.set_output_directory(previous_output)
        if previous_root is None:
            os.environ.pop('COMFY_PARTITION_ROOT', None)
        else:
            os.environ['COMFY_PARTITION_ROOT'] = previous_root
