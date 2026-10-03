"""Actual worker sampling and paired-list restoration on an isolated server."""

import base64
import copy
import importlib.util
import os
from pathlib import Path
import sys


def batch_boundaries():
    return [{'key': 'motions', 'type_name': 'UNIMATE_MOTION'},
            {'key': 'rigs', 'type_name': 'UNIMATE_RIG'}]


def capture_nodes(graph, sources):
    for item in batch_boundaries():
        key = item['key']
        graph['out_' + key] = {'class_type': 'CloudPartitionOutput', 'inputs': {
            'value': sources[key], 'boundary_key': key, 'output_path': '', 'type_name': item['type_name']}}


def restored_batch_graph():
    graph = {f'in_{item["key"]}': {'class_type': 'CloudPartitionInput', 'inputs': {
        'boundary_key': item['key'], 'artifact_path': '', 'type_name': item['type_name']}}
        for item in batch_boundaries()}
    capture_nodes(graph, {'motions': ['in_motions', 0], 'rigs': ['in_rigs', 0]})
    graph['export'] = {'class_type': 'UniMateExportGLB', 'inputs': {
        'rig': ['in_rigs', 0], 'motion': ['in_motions', 0], 'filename_prefix': 'verified/restored-batch'}}
    return graph


def run_batch_worker_workflows(base, graph, workspace, cloud_root, client_root, bundle):
    sys.path.insert(0, str(cloud_root.resolve()))
    from cloud_offload.assets import resolve_partition_assets
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker, sha256_file
    from cloud_offload.partition_protocol import load_bundle, unpack_execution_values
    from tools.verify_workflow import bundle_inventory
    from unimate_pack.contracts import validate_motion, validate_rig
    import folder_paths

    spec = importlib.util.spec_from_file_location('batch_cloud_client', client_root / 'client.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    config = CloudConfig(storage_path=str(workspace / 'worker-store'), queue_db_path=str(workspace / 'worker-queue.db'))
    worker = Worker.__new__(Worker)
    worker.queue, worker.storage = JobQueue(config.queue_db_path), LocalStorage(config.storage_path)
    executor = ComfyUIWorkflowExecutor(base_url=base)
    jobs, entries, declared = [], [], []
    previous_output = folder_paths.get_output_directory()
    previous_root = os.environ.get('COMFY_PARTITION_ROOT')
    os.environ['COMFY_PARTITION_ROOT'] = str(workspace / 'partition')
    folder_paths.set_output_directory(str(workspace / 'client-restored'))
    try:
        # Only owned fixture copies/hard links are removed; source artifacts stay intact.
        staged = [(path, '__input__') for path in sorted((workspace / 'input').glob('*.glb'))]
        staged.append((bundle, 'unimate'))
        for path, category in staged:
            assert path.resolve().is_relative_to(workspace.resolve())
            artifact = worker._upload_partition_artifact(path)
            asset = {'category': category, 'filename': path.name, 'sha256': artifact['artifact_id'],
                     'size': artifact['size'], 'format': 'other'}
            resolved, missing = resolve_partition_assets(config, [asset], {}, worker.storage)
            assert not missing and len(resolved) == 1
            path.unlink()
            worker._stage_declared_asset(resolved[0], workspace / 'models', None)
            assert sha256_file(path) == artifact['artifact_id']
            declared.append(asset)

        def execute(workflow, inputs, artifacts):
            partition = {'schema': 'comfy.partition.job.v1', 'partition_id': f'batch-{len(jobs)}',
                'workflow': workflow, 'inputs': inputs, 'outputs': batch_boundaries(), 'assets': declared}
            job = worker.queue.create(model='comfyui-partition-v1', input_path='artifacts://partition',
                request={'kind': 'comfyui-partition', 'partition': partition,
                         'input_artifacts': artifacts, 'timeout_seconds': 7200})
            result = worker._run_comfyui_partition(job, executor)
            history = executor._request('GET', '/history/' + result['prompt_id']).json()[result['prompt_id']]
            assert history['status']['status_str'] == 'success' and history['status']['completed']
            ui = client.restore_partition_files({**result, 'job_id': job.id})
            assert ui.get('3d') and ui.get('files')
            root = Path(folder_paths.get_output_directory()) / 'cloud_offload' / job.id
            for file in result['files']:
                assert (root / file.get('subfolder', '') / file['filename']).read_bytes() == base64.b64decode(file['data'], validate=True)
            jobs.append({'job_id': job.id, 'partition': partition, 'prompt_id': result['prompt_id'],
                         'output_artifacts': result['output_artifacts'], 'status': history['status']})
            entries.append(history)
            return result

        capture = copy.deepcopy(graph)
        capture_nodes(capture, {'motions': ['4', 0], 'rigs': ['4', 1]})
        first = execute(capture, [], {})
        paths, values = {}, {}
        for key, artifact in first['output_artifacts'].items():
            paths[key] = workspace / 'partition' / ('batch-' + key + '.part')
            worker._download_partition_artifact(artifact, paths[key])
            values[key] = unpack_execution_values(load_bundle(paths[key]))
        assert len(values['rigs']) == len(values['motions']) == 8
        for rig, motion in zip(values['rigs'], values['motions']):
            validate_rig(rig)
            validate_motion(motion, rig)
        assert len({rig['rig_id'] for rig in values['rigs']}) == 2
        second = execute(restored_batch_graph(), batch_boundaries(), first['output_artifacts'])
        for key, artifact in second['output_artifacts'].items():
            restored = workspace / 'partition' / ('restored-batch-' + key + '.part')
            worker._download_partition_artifact(artifact, restored)
            assert bundle_inventory(restored) == bundle_inventory(paths[key]), key
        return entries, {'jobs': jobs, 'staged_assets': declared, 'paired_cases_verified': 8,
                         'scope': 'Declared asset staging, actual partition handler, sampling, storage and client restoration; no provider scheduling or deployed container'}
    finally:
        folder_paths.set_output_directory(previous_output)
        if previous_root is None:
            os.environ.pop('COMFY_PARTITION_ROOT', None)
        else:
            os.environ['COMFY_PARTITION_ROOT'] = previous_root
