"""Headless dataset nodes through real ComfyUI and Cloud Offload handlers."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = list(itertools.product((False, True), repeat=3))
LABELS = [dict(id='train-a', dataset_type='objaverse', object_type='shared', split='train'),
          dict(id='train-a2', dataset_type='objaverse', object_type='shared', split='train'),
          dict(id='train-other', dataset_type='objaverse', object_type='other', split='train'),
          dict(id='train-b', dataset_type='mixamo', object_type='shared', split='train'),
          dict(id='evaluation', dataset_type='objaverse', object_type='evaluation', split='train')]
SPLIT_OPTIONS = json.dumps(dict(explicit_eval_objects=dict(objaverse=['evaluation'])))
SAMPLING_MODES = [(.5, None, 0), (1, None, 3), (.5, .25, 0), (1, 1, 3)]


def workflow_graph(mode):
    graph, inputs = {}, []
    def boundary(key, kind):
        inputs.append(dict(key=key, type_name=kind))
        graph['in_' + key] = dict(class_type='CloudPartitionInput', inputs=dict(
            boundary_key=key, type_name=kind, artifact_path=''))
    if mode == 'build':
        boundary('rig', 'UNIMATE_RIG')
        boundary('motion', 'UNIMATE_MOTION')
        graph['dataset_build'] = dict(class_type='UniMateBuildDataset', inputs=dict(
            rigs=['in_rig', 0], motions=['in_motion', 0], labels=json.dumps(LABELS),
            default_dataset='objaverse'))
        graph['dataset'] = dict(class_type='UniMateSplitDataset', inputs=dict(
            dataset=['dataset_build', 0], ratio=0, seed=17, options=SPLIT_OPTIONS))
    elif mode == 'restore':
        for key, kind in (('dataset', 'UNIMATE_DATASET'), ('statistics', 'UNIMATE_STATISTICS')):
            boundary(key, kind)
    elif mode == 'reload':
        for key, node, suffix in (('dataset', 'UniMateLoadDataset', 'unimatedata'),
                                  ('statistics', 'UniMateLoadStatistics', 'unimatestats')):
            graph[key] = dict(class_type=node, inputs=dict(archive='reloaded.' + suffix))
    else:
        raise ValueError('Unknown dataset workflow mode')
    dataset_node = 'in_dataset' if mode == 'restore' else 'dataset'
    stats_node = 'in_statistics' if mode == 'restore' else 'statistics'
    outputs = [dict(key='dataset', type_name='UNIMATE_DATASET')]
    if mode != 'restore':
        for index, (per_dataset, balanced, tie_std) in enumerate(MODES):
            key = 'stats_' + str(index)
            graph[key] = dict(class_type='UniMateDatasetStatistics', inputs=dict(
                dataset=[dataset_node, 0], per_dataset=per_dataset, balanced=balanced, tie_std=tie_std))
            outputs.append(dict(key=key, type_name='UNIMATE_STATISTICS'))
        if mode == 'build':
            stats_node = 'stats_0'
    outputs.append(dict(key='statistics', type_name='UNIMATE_STATISTICS'))
    for index, (alpha, dataset_alpha, epoch) in enumerate(SAMPLING_MODES):
        key = 'plan_' + str(index)
        if mode == 'restore':
            boundary(key, 'UNIMATE_SAMPLING')
        else:
            graph[key] = dict(class_type='UniMatePlanSampling', inputs=dict(
                dataset=[dataset_node, 0], alpha=alpha, two_level=dataset_alpha is not None,
                dataset_alpha=dataset_alpha if dataset_alpha is not None else .25, epoch=epoch))
        outputs.append(dict(key=key, type_name='UNIMATE_SAMPLING'))
    for output in outputs:
        key = output['key']
        source = dataset_node if key == 'dataset' else stats_node if key == 'statistics' else key
        if mode == 'restore' and key.startswith('plan_'):
            source = 'in_' + key
        graph['out_' + key] = dict(class_type='CloudPartitionOutput', inputs=dict(
            value=[source, 0], boundary_key=key, type_name=output['type_name'], output_path=''))
    for key, node, source in (('dataset', 'UniMateSaveDataset', dataset_node),
                              ('statistics', 'UniMateSaveStatistics', stats_node)):
        graph['save_' + key] = dict(class_type=node, inputs={key: [source, 0],
                                                          'filename_prefix': 'verified/' + mode + '-' + key})
    return graph, inputs, outputs


def run_checks(base, args, workspace):
    import numpy as np
    import folder_paths
    from cloud_offload.assets import resolve_partition_assets
    from cloud_offload.comfyui import ComfyUIWorkflowExecutor
    from cloud_offload.config import CloudConfig
    from cloud_offload.partition_protocol import dump_bundle, load_bundle, pack_execution_values, unpack_execution_values
    from cloud_offload.queue import JobQueue
    from cloud_offload.storage import LocalStorage
    from cloud_offload.worker import Worker
    from tools.verify_workflow import submit
    from unimate_pack.contracts import decode_arrays, encode_arrays, validate_motion, validate_rig
    from unimate_pack.dataset_builder import build_dataset
    from unimate_pack.dataset_io import load_dataset
    from unimate_pack.dataset_selection import split_dataset, sampling_plan, validate_sampling
    from unimate_pack.statistics import dataset_statistics, validate_statistics
    from unimate_pack.statistics_io import load_statistics

    def read_values(path):
        return unpack_execution_values(load_bundle(path))
    rig_values, motion_values = read_values(args.rig), read_values(args.motion)
    assert len(rig_values) == len(motion_values) == 1
    rig, motion = rig_values[0], motion_values[0]
    validate_rig(rig)
    validate_motion(motion, rig)
    motions = [copy.deepcopy(motion) for _ in LABELS]
    for case, shift in zip(motions, (0, .125, .25, .5, 10000), strict=True):
        arrays = decode_arrays(case['features'])
        arrays['features'][:, :, 9:12] += shift
        case['features'] = encode_arrays(**arrays)
        validate_motion(case, rig)
    raw_dataset = build_dataset([rig] * len(LABELS), motions, json.dumps(LABELS), 'objaverse')
    dataset, split_report = split_dataset(raw_dataset, 0, 17, SPLIT_OPTIONS)
    assert split_report['eval_ids'] == ['evaluation']
    expected = [dataset_statistics(dataset, per_dataset=p, balanced=b, tie_std=t) for p, b, t in MODES]
    assert all(value['clip_count'] == 4 and value['datasets'] == ['mixamo', 'objaverse'] for value in expected)
    expected_plans = [sampling_plan(dataset, alpha=a, dataset_alpha=d, epoch=e) for a, d, e in SAMPLING_MODES]
    config = CloudConfig(storage_path=str(workspace / 'worker-store'),
                         queue_db_path=str(workspace / 'worker-queue.db'))
    worker = Worker.__new__(Worker)
    worker.queue, worker.storage = JobQueue(config.queue_db_path), LocalStorage(config.storage_path)
    executor = ComfyUIWorkflowExecutor(base_url=base)
    spec = importlib.util.spec_from_file_location('dataset_cloud_client', args.client_root / 'client.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    previous_output = folder_paths.get_output_directory()
    previous_partition = os.environ.get('COMFY_PARTITION_ROOT')
    folder_paths.set_output_directory(str(workspace / 'client-restored'))
    os.environ['COMFY_PARTITION_ROOT'] = str(workspace / 'partition')
    jobs, staged, direct_outputs = [], [], {}

    def verify_values(values):
        assert values['dataset'] == dataset
        for key, value in values.items():
            if key == 'dataset':
                continue
            if key.startswith('plan_'):
                validate_sampling(value, dataset)
                assert value == expected_plans[int(key.removeprefix('plan_'))]
                continue
            validate_statistics(value)
            target = expected[int(key.removeprefix('stats_'))] if key.startswith('stats_') else expected[0]
            assert {k: v for k, v in value.items() if k != 'arrays'} == {k: v for k, v in target.items() if k != 'arrays'}
            for field, array in decode_arrays(target['arrays']).items():
                np.testing.assert_array_equal(decode_arrays(value['arrays'])[field], array)
            assert value['arrays'] == target['arrays']

    def execute(mode, artifacts):
        graph, inputs, outputs = workflow_graph(mode)
        partition = dict(schema='comfy.partition.job.v1', partition_id='dataset-' + mode,
                         workflow=graph, inputs=inputs, outputs=outputs)
        job = worker.queue.create(model='comfyui-partition-v1', input_path='artifacts://partition',
                                  request=dict(kind='comfyui-partition', partition=partition,
                                               input_artifacts=artifacts, timeout_seconds=300))
        result = worker._run_comfyui_partition(job, executor)
        restored_ui = client.restore_partition_files({**result, 'job_id': job.id})
        assert restored_ui.get('files'), 'Dataset file descriptors lost during restoration'
        assert len(result['files']) == 2
        values = {}
        for key, artifact in result['output_artifacts'].items():
            path = workspace / 'partition' / (mode + '-' + key + '.part')
            worker._download_partition_artifact(artifact, path)
            items = read_values(path)
            assert len(items) == 1
            values[key] = items[0]
        verify_values(values)
        for file in result['files']:
            data = base64.b64decode(file['data'], validate=True)
            target = Path(folder_paths.get_output_directory()) / 'cloud_offload' / job.id / file.get('subfolder', '') / file['filename']
            assert target.read_bytes() == data
            if file['filename'].endswith('.unimatedata'):
                assert load_dataset(data) == dataset
            else:
                assert load_statistics(data) == expected[0]
        jobs.append(dict(job_id=job.id, prompt_id=result['prompt_id'], partition=partition,
                         files_restored=len(result['files']), output_artifacts=result['output_artifacts']))
        return result

    try:
        source_artifacts = {}
        for key, values in (('rig', [rig] * len(LABELS)), ('motion', motions)):
            path = workspace / 'partition' / ('source-' + key + '.part')
            dump_bundle(pack_execution_values(values), path)
            artifact = worker._upload_partition_artifact(path)
            source_artifacts[key] = artifact['artifact_id']
        graph, _, outputs = workflow_graph('build')
        for key in ('rig', 'motion'):
            graph['in_' + key]['inputs']['artifact_path'] = str(workspace / 'partition' / ('source-' + key + '.part'))
        for output in outputs:
            path = workspace / 'partition' / ('direct-' + output['key'] + '.part')
            graph['out_' + output['key']]['inputs']['output_path'] = str(path)
            direct_outputs[output['key']] = path
        direct = submit(base, graph, timeout=300)
        verify_values({key: read_values(path)[0] for key, path in direct_outputs.items()})
        first = execute('build', source_artifacts)
        execute('restore', {key: artifact for key, artifact in first['output_artifacts'].items()
                            if key in ('dataset', 'statistics') or key.startswith('plan_')})
        for file in first['files']:
            suffix = Path(file['filename']).suffix
            data = base64.b64decode(file['data'], validate=True)
            path = workspace / 'input' / ('reloaded' + suffix)
            path.write_bytes(data)
            uploaded = worker._upload_partition_artifact(path)
            declared = dict(category='__input__', filename=path.name, sha256=uploaded['artifact_id'],
                            size=len(data), format='other')
            resolved, missing = resolve_partition_assets(config, [declared], {}, worker.storage)
            assert not missing and len(resolved) == 1
            path.unlink()
            worker._stage_declared_asset(resolved[0], workspace / 'models', None)
            assert path.read_bytes() == data
            staged.append(declared)
        execute('reload', {})
        return dict(direct_status=direct['status'], jobs=jobs, staged_assets=staged,
                    rig_id=rig['rig_id'], source_artifact_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest()
                        for key, path in (('rig', args.rig), ('motion', args.motion))},
                    statistics_modes=[dict(per_dataset=p, balanced=b, tie_std=t) for p, b, t in MODES],
                    split_report=split_report,
                    sampling_modes=[dict(alpha=a, dataset_alpha=d, epoch=e) for a, d, e in SAMPLING_MODES],
                    scope='Direct and real partition handlers, value capture/restore, archive staging and client retrieval; no provider or container deployment')
    finally:
        folder_paths.set_output_directory(previous_output)
        if previous_partition is None:
            os.environ.pop('COMFY_PARTITION_ROOT', None)
        else:
            os.environ['COMFY_PARTITION_ROOT'] = previous_partition


def verify(args):
    from tools.verify_workflow import install_link, request
    workspace = args.workdir.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Select a new empty work directory')
    for name in ('input', 'output', 'temp', 'user', 'models', 'custom_nodes', 'partition'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    install_link(ROOT, workspace / 'custom_nodes/comfy-unimate')
    install_link((args.cloud_root / 'deploy/runtime-profiles/comfyui/ComfyUI-Cloud-Offload-Runtime').resolve(),
                 workspace / 'custom_nodes/cloud-runtime')
    sys.path[:0] = [str(args.cloud_root.resolve()), str(args.comfy_root.resolve())]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = os.environ.copy()
    env.update(COMFY_PARTITION_ROOT=str(workspace / 'partition'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(args.cloud_root.resolve()) + os.pathsep + env.get('PYTHONPATH', ''))
    command = [str(args.python.absolute()), str(args.comfy_root.resolve() / 'main.py'),
               '--listen', '127.0.0.1', '--port', str(port), '--base-directory', str(workspace),
               '--user-directory', str(workspace / 'user'), '--database-url', 'sqlite:///:memory:',
               '--disable-api-nodes', '--disable-auto-launch', '--disable-all-custom-nodes',
               '--whitelist-custom-nodes', 'comfy-unimate', 'cloud-runtime', '--cpu']
    with (workspace / 'server.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=args.comfy_root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError((workspace / 'server.log').read_text(errors='replace')[-6000:])
                try:
                    info = json.loads(request(base, '/object_info'))
                    break
                except (OSError, RuntimeError):
                    time.sleep(0.5)
            else:
                raise TimeoutError('ComfyUI startup timed out')
            assert all(node['class_type'] in info for mode in ('build', 'restore', 'reload')
                       for node in workflow_graph(mode)[0].values())
            report = dict(system_stats=json.loads(request(base, '/system_stats')), python=sys.version,
                          evidence=run_checks(base, args, workspace))
            report['revisions'] = {key: subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True).stdout.strip() for key, path in (
                    ('pack', ROOT), ('comfy', args.comfy_root), ('runner', args.cloud_root), ('client', args.client_root))}
            (workspace / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(result='passed', jobs=len(report['evidence']['jobs']),
                                  files_restored=sum(job['files_restored'] for job in report['evidence']['jobs']),
                                  report=str(workspace / 'report.json'))))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comfy-root', 'cloud-root', 'client-root', 'python', 'rig', 'motion', 'workdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    verify(parser.parse_args())
