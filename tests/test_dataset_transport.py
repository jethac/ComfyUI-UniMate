import importlib.util
import os
from pathlib import Path

import pytest

from unimate_pack.statistics import dataset_statistics, validate_statistics
from unimate_pack.dataset_contracts import validate_dataset
from test_dataset_adapters import paired_dataset


def codec(path, name):
    if not path.is_file():
        pytest.skip('Requires installed Cloud Offload source checkouts')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_client_runner_codecs_preserve_dataset_statistics_execution_values(tmp_path):
    root = Path(__file__).resolve().parents[2]
    client_root = Path(os.environ.get('CLOUD_CLIENT_ROOT', root / 'ComfyUI-Cloud-Offload'))
    runner_root = Path(os.environ.get('CLOUD_RUNNER_ROOT', root / 'cloud-offload'))
    client = codec(client_root / 'partition_protocol.py', 'dataset_client_codec')
    runner = codec(runner_root / 'cloud_offload/partition_protocol.py', 'dataset_runner_codec')
    dataset, _, _ = paired_dataset()
    statistics = dataset_statistics(dataset)
    for kind, value, validate in (
        ('UNIMATE_DATASET', dataset, validate_dataset),
        ('UNIMATE_STATISTICS', statistics, validate_statistics),
    ):
        client.validate_boundary_type(kind)
        runner.validate_boundary_type(kind)
        first = tmp_path / f'{kind} client with spaces.partition'
        client.dump_bundle(client.pack_execution_values([value]), first)
        restored = runner.unpack_execution_values(runner.load_bundle(first))
        assert restored == [value]
        validate(restored[0])
        second = tmp_path / f'{kind} worker with spaces.partition'
        runner.dump_bundle(runner.pack_execution_values(restored), second)
        final = client.unpack_execution_values(client.load_bundle(second))
        assert final == [value]
