"""Public dataset/statistics schemas and managed archive files."""

import json
import os
from pathlib import Path
import tempfile
import uuid

import folder_paths
from comfy_api.latest import io

Dataset = io.Custom('UNIMATE_DATASET')
Statistics = io.Custom('UNIMATE_STATISTICS')
Rig = io.Custom('UNIMATE_RIG')
Motion = io.Custom('UNIMATE_MOTION')
CATEGORY = '3D/UniMate'


def _cancel():
    from comfy import model_management
    model_management.throw_exception_if_processing_interrupted()


def _one(value):
    if type(value) is not list or len(value) != 1:
        raise ValueError('Dataset list collection requires one scalar configuration value')
    return value[0]


class UniMateBuildDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='Build UniMate Dataset', category=CATEGORY,
                         inputs=[Rig.Input('rigs'), Motion.Input('motions'),
                                 io.String.Input('labels', default='[]', multiline=True),
                                 io.Combo.Input('default_dataset', options=['objaverse', 'mixamo', 'truebones'])],
                         outputs=[Dataset.Output()], is_input_list=True)

    @classmethod
    def execute(cls, rigs, motions, labels, default_dataset):
        from .unimate_pack.dataset_builder import build_dataset
        return io.NodeOutput(build_dataset(rigs, motions, _one(labels), _one(default_dataset), cancel=_cancel))


class UniMateDatasetStatistics(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='UniMate Dataset Statistics', category=CATEGORY,
                         inputs=[Dataset.Input('dataset'), io.Boolean.Input('per_dataset', default=True),
                                 io.Boolean.Input('balanced', default=False), io.Boolean.Input('tie_std', default=False)],
                         outputs=[Statistics.Output(), io.String.Output(display_name='provenance')])

    @classmethod
    def execute(cls, dataset, per_dataset=True, balanced=False, tie_std=False):
        from .unimate_pack.statistics import dataset_statistics
        value = dataset_statistics(dataset, per_dataset=per_dataset, balanced=balanced,
                                   tie_std=tie_std, cancel=_cancel)
        report = {field: item for field, item in value.items() if field not in ('arrays', 'sha256', 'schema')}
        return io.NodeOutput(value, json.dumps(report, ensure_ascii=False, allow_nan=False))


def _input_path(name, suffix, limit):
    from .nodes import _relative_name, _contained
    name = _relative_name(name)
    if Path(name).suffix.lower() != suffix:
        raise ValueError(f'Select a {suffix} archive')
    root = Path(folder_paths.get_input_directory()).resolve()
    path = _contained(root / name, root)
    if not path.is_file():
        raise FileNotFoundError('UniMate numeric archive is missing')
    if not 0 < path.stat().st_size <= limit:
        raise ValueError('UniMate numeric archive exceeds size limit or is empty')
    return path


class _LoadArchive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = Path(folder_paths.get_input_directory())
        names = []
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() != cls.SUFFIX:
                continue
            name = path.relative_to(root).as_posix()
            try:
                _input_path(name, cls.SUFFIX, cls.LIMIT)
            except (ValueError, OSError):
                continue
            names.append(name)
        return io.Schema(node_id=cls.__name__, display_name=cls.DISPLAY_NAME, category=CATEGORY,
                         inputs=[io.Combo.Input('archive', options=sorted(names))], outputs=[cls.TYPE.Output()])

    @classmethod
    def execute(cls, archive):
        _cancel()
        with _input_path(archive, cls.SUFFIX, cls.LIMIT).open('rb') as stream:
            payload = stream.read(cls.LIMIT + 1)
        if len(payload) > cls.LIMIT:
            raise ValueError('UniMate numeric archive exceeds size limit')
        value = cls.load(payload)
        _cancel()
        return io.NodeOutput(value)

    @classmethod
    def fingerprint_inputs(cls, archive):
        from .nodes import _fingerprint
        return _fingerprint(_input_path(archive, cls.SUFFIX, cls.LIMIT))

    @classmethod
    def cloud_offload_assets(cls, inputs):
        from .nodes import _relative_name
        archive = inputs.get('archive')
        _input_path(archive, cls.SUFFIX, cls.LIMIT)
        return [{'category': '__input__', 'filename': _relative_name(archive)}]


class UniMateLoadDataset(_LoadArchive):
    from .unimate_pack.dataset_io import MAX_ARCHIVE_BYTES as LIMIT
    SUFFIX, TYPE, DISPLAY_NAME = '.unimatedata', Dataset, 'Load UniMate Dataset'

    @staticmethod
    def load(payload):
        from .unimate_pack.dataset_io import load_dataset
        return load_dataset(payload, cancel=_cancel)


class UniMateLoadStatistics(_LoadArchive):
    from .unimate_pack.statistics_io import MAX_STATISTICS_ARCHIVE_BYTES as LIMIT
    SUFFIX, TYPE, DISPLAY_NAME = '.unimatestats', Statistics, 'Load UniMate Statistics'

    @staticmethod
    def load(payload):
        from .unimate_pack.statistics_io import load_statistics
        return load_statistics(payload)


def _save_archive(value, filename_prefix, suffix, serializer):
    from .nodes import _relative_name, _contained
    root = Path(folder_paths.get_output_directory()).resolve()
    prefix = _relative_name(filename_prefix)
    _contained(root / prefix, root)
    _cancel()
    payload = serializer(value)
    _cancel()
    directory, filename, counter, _, _ = folder_paths.get_save_image_path(prefix, str(root))
    parent = _contained(Path(directory), root)
    final = _contained(parent / f'{filename}_{counter:05}_{uuid.uuid4().hex}{suffix}', root)
    descriptor, stage_name = tempfile.mkstemp(prefix='.unimate-numeric-', dir=parent)
    stage = Path(stage_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _cancel()
        os.replace(stage, final)
    finally:
        stage.unlink(missing_ok=True)
    return io.NodeOutput(value, ui={'files': [{'filename': final.name,
                         'subfolder': parent.relative_to(root).as_posix(), 'type': 'output'}]})


class UniMateSaveDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='Save UniMate Dataset', category=CATEGORY,
                         inputs=[Dataset.Input('dataset'),
                                 io.String.Input('filename_prefix', default='unimate/dataset')],
                         outputs=[Dataset.Output()], is_output_node=True)

    @classmethod
    def execute(cls, dataset, filename_prefix='unimate/dataset'):
        from .unimate_pack.dataset_io import dump_dataset
        return _save_archive(dataset, filename_prefix, '.unimatedata',
                             lambda value: dump_dataset(value, cancel=_cancel))


class UniMateSaveStatistics(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='Save UniMate Statistics', category=CATEGORY,
                         inputs=[Statistics.Input('statistics'),
                                 io.String.Input('filename_prefix', default='unimate/statistics')],
                         outputs=[Statistics.Output()], is_output_node=True)

    @classmethod
    def execute(cls, statistics, filename_prefix='unimate/statistics'):
        from .unimate_pack.statistics_io import dump_statistics
        return _save_archive(statistics, filename_prefix, '.unimatestats', dump_statistics)
