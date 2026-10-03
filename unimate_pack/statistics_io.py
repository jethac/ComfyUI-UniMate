"""Statistics persistence using only Unicode, JSON bytes and numeric payloads."""

import json

import numpy as np

from .contracts import MAX_JSON_BYTES, _unique_json_pairs, decode_arrays, encode_arrays
from .dataset_contracts import manifest_bytes
from .statistics import preflight_arrays, validate_statistics

MAX_STATISTICS_ARCHIVE_BYTES = 16 * 1024 * 1024


def dump_statistics(value):
    validate_statistics(value)
    metadata = {field: item for field, item in value.items() if field != 'arrays'}
    return encode_arrays(schema=np.asarray('unimate.statistics.file.v1'),
                         manifest=np.frombuffer(manifest_bytes(metadata), dtype=np.uint8),
                         arrays=np.frombuffer(value['arrays'], dtype=np.uint8))


def load_statistics(payload):
    preflight_arrays(payload, MAX_STATISTICS_ARCHIVE_BYTES, 3)
    arrays = decode_arrays(payload)
    if set(arrays) != {'schema', 'manifest', 'arrays'}:
        raise ValueError('Invalid statistics archive fields')
    schema = arrays['schema']
    if (schema.shape != () or schema.dtype.kind != 'U'
            or str(schema) != 'unimate.statistics.file.v1'):
        raise ValueError('Unsupported statistics archive schema')
    for field in ('manifest', 'arrays'):
        if arrays[field].dtype != np.dtype('uint8') or arrays[field].ndim != 1:
            raise ValueError('Statistics payloads require byte vectors')
    if arrays['manifest'].nbytes > MAX_JSON_BYTES:
        raise ValueError('Statistics manifest exceeds size limit')
    metadata = json.loads(arrays['manifest'].tobytes().decode('utf-8'),
                          object_pairs_hook=_unique_json_pairs)
    if type(metadata) is not dict or 'arrays' in metadata:
        raise ValueError('Invalid statistics manifest')
    value = dict(metadata, arrays=arrays['arrays'].tobytes())
    validate_statistics(value)
    return value
