"""Deterministic dataset archives with no pickle or filesystem extraction."""

import io
import json
import zipfile

from .contracts import MAX_ARRAY_BYTES, MAX_JSON_BYTES, _digest, _members, _unique_json_pairs
from .dataset_contracts import MAX_FILES, make_dataset, manifest_bytes, validate_dataset

MAX_ARCHIVE_BYTES = MAX_ARRAY_BYTES + MAX_JSON_BYTES + 4 * 1024 * 1024


def dump_dataset(value, *, cancel=None):
    validate_dataset(value, cancel=cancel)
    envelope = dict(schema='unimate.dataset.file.v1', manifest=value['manifest'],
                    files=sorted(value['files']))
    entries = [('manifest.json', manifest_bytes(envelope))]
    entries.extend((f'arrays/{digest}.npz', value['files'][digest]) for digest in envelope['files'])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, payload in entries:
            if cancel is not None:
                cancel()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    payload = stream.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError('Dataset archive exceeds size limit')
    return payload


def load_dataset(payload, *, cancel=None):
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_ARCHIVE_BYTES:
        raise ValueError('Dataset archive exceeds size limit or is empty')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = _members(archive, MAX_ARCHIVE_BYTES, MAX_ARRAY_BYTES, MAX_FILES + 1)
            by_name = {info.filename: info for info in infos}
            manifest_info = by_name.get('manifest.json')
            if manifest_info is None or manifest_info.file_size > MAX_JSON_BYTES:
                raise ValueError('Dataset manifest missing or exceeds size limit')
            envelope = json.loads(archive.read(manifest_info).decode('utf-8'),
                                  object_pairs_hook=_unique_json_pairs)
            if (type(envelope) is not dict or set(envelope) != {'schema', 'manifest', 'files'}
                    or envelope['schema'] != 'unimate.dataset.file.v1'):
                raise ValueError('Invalid dataset archive schema or fields')
            digests = envelope['files']
            if type(digests) is not list or not 1 <= len(digests) <= MAX_FILES:
                raise ValueError('Dataset archive requires bounded payload declarations')
            for digest in digests:
                _digest(digest, 'dataset file')
            if len(set(digests)) != len(digests):
                raise ValueError('Duplicate dataset payload declaration')
            if set(by_name) != {'manifest.json'} | {f'arrays/{d}.npz' for d in digests}:
                raise ValueError('Dataset archive members differ from declarations')
            if sum(by_name[f'arrays/{d}.npz'].file_size for d in digests) > MAX_ARRAY_BYTES:
                raise ValueError('Dataset payload declarations exceed shard size limit')
            files = {}
            for digest in digests:
                if cancel is not None:
                    cancel()
                files[digest] = archive.read(f'arrays/{digest}.npz')
            return make_dataset(envelope['manifest'], files, cancel=cancel)
    except (EOFError, zipfile.BadZipFile, UnicodeError) as error:
        raise ValueError('Invalid dataset archive') from error
