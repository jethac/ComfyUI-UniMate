import io
import json
import warnings
import zipfile

import numpy as np
import pytest

from unimate_pack.contracts import encode_arrays
from unimate_pack.dataset_contracts import make_dataset
from unimate_pack import dataset_io
from test_dataset_contracts import dataset_parts, replace_payload


def test_deterministic_archive_preserves_unicode_payloads_and_shared_feature_bytes():
    manifest, files = dataset_parts(151)
    manifest['clips'].append(dict(manifest['clips'][0], id='clip2', split='eval'))
    value = make_dataset(manifest, files)
    payload = dataset_io.dump_dataset(value)
    assert dataset_io.dump_dataset(value) == payload
    restored = dataset_io.load_dataset(payload)
    assert restored == value
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {'manifest.json'} | {f'arrays/{d}.npz' for d in files}
        assert all(info.create_system == 3 for info in archive.infolist())


def archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as output:
        for name, payload in entries:
            output.writestr(name, payload)
    return stream.getvalue()


def valid_entries():
    manifest, files = dataset_parts()
    envelope = dict(schema='unimate.dataset.file.v1', manifest=manifest, files=sorted(files))
    return [('manifest.json', json.dumps(envelope).encode())] + [
        (f'arrays/{digest}.npz', payload) for digest, payload in files.items()
    ]


@pytest.mark.parametrize('kind', ['duplicate', 'traversal', 'extra', 'missing', 'tampered', 'json_duplicate'])
def test_unsafe_or_inconsistent_archives_fail(kind):
    entries = valid_entries()
    if kind == 'duplicate':
        entries.append(entries[1])
    elif kind == 'traversal':
        entries.append(('../outside.npz', b'bad'))
    elif kind == 'extra':
        entries.append(('extra.npz', b'bad'))
    elif kind == 'missing':
        entries.pop()
    elif kind == 'tampered':
        entries[1] = (entries[1][0], b'bad')
    else:
        entries[0] = ('manifest.json', b'{"schema":"one","schema":"two"}')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Duplicate name:', category=UserWarning)
        with pytest.raises(ValueError):
            dataset_io.load_dataset(archive(entries))


def test_archive_declarations_bound_aggregate_expansion_before_read(monkeypatch):
    payload = archive(valid_entries())
    monkeypatch.setattr(dataset_io, 'MAX_ARCHIVE_BYTES', 100)
    with pytest.raises(ValueError, match='size|limit|large'):
        dataset_io.load_dataset(payload)


@pytest.mark.parametrize('error', [InterruptedError, RuntimeError])
def test_load_propagates_cancellation_without_relabeling_it_as_corruption(error):
    payload = archive(valid_entries())

    def cancel():
        raise error('requested cancellation')

    with pytest.raises(error, match='requested cancellation'):
        dataset_io.load_dataset(payload, cancel=cancel)


def test_aggregate_payload_budget_rejected_before_any_array_member_read(monkeypatch):
    entries = valid_entries()
    sizes = [len(payload) for _, payload in entries[1:]]
    limit = max(sizes) + (sum(sizes) - max(sizes)) // 2
    monkeypatch.setattr(dataset_io, 'MAX_ARRAY_BYTES', limit)
    original_read = zipfile.ZipFile.read

    def read(self, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert filename == 'manifest.json', 'array member read before aggregate budget check'
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', read)
    with pytest.raises(ValueError, match='size|limit|large'):
        dataset_io.load_dataset(archive(entries))


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, '>f8'])
def test_archive_roundtrip_retains_precision_and_byte_order(dtype):
    manifest, files = dataset_parts()
    features = np.full((3, 5, 12), 1 + 2**-40, dtype=dtype)
    replace_payload(manifest, files, 'features', encode_arrays(features=features))
    original = make_dataset(manifest, files)
    assert dataset_io.load_dataset(dataset_io.dump_dataset(original)) == original


def test_dump_cancellation_returns_no_partial_archive_and_preserves_input():
    manifest, files = dataset_parts()
    original = make_dataset(manifest, files)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise InterruptedError('cancelled after manifest write')

    with pytest.raises(InterruptedError, match='manifest write'):
        dataset_io.dump_dataset(original, cancel=cancel)
    assert original == make_dataset(manifest, files)


def test_nested_object_array_is_rejected_without_pickle_loading():
    manifest, files = dataset_parts()
    stream = io.BytesIO()
    np.savez(stream, features=np.full((3, 5, 12), None, dtype=object))
    replace_payload(manifest, files, 'features', stream.getvalue())
    envelope = dict(schema='unimate.dataset.file.v1', manifest=manifest, files=sorted(files))
    entries = [('manifest.json', json.dumps(envelope).encode())] + [
        (f'arrays/{digest}.npz', payload) for digest, payload in files.items()
    ]
    with pytest.raises(ValueError, match='dtype|NPY'):
        dataset_io.load_dataset(archive(entries))
