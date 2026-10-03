import copy
import hashlib
import io
import json
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from unimate_pack.contracts import (encode_arrays, decode_arrays, make_asset, make_rig,
                                   make_motion, validate_motion, validate_rig)
from unimate_pack.rig_math import prepare_document, parse_glb
from test_blender_math import synthetic_glb


def archive_for_platform(arrays, platform):
    original = zipfile.ZipInfo
    class PlatformZipInfo(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.create_system = platform
    output = io.BytesIO()
    with patch.object(zipfile, 'ZipInfo', PlatformZipInfo):
        np.savez(output, **arrays)
    return output.getvalue()


def legacy_identity(asset, archive, mapping):
    digest = hashlib.sha256(b'unimate.rig.v1\0')
    metadata = json.dumps(mapping, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(',', ':')).encode()
    for part in (asset['sha256'].encode(), archive, metadata):
        digest.update(len(part).to_bytes(8, 'little'))
        digest.update(part)
    return digest.hexdigest()


def test_numeric_encoding_uses_one_platform_marker():
    arrays = {'values': np.arange(15, dtype=np.float32).reshape(5, 3)}
    original = zipfile.ZipInfo
    class WindowsZipInfo(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.create_system = 0
    with patch.object(zipfile, 'ZipInfo', WindowsZipInfo):
        encoded = encode_arrays(**arrays)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert all(info.create_system == 3 for info in archive.infolist())
    np.testing.assert_array_equal(decode_arrays(encoded)['values'], arrays['values'])


def test_platform_independent_preparation_accepts_exact_legacy_motion_ids():
    source = synthetic_glb(True)
    asset = make_asset(source, 'rig.glb')
    arrays, mapping = prepare_document(parse_glb(source)[0], '+Z')
    win, linux = (archive_for_platform(arrays, platform) for platform in (0, 3))
    assert win != linux
    wrig, lrig = (make_rig(asset, archive, mapping) for archive in (win, linux))
    assert wrig['rig_id'] == lrig['rig_id']
    features = np.zeros((7, 7, 12), np.float32)
    for archive in (win, linux):
        old = copy.deepcopy(wrig)
        old['conditioning'] = archive
        old['rig_id'] = legacy_identity(asset, archive, mapping)
        validate_rig(old)
        motion = make_motion(old['rig_id'], encode_arrays(features=features), {})
        validate_motion(motion, wrig)
    motion['rig_id'] = '0' * 64
    with pytest.raises(ValueError, match='identities'):
        validate_motion(motion, wrig)
