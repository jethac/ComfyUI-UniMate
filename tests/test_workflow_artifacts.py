import numpy as np
import pytest

from tools.verify_workflow import validate_output, validate_batch_cases
from unimate_pack.contracts import make_motion, encode_arrays
from unimate_pack.motion_io import dump_motion
from PIL import Image
import io


def test_numeric_output_validation_checks_archive_identity_and_arrays():
    motion = make_motion("a" * 64, encode_arrays(features=np.zeros((7, 5, 12), np.float32)), {})
    validate_output("motion.npz", dump_motion(motion))
    with pytest.raises(ValueError):
        validate_output("bad.npz", encode_arrays(features=np.zeros((7, 5, 12), np.float32)))


def test_empty_output_is_rejected():
    with pytest.raises(ValueError):
        validate_output("motion.npz", b"")


def test_preview_artifact_requires_valid_png_pixels():
    output = io.BytesIO()
    Image.new('RGB', (128, 128), (10, 20, 30)).save(output, format='PNG')
    validate_output('frame.png', output.getvalue())
    with pytest.raises(ValueError):
        validate_output('bad.png', b'not a png')


def test_fbx_artifact_requires_binary_fbx_header():
    validate_output('motion.fbx', b'Kaydara FBX Binary  \x00\x1a\x00' + bytes(32))
    with pytest.raises(ValueError):
        validate_output('bad.fbx', b'not binary fbx')


def test_batch_artifact_provenance_detects_duplicate_or_misassigned_cases():
    prompts = ["stand", "walk"]
    cases = [{"seed": i, "prompt": prompts[i // 2], "frames": 60} for i in range(4)]
    validate_batch_cases(cases, prompts, 2)
    cases[3] = cases[2]
    with pytest.raises(ValueError):
        validate_batch_cases(cases, prompts, 2)


def test_multi_rig_batch_provenance_detects_crossed_rig_pairings():
    from tools.verify_workflow import validate_multi_rig_batch_cases
    prompts = ['stand', 'walk']
    sources = ['a' * 64, 'b' * 64]
    cases = [{'seed': i, 'prompt': prompts[(i % 4) // 2], 'frames': 60,
              'source_sha256': sources[i // 4], 'rig_id': ('c' if i < 4 else 'd') * 64}
             for i in range(8)]
    validate_multi_rig_batch_cases(cases, prompts, 2, sources)
    cases[4]['source_sha256'] = sources[0]
    with pytest.raises(ValueError):
        validate_multi_rig_batch_cases(cases, prompts, 2, sources)
