import numpy as np
import pytest

from tools.verify_workflow import validate_output, validate_batch_cases
from unimate_pack.contracts import make_motion, encode_arrays
from unimate_pack.motion_io import dump_motion


def test_numeric_output_validation_checks_archive_identity_and_arrays():
    motion = make_motion("a" * 64, encode_arrays(features=np.zeros((7, 5, 12), np.float32)), {})
    validate_output("motion.npz", dump_motion(motion))
    with pytest.raises(ValueError):
        validate_output("bad.npz", encode_arrays(features=np.zeros((7, 5, 12), np.float32)))


def test_empty_output_is_rejected():
    with pytest.raises(ValueError):
        validate_output("motion.npz", b"")


def test_batch_artifact_provenance_detects_duplicate_or_misassigned_cases():
    prompts = ["stand", "walk"]
    cases = [{"seed": i, "prompt": prompts[i // 2], "frames": 60} for i in range(4)]
    validate_batch_cases(cases, prompts, 2)
    cases[3] = cases[2]
    with pytest.raises(ValueError):
        validate_batch_cases(cases, prompts, 2)
