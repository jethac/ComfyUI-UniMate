import numpy as np
import pytest

from tools.verify_workflow import validate_output
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
