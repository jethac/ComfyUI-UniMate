import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.blender import prepare_rig, export_fbx
from unimate_pack.contracts import make_asset, make_motion, encode_arrays, decode_arrays


@pytest.mark.skipif(not os.environ.get("UNIMATE_BLENDER"), reason="Requires external Blender")
@pytest.mark.parametrize("frames", [7, 110])
def test_fbx_animation_round_trip_preserves_branching_skin_and_coordinates(tmp_path, frames):
    rig = prepare_rig(make_asset(synthetic_glb(True), "branch.glb"), "+X")
    cond = decode_arrays(rig["conditioning"])
    features = np.zeros((frames, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    features[:, 0, 9] = 0.01
    motion = make_motion(rig["rig_id"], encode_arrays(features=features), {})
    output = export_fbx(rig, motion)
    assert output.startswith(b"Kaydara FBX Binary")
    (tmp_path / "motion.fbx").write_bytes(output)
