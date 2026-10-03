import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import parse_glb, prepare_document
from unimate_pack.contracts import make_asset, make_rig, encode_arrays
from unimate_pack.conditioning_output import extract_conditioning


def test_conditioning_output_preserves_arrays_basis_and_rig_identity():
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    arrays, mapping = prepare_document(document, "+X")
    rig = make_rig(make_asset(source, "rig.glb"), encode_arrays(**arrays), mapping)
    result = extract_conditioning(rig)
    assert result["schema"] == "unimate.conditioning.v1"
    assert result["rig_id"] == rig["rig_id"]
    assert result["arrays"] == rig["conditioning"]
    assert result["sha256"] == hashlib.sha256(rig["conditioning"]).hexdigest()
    assert result["mapping"] == mapping
