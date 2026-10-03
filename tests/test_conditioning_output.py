import hashlib
import importlib.util
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import parse_glb, prepare_document
from unimate_pack.contracts import make_asset, make_rig, encode_arrays
from unimate_pack.conditioning_output import extract_conditioning
from unimate_pack.canonical_asset import canonical_asset


def prepared_rig():
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    arrays, mapping = prepare_document(document, "+X")
    return make_rig(make_asset(source, "rig.glb"), encode_arrays(**arrays), mapping)


def test_conditioning_output_preserves_arrays_basis_and_rig_identity():
    rig = prepared_rig()
    result = extract_conditioning(rig)
    assert result["schema"] == "unimate.conditioning.v1"
    assert result["rig_id"] == rig["rig_id"]
    assert result["arrays"] == rig["conditioning"]
    assert result["sha256"] == hashlib.sha256(rig["conditioning"]).hexdigest()
    assert result["mapping"] == rig["mapping"]


def test_preprocessing_outputs_cross_actual_cloud_bundle_codec(tmp_path):
    root = Path(os.environ.get("CLOUD_OFFLOAD_NODES_ROOT", Path(__file__).resolve().parents[2] / "ComfyUI-Cloud-Offload"))
    path = root / "partition_protocol.py"
    if not path.is_file():
        pytest.skip("Requires Cloud Offload node checkout")
    spec = importlib.util.spec_from_file_location("conditioning_cloud_protocol", path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    rig = prepared_rig()
    for socket, value in (("UNIMATE_CONDITIONING", extract_conditioning(rig)),
                          ("UNIMATE_ASSET", canonical_asset(rig))):
        protocol.validate_boundary_type(socket)
        bundle = tmp_path / f"{socket}.part"
        protocol.dump_bundle(value, bundle)
        assert protocol.load_bundle(bundle) == value
