"""Opt-in execution of real five-node ComfyUI and partition workflows."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    not os.environ.get("UNIMATE_INTEGRATION_BUNDLE"),
    reason="Set UNIMATE_INTEGRATION_BUNDLE for real model/API integration",
)
def test_real_comfy_api_and_cloud_partition_workflow(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "verify_workflow", root / "tools/verify_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys

    report = module.verify(
        SimpleNamespace(
            comfy_root=Path(os.environ["COMFYUI_ROOT"]),
            python=Path(sys.executable),
            bundle=Path(os.environ["UNIMATE_INTEGRATION_BUNDLE"]),
            blender=Path(os.environ["UNIMATE_BLENDER"]),
            workdir=tmp_path / "isolated ComfyUI",
            cloud_root=Path(os.environ["CLOUD_OFFLOAD_ROOT"]),
            branching=True,
        )
    )
    assert report["status"] == "passed"
    assert len(report["graphs"]) == 2
    assert len(report["retrieved_outputs"]) == 6
