"""Managed, cancellable Blender jobs. No bpy import occurs in ComfyUI."""

from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .contracts import (
    validate_asset,
    validate_rig,
    validate_motion,
    make_rig,
    encode_arrays,
    decode_arrays,
)
from .assets import validate_glb


def blender_executable():
    selected = os.environ.get("UNIMATE_BLENDER", "")
    path = Path(selected) if selected else None
    if path is None or not path.is_file():
        raise RuntimeError(
            "Set UNIMATE_BLENDER to an installed Blender executable on this worker"
        )
    return str(path.resolve())


def _interrupt():
    try:
        import comfy.model_management
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def _stop_posix_group(process):
    import signal

    group = process.pid  # start_new_session=True makes the child the group leader.
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A reaped leader does not imply its descendants exited. Always
        # escalate the group, including grandchildren that ignore SIGTERM.
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_process(arguments, folder, cancel=None, timeout=300):
    cancel = cancel or _interrupt
    log = Path(folder) / "blender.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            arguments,
            cwd=str(folder),
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            start_new_session=os.name != "nt",
        )
        try:
            start = time.monotonic()
            while process.poll() is None:
                cancel()
                if time.monotonic() - start > timeout:
                    raise RuntimeError("Blender job exceeded its 300 second limit")
                time.sleep(0.1)
            cancel()
            if process.returncode:
                tail = log.read_bytes()[-6000:].decode("utf-8", errors="replace")
                raise RuntimeError(f"Blender job failed ({process.returncode}): {tail}")
        finally:
            if os.name != "nt":
                # Descendants may outlive an already-exited group leader,
                # including cancellation observed after the polling loop.
                _stop_posix_group(process)
            elif process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _temporary_root():
    try:
        import folder_paths
    except ImportError:
        return None
    root = Path(folder_paths.get_temp_directory())
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _job(operation, asset, **extra):
    executable = blender_executable()
    with tempfile.TemporaryDirectory(
        prefix="unimate ", dir=_temporary_root()
    ) as temporary:
        folder = Path(temporary)
        (folder / "input.glb").write_bytes(asset["glb"])
        request = {"operation": operation, "name": Path(asset["name"]).stem}
        for key, value in extra.items():
            if isinstance(value, bytes):
                (folder / (key + ".npz")).write_bytes(value)
            else:
                request[key] = value
        (folder / "job.json").write_text(
            json.dumps(request, allow_nan=False), encoding="utf-8"
        )
        worker = Path(__file__).with_name("blender_job.py").resolve()
        _run_process(
            [
                executable,
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(worker),
                "--",
                str(folder / "job.json"),
            ],
            folder,
        )
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if operation == "prepare":
            return (folder / "conditioning.npz").read_bytes(), report["mapping"]
        return (folder / "output.glb").read_bytes()


def prepare_rig(asset, facing, left_joint="", right_joint=""):
    validate_asset(asset)
    validate_glb(asset["glb"])
    conditioning, mapping = _job(
        "prepare", asset, facing=facing, left_joint=left_joint, right_joint=right_joint
    )
    # Keep spectral decomposition in the inference NumPy runtime. Blender's
    # bundled LAPACK can choose different eigenvector signs from the server.
    from ._vendor.rig_topology import compute_laplacian_eigenvectors

    arrays = decode_arrays(conditioning)
    arrays["spectral_feats"] = compute_laplacian_eigenvectors(arrays["parents"])[0]
    return make_rig(asset, encode_arrays(**arrays), mapping)


def export_glb(rig, motion):
    validate_rig(rig)
    validate_motion(motion, rig["rig_id"])
    cond = decode_arrays(rig["conditioning"])
    features = decode_arrays(motion["features"])["features"]
    if features.shape[1] != len(cond["parents"]):
        raise ValueError("Motion joint count does not match prepared skeleton")
    output = _job(
        "export",
        rig["asset"],
        conditioning=rig["conditioning"],
        features=motion["features"],
        mapping=rig["mapping"],
        root_origin=motion["metadata"].get("canonical_root_origin", [0, 0, 0]),
    )
    validate_glb(output)
    return output
