import os
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb

sys.path.insert(0, str(Path(__file__).parent))
from test_blender_math import evaluated_vertices
from unimate_pack.blender import (
    prepare_rig,
    export_glb,
    blender_executable,
    _run_process,
)
from unimate_pack.contracts import make_asset, make_motion, encode_arrays, decode_arrays


@pytest.mark.skipif(not os.environ.get("UNIMATE_BLENDER"), reason="Requires external Blender")
@pytest.mark.parametrize("body_axis", [False, True])
def test_blender_prepares_four_joint_facing_and_preserves_rest(body_axis):
    source = synthetic_glb(True)
    rig = prepare_rig(make_asset(source, "four-facing.glb"), "joint_pair", "Joint_2", "Joint_1",
        left_shoulder="Joint_4", right_shoulder="Joint_3", body_axis=body_axis)
    cond = decode_arrays(rig["conditioning"])
    assert cond["face_joint_idxs"].shape == (4,)
    assert rig["mapping"]["body_axis"] == body_axis
    features = np.zeros((7, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    output = export_glb(rig, make_motion(rig["rig_id"], encode_arrays(features=features), {}))
    np.testing.assert_allclose(evaluated_vertices(output, 0), evaluated_vertices(source), atol=2e-6)


@pytest.mark.skipif(
    not os.environ.get("UNIMATE_BLENDER"),
    reason="Set UNIMATE_BLENDER for external Blender integration",
)
@pytest.mark.parametrize("branching", [False, True])
def test_blender_prepare_identity_and_animation(branching, tmp_path):
    tmp_path = tmp_path / "character motion 日本語"
    tmp_path.mkdir()
    asset = make_asset(synthetic_glb(branching), "synthetic.glb")
    rig = prepare_rig(asset, "+Z")
    cond = decode_arrays(rig["conditioning"])
    j = len(cond["parents"])
    features = np.zeros((60, j, 12), dtype=np.float32)
    features[:, :, 3] = 1
    features[:, :, 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    motion = make_motion(rig["rig_id"], encode_arrays(features=features), {})
    output = export_glb(rig, motion)
    assert len(output) > len(asset["glb"])
    (tmp_path / "identity.glb").write_bytes(output)
    features[:, 0, 9] = 0.01
    features[:, 1, 3:9] = [np.cos(0.2), np.sin(0.2), 0, -np.sin(0.2), np.cos(0.2), 0]
    moving = export_glb(
        rig, make_motion(rig["rig_id"], encode_arrays(features=features), {})
    )
    (tmp_path / "animated.glb").write_bytes(moving)
    assert moving != output
    np.testing.assert_allclose(
        evaluated_vertices(output, 0), evaluated_vertices(asset["glb"]), atol=2e-6
    )
    probe = Path(__file__).parent / "fixtures" / "blender_inspect.py"
    _run_process(
        [
            blender_executable(),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(probe.resolve()),
            "--",
            str(tmp_path / "animated.glb"),
            str(tmp_path / "playback.npz"),
        ],
        tmp_path,
    )
    with np.load(tmp_path / "playback.npz", allow_pickle=False) as result:
        for frame in range(60):
            # Blender's importer deduplicates/reorders triangles. Compare vertex sets.
            expected = evaluated_vertices(moving, frame)
            actual = result["vertices"][frame]
            error = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
            assert np.max(np.min(error, axis=1)) < 2e-5
            assert np.max(np.min(error, axis=0)) < 2e-5


def test_blender_missing_is_actionable(monkeypatch):
    monkeypatch.setenv("UNIMATE_BLENDER", "/nonexistent/unimate/blender")
    with pytest.raises(RuntimeError, match="UNIMATE_BLENDER"):
        blender_executable()


def test_process_cancellation_reaps_child(tmp_path, monkeypatch):
    import subprocess

    original = subprocess.Popen
    children = []

    def record(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", record)

    class Cancelled(Exception):
        pass

    def cancel():
        raise Cancelled()

    with pytest.raises(Cancelled):
        _run_process(
            [sys.executable, "-c", "import time;time.sleep(300)"],
            tmp_path,
            cancel=cancel,
        )
    assert children[0].poll() is not None


@pytest.mark.skipif(not os.environ.get("UNIMATE_BLENDER"), reason="Requires Blender")
@pytest.mark.parametrize("channel_path", ["translation", "rotation"])
def test_source_ancestor_animation_is_ignored(tmp_path, channel_path):
    from unimate_pack.rig_math import parse_glb, pack_glb

    source = synthetic_glb(True)
    doc, binary = parse_glb(source)
    binary = bytearray(binary)

    def add(values, kind):
        a = np.asarray(values, dtype="<f4")
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(a.tobytes())
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": offset, "byteLength": a.nbytes}
        )
        ac = {
            "bufferView": len(doc["bufferViews"]) - 1,
            "componentType": 5126,
            "count": len(a),
            "type": kind,
        }
        if kind == "SCALAR":
            ac.update(min=[float(a.min())], max=[float(a.max())])
        doc["accessors"].append(ac)
        return len(doc["accessors"]) - 1

    times = add([0, 1], "SCALAR")
    values = (
        [[99, 22, 17], [45, 88, 12]]
        if channel_path == "translation"
        else [[0, 0, 0.4, np.sqrt(0.84)], [0, 0.3, 0, np.sqrt(0.91)]]
    )
    translations = add(values, "VEC3" if channel_path == "translation" else "VEC4")
    doc["animations"] = [
        {
            "samplers": [
                {"input": times, "output": translations, "interpolation": "LINEAR"}
            ],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": channel_path}}],
        }
    ]
    doc["buffers"][0]["byteLength"] = len(binary)
    animated = pack_glb(doc, bytes(binary))
    rest_rig = prepare_rig(make_asset(source, "character.glb"), "+Z")
    animated_rig = prepare_rig(make_asset(animated, "character.glb"), "+Z")
    rest = decode_arrays(rest_rig["conditioning"])
    actual = decode_arrays(animated_rig["conditioning"])
    for key in rest:
        np.testing.assert_equal(actual[key], rest[key], err_msg=key)
    assert rest_rig["mapping"] == animated_rig["mapping"]
    assert animated_rig["asset"]["glb"] == animated
    features = np.zeros((60, len(rest["parents"]), 12), dtype=np.float32)
    features[:, :, 3] = 1
    features[:, :, 7] = 1
    features[:, 0, 1] = rest["tpos_first_frame"][0, 1]
    output = export_glb(
        animated_rig,
        make_motion(animated_rig["rig_id"], encode_arrays(features=features), {}),
    )
    np.testing.assert_allclose(
        evaluated_vertices(output, 0), evaluated_vertices(source), atol=2e-6
    )


def test_posix_escalation_targets_group_after_leader_exit(monkeypatch):
    import signal
    from unimate_pack.blender import _stop_posix_group

    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    calls = []

    class Process:
        pid = 1234

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return 0

    monkeypatch.setattr(
        os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False
    )
    _stop_posix_group(Process())
    assert (1234, signal.SIGTERM) in calls
    assert (1234, signal.SIGKILL) in calls


@pytest.mark.skipif(
    os.name == "nt", reason="Actual POSIX process-group lifecycle requires a POSIX host"
)
def test_posix_cancellation_kills_term_resistant_grandchild(tmp_path):
    import time

    pid_file = tmp_path / "grandchild.pid"
    code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(300)"
    parent = "import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(300)"

    class Cancelled(Exception):
        pass

    def cancel():
        if pid_file.exists():
            raise Cancelled()

    with pytest.raises(Cancelled):
        _run_process(
            [sys.executable, "-c", parent, str(pid_file), code], tmp_path, cancel=cancel
        )
    pid = int(pid_file.read_text())
    for attempt in range(50):
        status = Path(f"/proc/{pid}/status")
        if not status.exists() or "State:\tZ" in status.read_text():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Cancelled grandchild remains running")


@pytest.mark.parametrize("cancel_after_exit", [False, True])
def test_run_process_cleans_posix_group_after_leader_exit(
    tmp_path, monkeypatch, cancel_after_exit
):
    import types
    import unimate_pack.blender as adapter

    class Process:
        pid = 9876
        returncode = 0

        def poll(self):
            return 0

    process = Process()
    stopped = []
    monkeypatch.setattr(adapter, "os", types.SimpleNamespace(name="posix"))
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(adapter, "_stop_posix_group", stopped.append)

    class Cancelled(Exception):
        pass

    def cancel():
        if cancel_after_exit:
            raise Cancelled()

    if cancel_after_exit:
        with pytest.raises(Cancelled):
            adapter._run_process(["unused"], tmp_path, cancel=cancel)
    else:
        adapter._run_process(["unused"], tmp_path, cancel=cancel)
    assert stopped == [process]
