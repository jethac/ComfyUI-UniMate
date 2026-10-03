import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from rig_generator import synthetic_glb
from unimate_pack.rig_math import parse_glb, prepare_document, animate_document, world_matrices
from unimate_pack.clip_sampling import sample_clip, sample_channel


def test_clip_sampling_preserves_all_exported_frames_and_world_transforms():
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    cond, mapping = prepare_document(document, "+Z")
    features = np.zeros((110, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    features[:, 0, 9] = 0.01
    animated = animate_document(source, cond, mapping, features)
    times, worlds, locals_ = sample_clip(animated, 0)
    np.testing.assert_allclose(times, np.arange(110) / 30)
    rest, _, _ = world_matrices(document)
    np.testing.assert_allclose(worlds[0], rest, atol=2e-6)
    shift = np.linalg.inv(np.asarray(mapping["source_to_canonical"]))[:3, :3] @ [1.09, 0, 0]
    for joint in mapping["joint_indices"]:
        np.testing.assert_allclose(worlds[-1, joint, :3, 3] - worlds[0, joint, :3, 3], shift, atol=2e-6)
    assert locals_.shape == worlds.shape


def test_linear_rotation_uses_shortest_arc_and_clamps_outside_keys():
    values = np.array([[0, 0, 0, 1], [0, 0, -1, 0]], float)
    actual = sample_channel(np.array([0, 1]), values, np.array([-1, 0.5, 2]), "LINEAR", "rotation")
    np.testing.assert_allclose(np.abs(actual[1]), [0, 0, 2**-0.5, 2**-0.5], atol=1e-7)
    # q and -q encode the same rotation; endpoint evaluation may change sign.
    np.testing.assert_allclose(np.abs(np.sum(actual[[0, 2]] * values, axis=1)), 1, atol=1e-7)


def test_cubic_translation_scales_tangents_by_key_interval():
    values = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0],
                       [0, 0, 0], [2, 0, 0], [0, 0, 0]], float)
    actual = sample_channel(np.array([0, 2]), values, np.array([1]), "CUBICSPLINE", "translation")
    np.testing.assert_allclose(actual, [[1.25, 0, 0]])


def test_missing_clip_is_rejected():
    with pytest.raises(ValueError, match="clip"):
        sample_clip(synthetic_glb(False), 0)


def test_step_interpolation_holds_until_the_next_key():
    actual = sample_channel(np.array([0, 1]), np.array([[0, 0, 0], [2, 0, 0]]),
                            np.array([-1, 0.5, 1, 2]), "STEP", "translation")
    np.testing.assert_array_equal(actual, [[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0]])


def test_clip_sampling_stops_on_cancellation():
    source = synthetic_glb(True)
    document, _ = parse_glb(source)
    cond, mapping = prepare_document(document, "+Z")
    features = np.zeros((60, 7, 12), np.float32)
    features[..., 3] = features[..., 7] = 1
    features[:, 0, 1] = cond["tpos_first_frame"][0, 1]
    animated = animate_document(source, cond, mapping, features)
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        sample_clip(animated, check_cancel=cancel)
    assert len(calls) == 3
