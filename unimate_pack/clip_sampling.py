"""glTF animation evaluation for numeric reference-motion extraction.

Interpolation follows the glTF 2.0 animation specification:
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#animations
"""

import copy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .assets import validate_glb, _accessors
from .rig_math import parse_glb, world_matrices


def sample_channel(keys, values, times, interpolation, path):
    keys = np.asarray(keys, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    clamped = np.clip(times, keys[0], keys[-1])
    if interpolation == "STEP" or len(keys) == 1:
        index = np.searchsorted(keys, clamped, side="right") - 1
        result = (values[1::3] if interpolation == "CUBICSPLINE" else values)[index].copy()
    elif interpolation == "LINEAR":
        if path == "rotation":
            result = Slerp(keys, Rotation.from_quat(values))(clamped).as_quat()
        else:
            result = np.stack([np.interp(clamped, keys, values[:, axis]) for axis in range(values.shape[1])], axis=1)
    elif interpolation == "CUBICSPLINE":
        index = np.clip(np.searchsorted(keys, clamped, side="right") - 1, 0, len(keys) - 2)
        duration = keys[index + 1] - keys[index]
        weight = ((clamped - keys[index]) / duration)[:, None]
        a, b = weight**2, weight**3
        result = ((2*b - 3*a + 1) * values[3*index + 1]
                  + (b - 2*a + weight) * duration[:, None] * values[3*index + 2]
                  + (-2*b + 3*a) * values[3*(index + 1) + 1]
                  + (b - a) * duration[:, None] * values[3*(index + 1)])
    else:
        raise ValueError("Unsupported clip interpolation")
    if path == "rotation":
        lengths = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(lengths < 1e-8):
            raise ValueError("Clip contains a degenerate interpolated quaternion")
        result /= lengths
    return result


def sample_clip(payload, clip_index=0, *, check_cancel=lambda: None):
    check_cancel()
    document = validate_glb(payload)
    if type(clip_index) is not int or not 0 <= clip_index < len(document.get("animations", [])):
        raise ValueError("Select an existing animation clip")
    _, binary = parse_glb(payload)
    _, arrays, _ = _accessors(document, binary)
    animation = document["animations"][clip_index]
    channels = []
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        channels.append((channel["target"], arrays[sampler["input"]].reshape(-1),
                         arrays[sampler["output"]], sampler.get("interpolation", "LINEAR")))
    start = min(float(keys[0]) for _, keys, _, _ in channels)
    end = max(float(keys[-1]) for _, keys, _, _ in channels)
    frames = int(np.floor((end - start) * 30 + 1e-5)) + 1
    nodes = len(document["nodes"])
    if frames < 2 or frames * nodes * 16 * 8 * 2 > 256 * 1024 * 1024:
        raise ValueError("Clip is too short or exceeds the pose memory budget")
    times = start + np.arange(frames) / 30
    evaluated = [(target, sample_channel(keys, values, times, interpolation, target["path"]))
                 for target, keys, values, interpolation in channels]
    worlds = np.empty((frames, nodes, 4, 4), dtype=np.float64)
    locals_ = np.empty_like(worlds)
    for frame in range(frames):
        check_cancel()
        pose = copy.deepcopy(document)
        for target, values in evaluated:
            node = pose["nodes"][target["node"]]
            if "matrix" in node:
                raise ValueError("Animated matrix nodes require TRS conversion before extraction")
            node[target["path"]] = values[frame].tolist()
        worlds[frame], locals_[frame], _ = world_matrices(pose)
    check_cancel()
    return times, worlds, locals_
