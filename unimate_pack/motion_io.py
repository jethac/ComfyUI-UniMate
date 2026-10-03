"""Offline numeric motion archives with explicit canonical rig identity."""

import json

import numpy as np

from .contracts import decode_arrays, encode_arrays, make_motion, validate_motion


def dump_motion(motion: dict) -> bytes:
    validate_motion(motion)
    return encode_arrays(
        features=decode_arrays(motion["features"])["features"],
        schema=np.asarray("unimate.motion.file.v1"),
        rig_id=np.asarray(motion["rig_id"]),
        fps=np.asarray(motion["fps"], dtype=np.int32),
        metadata=np.asarray(json.dumps(motion["metadata"], ensure_ascii=False, allow_nan=False)),
    )


def load_motion(payload: bytes, rig_id: str | None = None) -> dict:
    arrays = decode_arrays(payload)
    if set(arrays) != {"features", "schema", "rig_id", "fps", "metadata"}:
        raise ValueError("Motion archive requires features and identity metadata")
    for name in ("schema", "rig_id", "metadata"):
        if arrays[name].shape != () or arrays[name].dtype.kind != "U":
            raise ValueError(f"Motion archive {name} must be a scalar Unicode string")
    if str(arrays["schema"]) != "unimate.motion.file.v1":
        raise ValueError("Unsupported motion archive schema")
    if arrays["fps"].shape != () or arrays["fps"].dtype != np.dtype("int32") or int(arrays["fps"]) != 30:
        raise ValueError("Motion archive requires fps 30")
    motion = make_motion(
        str(arrays["rig_id"]), encode_arrays(features=arrays["features"]),
        json.loads(str(arrays["metadata"])),
    )
    validate_motion(motion, rig_id)
    return motion
