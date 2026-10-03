"""Forward UniMate features from canonical joint poses without Motion runtime."""

import numpy as np


def rebase_rotations(rest, animated, parents):
    """Cancel rest orientations in each joint's cumulative parent basis."""
    rest = np.asarray(rest, dtype=np.float64)
    animated = np.asarray(animated, dtype=np.float64)
    parents = np.asarray(parents)
    joints = len(parents)
    if rest.shape != (joints, 3, 3) or animated.ndim != 4 or animated.shape[1:] != rest.shape:
        raise ValueError("Rest and animated rotations must describe the same skeleton")
    if joints < 1 or parents.dtype.kind not in "iu" or parents[0] != -1 or any(
        not 0 <= int(parent) < index for index, parent in enumerate(parents[1:], 1)
    ):
        raise ValueError("Parents must define one ordered connected skeleton")
    for value in (rest, animated):
        if not np.isfinite(value).all() or not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), atol=1e-5) or not np.allclose(
            np.linalg.det(value), 1, atol=1e-5
        ):
            raise ValueError("Pose rotations must be proper orthonormal matrices")
    rebased = animated.copy()
    rebased[:, 0] = animated[:, 0] @ rest[0].T
    cumulative = rest.copy()
    for joint, parent in enumerate(parents[1:], 1):
        cumulative[joint] = cumulative[parent] @ rest[joint]
        rebased[:, joint] = cumulative[parent] @ animated[:, joint] @ rest[joint].T @ cumulative[parent].T
    return rebased


def encode_motion_features(positions, rotations, parents, facing):
    """Encode F canonical poses as F-1 RIFKE/6D/velocity feature frames.

    Rotations are rest-relative joint rotation matrices; facing matrices map
    global vectors into the root-facing frame. Parent rotations occupy child
    feature slots, matching the released HML ordering.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    facing = np.asarray(facing, dtype=np.float64)
    parents = np.asarray(parents)
    if positions.ndim != 3 or positions.shape[2] != 3 or len(positions) < 2:
        raise ValueError("Feature encoding requires at least two joint poses (F,J,3)")
    frames, joints = positions.shape[:2]
    if rotations.shape != (frames, joints, 3, 3) or facing.shape != (frames, 3, 3):
        raise ValueError("Rotation/facing dimensions do not match joint poses")
    if parents.shape != (joints,) or parents.dtype.kind not in "iu" or parents[0] != -1 or any(
        not 0 <= int(parent) < index for index, parent in enumerate(parents[1:], 1)
    ):
        raise ValueError("Parents must define one ordered connected skeleton")
    if not all(np.isfinite(value).all() for value in (positions, rotations, facing)):
        raise ValueError("Joint poses must be finite")
    for value in (rotations, facing):
        if not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), atol=1e-5) or not np.allclose(
            np.linalg.det(value), 1, atol=1e-5
        ):
            raise ValueError("Pose rotations must be proper orthonormal matrices")
    centered = positions.copy()
    centered[..., 0] -= positions[:, :1, 0]
    centered[..., 2] -= positions[:, :1, 2]
    rifke = np.einsum("tij,tkj->tki", facing, centered)
    velocity = np.einsum("tij,tkj->tki", facing[1:], np.diff(positions, axis=0))
    reordered = np.empty_like(rotations)
    reordered[:, 0] = facing
    reordered[:, 1:] = rotations[:, parents[1:]]
    six = np.concatenate([reordered[..., 0], reordered[..., 1]], axis=-1)
    return np.concatenate([rifke[:-1], six[:-1], velocity], axis=-1).astype(np.float32)
