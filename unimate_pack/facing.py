"""Dependency-light equivalent of upstream skeleton.get_root_facing_quat."""

import numpy as np


def facing_rotations(positions, indices, body_axis=False):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Facing requires finite (F,J,3) positions")
    if type(body_axis) is not bool:
        raise ValueError("Body-axis selection must be boolean")
    result = np.broadcast_to(np.eye(3), (len(positions), 3, 3)).copy()
    if indices is None:
        return result
    indices = np.asarray(indices)
    if indices.shape == (2,) and np.all(indices == -1):
        return result
    if (indices.shape not in ((2,), (4,)) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= positions.shape[1])):
        raise ValueError("Facing requires two or four valid joint indices")
    across = positions[:, indices[0]] - positions[:, indices[1]]
    if len(indices) == 4:
        across += positions[:, indices[2]] - positions[:, indices[3]]
    forward = np.cross([0, 1, 0], across)
    if np.any(np.linalg.norm(forward, axis=-1) < 1e-8):
        raise ValueError("Source clip has a degenerate facing joint pair")
    if body_axis:
        forward = forward @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).T
    angle = -np.arctan2(forward[:, 0], forward[:, 2])
    c, s = np.cos(angle), np.sin(angle)
    result[:, 0, 0] = result[:, 2, 2] = c
    result[:, 0, 2], result[:, 2, 0] = s, -s
    return result
