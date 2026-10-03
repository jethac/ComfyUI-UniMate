"""Pinned MIT UniMate reduction, rotation and geodesic loss helpers."""
import torch
import torch as th


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def rotation_6d_to_matrix_safe(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to matrix with NaN protection and epsilon offset.

    Args:
        cont6d: 6D rotation representation (*, 6).

    Returns:
        Rotation matrices (*, 3, 3).
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    epsilon = 1e-8
    cont6d = torch.nan_to_num(cont6d) + epsilon
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.linalg.norm(x_raw, dim=-1, keepdims=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.linalg.norm(z, dim=-1, keepdims=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def geodesic_distance(R_pred, R_gt):
    """Compute geodesic distance (angle in radians) between rotation matrices.

    Args:
        R_pred: Predicted rotation matrices (..., 3, 3).
        R_gt: Ground truth rotation matrices (..., 3, 3).

    Returns:
        Geodesic angle in radians (..., 1).
    """
    R_rel = th.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = th.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)

    epsilon = 1e-6
    theta = th.arccos(th.clamp((trace - 1) / 2, -1 + epsilon, 1 - epsilon))

    return theta[..., None]
