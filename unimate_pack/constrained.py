"""Replacement-style Euler integration for UniMate's linear velocity flow."""

from __future__ import annotations


def sample_replacement(
    model, cond, known, keep_mask, noise, *, steps=50,
    check_cancel=lambda: None, progress=None,
):
    """Use the same fixed noise for integration and pinned-token interpolation.

    Callers normalize known motion and supply a CFG-wrapped velocity model.
    This path uses [0,1], matching the supported linear velocity transport.
    Frame and joint masks may broadcast across the other dimensions.
    """
    import torch

    if type(steps) is not int or steps < 1:
        raise ValueError("Euler steps must be a positive integer")
    if known.shape != noise.shape or noise.ndim != 4:
        raise ValueError("Known motion and noise must share (B,J,D,T) shape")
    if keep_mask.dtype != torch.bool:
        raise ValueError("Keep mask must be boolean")
    if known.device != noise.device or keep_mask.device != noise.device:
        raise ValueError("Known motion, noise and mask must share a device")
    if known.dtype != noise.dtype or not noise.is_floating_point():
        raise ValueError("Known motion and noise must share a floating dtype")
    try:
        if torch.broadcast_shapes(keep_mask.shape, noise.shape) != noise.shape:
            raise ValueError("Keep mask must broadcast to the motion shape")
    except RuntimeError as error:
        raise ValueError("Keep mask must broadcast to the motion shape") from error
    if not torch.isfinite(known).all() or not torch.isfinite(noise).all():
        raise ValueError("Known motion and noise must be finite")

    def replace(value, time):
        return torch.where(keep_mask, (1 - time) * noise + time * known, value)

    times = torch.linspace(0, 1, steps + 1, device=noise.device)
    value = replace(noise.clone(), times[0])
    for index in range(steps):
        check_cancel()
        time, following = times[index], times[index + 1]
        batch_time = torch.full(
            (value.size(0),), time.item(), device=value.device, dtype=value.dtype,
        )
        velocity = model(value, batch_time, cond=cond)
        value = replace(value + (following - time) * velocity, following)
        if progress:
            progress(index + 1, steps)
    check_cancel()
    return value
