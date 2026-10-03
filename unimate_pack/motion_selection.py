"""Frame and joint selections in the prepared skeleton's canonical ordering."""

import numpy as np


def frame_mask(spec: str, valid_frames: int, window: int) -> np.ndarray:
    if type(valid_frames) is not int or not 1 <= valid_frames <= window:
        raise ValueError("Reference length must fit the model window")
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Select at least one frame")
    mask = np.zeros((1, 1, 1, window), dtype=bool)
    for token in spec.split(","):
        try:
            index = int(token.strip())
        except ValueError as error:
            raise ValueError("Frame selection requires comma-separated integers") from error
        if index < 0:
            index += valid_frames
        if not 0 <= index < valid_frames:
            raise ValueError("Selected frame is outside the reference clip")
        mask[..., index] = True
    return mask


def joint_mask(spec: str, names: list[str], padded_joints: int) -> np.ndarray:
    if len(names) > padded_joints:
        raise ValueError("Skeleton exceeds the model joint dimension")
    if not isinstance(spec, str):
        raise ValueError("Joint selection must be text")
    selected = {name.strip().lower() for name in spec.split(",") if name.strip()}
    mask = np.zeros((1, padded_joints, 1, 1), dtype=bool)
    for index, name in enumerate(names):
        aliases = {alias.strip().lower() for alias in name.split("|")}
        mask[:, index] = bool(aliases & selected)
    if not mask.any():
        raise ValueError("No selected joints matched this skeleton")
    return mask
