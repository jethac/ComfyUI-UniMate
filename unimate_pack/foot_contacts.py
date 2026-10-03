"""Contact-joint selection and detection from UniMate Appendix E.5."""

import re

import numpy as np

from .contracts import MAX_ARRAY_BYTES


_CONTACT_WORD = re.compile(r'(?<![a-z])(foot|toe|ball|paw|hoof|pastern)(?![a-z])', re.IGNORECASE)


def select_contact_joints(parents, names):
    """Select distal canonical contact names on each ordered limb chain."""
    parents = np.asarray(parents)
    if (parents.ndim != 1 or not len(parents) or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or np.any(parents[1:] < 0)
            or np.any(parents[1:] >= np.arange(1, len(parents)))
            or len(names) != len(parents) or any(not isinstance(name, str) for name in names)):
        raise ValueError('Contact selection requires ordered parents and joint names')
    candidates = {index for index, name in enumerate(names) if index and
                  _CONTACT_WORD.search(re.sub(r'([a-z])([A-Z])', r'\1 \2', name))}
    distal = candidates.copy()
    for joint in candidates:
        parent = int(parents[joint])
        while parent >= 0:
            distal.discard(parent)
            parent = int(parents[parent])
    return sorted(distal)


def contact_segments(positions, contacts, *, ground_height, root_height,
                     height_threshold=0.05, displacement_threshold=0.025,
                     check_cancel=lambda: None):
    """Find filtered contact runs and ground-plane mean anchors at 30 fps.

    The first frame has zero measured displacement. Median filtering repeats
    endpoint values; this boundary convention is not specified by the paper.
    Runs shorter than three frames are removed from the returned mask.
    """
    check_cancel()
    positions = np.asarray(positions)
    if (positions.ndim != 3 or not positions.shape[0] or not positions.shape[1] or positions.shape[2] != 3
            or positions.dtype.kind != 'f' or positions.nbytes > MAX_ARRAY_BYTES
            or not np.isfinite(positions).all()):
        raise ValueError('Contact detection requires finite bounded (T,J,3) positions')
    if not np.isfinite(root_height) or root_height <= 0:
        raise ValueError('Rest root height must be finite and positive')
    if (not np.isfinite(ground_height) or not np.isfinite(height_threshold)
            or height_threshold <= 0 or not np.isfinite(displacement_threshold)
            or displacement_threshold <= 0):
        raise ValueError('Ground and contact thresholds must be finite; thresholds must be positive')
    contacts = list(contacts)
    if (any(not isinstance(joint, (int, np.integer)) or isinstance(joint, (bool, np.bool_))
            or not 0 < joint < positions.shape[1] for joint in contacts)
            or len(set(contacts)) != len(contacts)):
        raise ValueError('Contact joints must be distinct non-root joint indices')
    if not contacts:
        return [], np.zeros((len(positions), 0), dtype=bool)
    selected = positions[:, contacts].astype(np.float64, copy=False)
    height = (selected[..., 1] - ground_height) / root_height
    displacement = np.zeros_like(height)
    displacement[1:] = np.linalg.norm(np.diff(selected[..., [0, 2]], axis=0), axis=-1) / root_height
    if not np.isfinite(height).all() or not np.isfinite(displacement).all():
        raise ValueError('Normalized contact measurements must be finite')
    raw = (height < height_threshold) & (displacement < displacement_threshold)
    padded = np.pad(raw, ((2, 2), (0, 0)), mode='edge')
    mask = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=0).sum(axis=-1) >= 3
    segments = []
    for column, joint in enumerate(contacts):
        check_cancel()
        edges = np.flatnonzero(np.diff(np.pad(mask[:, column].astype(np.int8), (1, 1))))
        for start, end in zip(edges[::2], edges[1::2]):
            check_cancel()
            if end - start < 3:
                mask[start:end, column] = False
                continue
            anchor = selected[start:end, column].mean(axis=0)
            anchor[1] = ground_height
            if not np.isfinite(anchor).all():
                raise ValueError('Contact anchor must be finite')
            segments.append({'joint': joint, 'start': int(start), 'end': int(end), 'anchor': anchor})
    check_cancel()
    return segments, mask
