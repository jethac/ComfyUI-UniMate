"""Convert an asset's selected clip into its prepared UniMate skeleton basis."""

import numpy as np

from .clip_sampling import sample_clip
from .contracts import validate_rig, decode_arrays, encode_arrays, make_motion
from .feature_encoding import encode_motion_features, rebase_rotations
from .rig_math import parse_glb, world_matrices, rotation_part


def extract_motion(rig, clip_index=0):
    validate_rig(rig)
    conditioning = decode_arrays(rig["conditioning"])
    parents = conditioning["parents"]
    joints = rig["mapping"]["joint_indices"]
    transform = np.asarray(rig["mapping"]["source_to_canonical"])
    basis = rotation_part(transform)
    times, worlds, _ = sample_clip(rig["asset"]["glb"], clip_index)
    positions = worlds[:, joints][..., :3, 3] @ transform[:3, :3].T + transform[:3, 3]
    lengths = np.linalg.norm(positions[:, 1:] - positions[:, parents[1:]], axis=-1)
    rest_positions = conditioning["tpos_first_frame"]
    expected = np.linalg.norm(rest_positions[1:] - rest_positions[parents[1:]], axis=-1)
    if not np.allclose(lengths, expected, atol=2e-5, rtol=2e-5):
        raise ValueError("Animated bone lengths do not match the prepared rest skeleton")
    doc, _ = parse_glb(rig["asset"]["glb"])
    rest_worlds, _, _ = world_matrices(doc)
    rest_global = basis @ rotation_part(rest_worlds[joints])
    animated_global = basis @ rotation_part(worlds[:, joints])
    rest_local = rest_global.copy()
    animated_local = animated_global.copy()
    for joint, parent in enumerate(parents[1:], 1):
        rest_local[joint] = rest_global[parent].T @ rest_global[joint]
        animated_local[:, joint] = animated_global[:, parent].swapaxes(-1, -2) @ animated_global[:, joint]
    rotations = rebase_rotations(rest_local, animated_local, parents)
    facing = np.broadcast_to(np.eye(3), (len(times), 3, 3)).copy()
    indices = conditioning["face_joint_idxs"]
    if np.all(indices >= 0):
        across = positions[:, indices[0]] - positions[:, indices[1]]
        forward = np.cross([0, 1, 0], across)
        if np.any(np.linalg.norm(forward, axis=-1) < 1e-8):
            raise ValueError("Source clip has a degenerate facing joint pair")
        angle = -np.arctan2(forward[:, 0], forward[:, 2])
        c, s = np.cos(angle), np.sin(angle)
        facing[:, 0, 0] = facing[:, 2, 2] = c
        facing[:, 0, 2], facing[:, 2, 0] = s, -s
    features = encode_motion_features(positions, rotations, parents, facing)
    return make_motion(rig["rig_id"], encode_arrays(features=features), dict(
        mode="source_clip", clip_index=clip_index, frames=len(features), fps=30,
        sample_start=float(times[0]), sample_end=float(times[-1]),
        source_sha256=rig["asset"]["sha256"],
        canonical_root_origin=[float(positions[0, 0, 0]), 0.0, float(positions[0, 0, 2])],
        upstream_revision=rig["mapping"]["upstream_revision"],
    ))
