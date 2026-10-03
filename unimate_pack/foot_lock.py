"""Independent UniMate E.5 foot-contact correction on export-driving features."""

import numpy as np
import hashlib

from .foot_contacts import contact_segments, select_contact_joints
from .foot_ik import forward_pose, limb_chain, solve_contact, boundary_weights, blend_rotation
from .rig_math import decode_features, rotation_6d_matrices
from .skeleton import recover_positions
from .contracts import validate_rig, validate_motion, decode_arrays, encode_arrays, make_motion


# Conservative numeric allocation estimate, not a measured process-memory peak.
MAX_WORKSPACE_BYTES = 512 * 1024 * 1024


def lock_motion(rig, motion, joint_names='', *, check_cancel=lambda: None):
    """Validate portable inputs and select canonical contacts or explicit names."""
    check_cancel()
    validate_rig(rig)
    validate_motion(motion, rig)
    if not isinstance(joint_names, str) or len(joint_names) > 8192:
        raise ValueError('Contact joint names must be a bounded comma-separated string')
    conditioning = decode_arrays(rig['conditioning'])
    parents, offsets = conditioning['parents'], conditioning['tpos_offsets']
    names = conditioning['joint_names'].tolist()
    cleaned = conditioning['clean_joint_names'].tolist()
    if joint_names.strip():
        requested = [name.strip().casefold() for name in joint_names.split(',')]
        contacts = []
        for name in requested:
            matches = [index for index, pair in enumerate(zip(names, cleaned))
                       if name and name in [value.casefold() for value in pair]]
            if len(matches) != 1 or matches[0] == 0 or matches[0] in contacts:
                raise ValueError('Contact joint override must identify distinct non-root joints')
            contacts.append(matches[0])
    else:
        contacts = select_contact_joints(parents, [a + ' ' + b for a, b in zip(names, cleaned)])
    features = decode_arrays(motion['features'])['features']
    if features.shape[1] != len(parents):
        raise ValueError('Motion joint count does not match prepared skeleton')
    corrected, report = lock_features(features, parents, offsets, contacts,
        root_origin=motion['metadata'].get('canonical_root_origin', [0, 0, 0]),
        check_cancel=check_cancel)
    report['contact_joints'] = contacts
    report['source_features_sha256'] = hashlib.sha256(motion['features']).hexdigest()
    metadata = {**motion['metadata'], 'foot_lock': report}
    payload = motion['features'] if np.array_equal(corrected, features) else encode_arrays(features=corrected)
    result = make_motion(motion['rig_id'], payload, metadata)
    check_cancel()
    return result, report


def lock_features(features, parents, offsets, contacts, *, root_origin=(0, 0, 0),
                  check_cancel=lambda: None):
    """Correct contact limbs; retain length, root channels and untouched features."""
    check_cancel()
    features = np.asarray(features)
    estimated_workspace = 20 * features.nbytes + (128 * len(features) if features.ndim else 0)
    if estimated_workspace > MAX_WORKSPACE_BYTES:
        raise ValueError('Foot locking exceeds the numeric workspace budget; shorten the motion')
    original = recover_positions(features, parents, offsets, 'fk', root_origin=root_origin,
                                 check_cancel=check_cancel)
    output = features.copy()
    rotations, root = decode_features(features, parents)
    root += np.asarray(root_origin)
    rest, _ = forward_pose(parents, offsets, np.tile(np.eye(3), (len(parents), 1, 1)), offsets[0])
    ground = float(rest[:, 1].min())
    height = float(rest[0, 1] - ground)
    report = {'method': 'unimate.e5.independent.v1', 'segments': [], 'root_height': height,
              'median_padding': 'edge', 'boundary_frames': 5, 'damping': .01,
              'max_iterations': 40, 'max_angle': .2}
    if not contacts:
        return output, report
    segments, _ = contact_segments(original, contacts, ground_height=ground, root_height=height,
                                   check_cancel=check_cancel)
    used_ancestors = set()
    for contact in contacts:
        chain = set(limb_chain(parents, contact))
        if used_ancestors.intersection(chain):
            raise ValueError('Contact joint overrides overlap optimized limb chains; select distal joints')
        used_ancestors.update(chain)
    solved = rotations.copy()
    changed = np.zeros(features.shape[:2], dtype=bool)
    for segment in segments:
        check_cancel()
        joint, start, end = segment['joint'], segment['start'], segment['end']
        chain = limb_chain(parents, joint)
        weights = boundary_weights(end - start)
        residuals = []
        for frame in range(start, end):
            check_cancel()
            pose, result = solve_contact(parents, offsets, rotations[frame], root[frame],
                joint, chain, segment['anchor'], scale=height, check_cancel=check_cancel)
            residuals.append(result['residual'])
            weight = weights[frame - start]
            if weight == 0:
                continue
            for ancestor in chain:
                solved[frame, ancestor] = blend_rotation(rotations[frame, ancestor], pose[ancestor], weight)
                changed[frame, ancestor] = True
        report['segments'].append({'joint': int(joint), 'start': start, 'end': end,
            'anchor': segment['anchor'].tolist(), 'chain': chain,
            'max_solver_residual': max(residuals), 'skipped': not bool(chain)})
    positions = original.copy()
    facing = rotation_6d_matrices(features[:, 0, 3:9])
    for frame in np.flatnonzero(changed.any(axis=1)):
        check_cancel()
        positions[frame], _ = forward_pose(parents, offsets, solved[frame], root[frame])
        for child in range(1, len(parents)):
            if changed[frame, parents[child]]:
                rotation = solved[frame, parents[child]]
                output[frame, child, 3:9] = np.concatenate([rotation[:, 0], rotation[:, 1]])
        moved = np.any(positions[frame] != original[frame], axis=1)
        moved[0] = False
        centered = positions[frame, moved].copy()
        centered[:, [0, 2]] -= root[frame, [0, 2]]
        output[frame, moved, :3] = (facing[frame] @ centered.T).T
    delta = positions - original
    velocity_delta = np.einsum('tij,tkj->tki', facing[1:], np.diff(delta, axis=0))
    output[:-1, 1:, 9:12] += velocity_delta[:, 1:]
    check_cancel()
    if not np.isfinite(output).all():
        raise ValueError('Foot locking produced nonfinite features')
    return output, report
