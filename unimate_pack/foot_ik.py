"""Numeric limb IK helpers; local increments multiply on the right.

Inputs here are validated numeric arrays at the owning motion boundary.
Offsets and rotations use the parent-relative export convention.
"""

import numpy as np

from .rig_math import matrix_quaternion, quaternion_matrix


def limb_chain(parents, contact):
    """Up to three movable parents, stopping before root or an unrelated branch."""
    children = np.bincount(parents[1:], minlength=len(parents))
    chain = []
    joint = int(parents[contact])
    while joint > 0 and len(chain) < 3:
        if children[joint] > 1:
            break
        chain.append(joint)
        joint = int(parents[joint])
    return chain


def boundary_weights(length):
    """Five-frame linear ramps including zero at each segment endpoint."""
    frames = np.arange(length)
    return np.minimum(np.minimum(frames, length - 1 - frames) / 4., 1.)


def blend_rotation(source, target, weight):
    """Shortest-arc quaternion interpolation with exact endpoint matrices."""
    if weight == 0:
        return source.copy()
    if weight == 1:
        return target.copy()
    first, second = matrix_quaternion(source), matrix_quaternion(target)
    dot = float(first @ second)
    if dot < 0:
        second, dot = -second, -dot
    dot = np.clip(dot, 0., 1.)
    if dot > 0.9995:
        quaternion = (1 - weight) * first + weight * second
    else:
        angle = np.arccos(dot)
        quaternion = (np.sin((1 - weight) * angle) * first
                      + np.sin(weight * angle) * second) / np.sin(angle)
    return quaternion_matrix(quaternion)


def rotation_increment(vector):
    """SO(3) exponential with stable coefficients near zero."""
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(vector)
    a = np.sinc(angle / np.pi)
    b = 0.5 * np.sinc(angle / (2 * np.pi)) ** 2
    return np.eye(3) + a * skew + b * (skew @ skew)


def forward_pose(parents, offsets, rotations, root):
    """Evaluate one pose with a prescribed root position."""
    positions = np.empty((len(parents), 3), dtype=np.float64)
    worlds = np.empty((len(parents), 3, 3), dtype=np.float64)
    positions[0] = root
    worlds[0] = rotations[0]
    for joint in range(1, len(parents)):
        parent = parents[joint]
        positions[joint] = positions[parent] + worlds[parent] @ offsets[joint]
        worlds[joint] = worlds[parent] @ rotations[joint]
    return positions, worlds


def contact_jacobian(positions, worlds, contact, joints):
    """Derivative of contact position w.r.t. each ancestor's local XYZ angle."""
    return np.concatenate([
        np.cross(worlds[joint].T, positions[contact] - positions[joint]).T
        for joint in joints
    ], axis=1) if joints else np.zeros((3, 0), dtype=np.float64)


def solve_contact(parents, offsets, rotations, root, contact, joints, target, *,
                  scale, damping=0.01, iterations=40, tolerance=1e-6,
                  max_angle=0.2, check_cancel=lambda: None):
    """Bounded damped least squares on validated non-root limb ancestors.

    Distances are normalized by rest root height. Return the best pose found,
    including residual for unreachable targets; never change root or offsets.
    """
    solved = np.asarray(rotations, dtype=np.float64).copy()
    best = solved.copy()
    best_error = float('inf')
    completed = 0
    for step in range(iterations + 1):
        check_cancel()
        positions, worlds = forward_pose(parents, offsets, solved, root)
        residual = (target - positions[contact]) / scale
        error = float(np.linalg.norm(residual))
        if error < best_error:
            best_error, best = error, solved.copy()
        if error <= tolerance or step == iterations or not joints:
            break
        jacobian = contact_jacobian(positions, worlds, contact, joints) / scale
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping ** 2 * np.eye(3), residual)
        for index, joint in enumerate(joints):
            angle = delta[3 * index:3 * index + 3]
            length = np.linalg.norm(angle)
            if length > max_angle:
                angle = angle * (max_angle / length)
            solved[joint] = solved[joint] @ rotation_increment(angle)
        completed = step + 1
    check_cancel()
    return best, {'residual': best_error * scale, 'iterations': completed}
