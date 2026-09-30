"""GLB-preserving adapter for the pinned UniMate coordinate conventions.

Matrix equations implement the published 12-channel representation. Graph and
name conditioning call the licensed, pinned reference (see _vendor notices).
No Motion/Quaternions implementation is redistributed.
"""

import copy
import json
import struct
import numpy as np
from ._vendor import rig_topology as topology
from ._vendor.rig_names import clean_joint_name, post_process

UPSTREAM_REVISION = "5d6aabedd947297b5ba6706d8e9113e68c0c3e4f"
ADAPTER_REVISION = "gltf-preserving-v1"


def parse_glb(payload):
    document = None
    binary = b""
    cursor = 12
    while cursor < len(payload):
        size, kind = struct.unpack_from("<II", payload, cursor)
        cursor += 8
        value = payload[cursor : cursor + size]
        cursor += size
        if kind == 0x4E4F534A:
            document = json.loads(value)
        elif kind == 0x004E4942:
            binary = value
    if document is None:
        raise ValueError("GLB JSON is missing")
    return document, binary


def pack_glb(document, binary):
    body = json.dumps(
        document, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    body += b" " * (-len(body) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 28 + len(body) + len(binary))
        + struct.pack("<II", len(body), 0x4E4F534A)
        + body
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=np.float64)
    length = np.linalg.norm(q)
    if length < 1e-12:
        raise ValueError("Zero quaternion")
    x, y, z, w = np.asarray(q) / length
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quaternion(r):
    # Eigenvector formulation, glTF xyzw; sign made deterministic.
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[1, 0] + r[0, 1],
                    r[2, 0] + r[0, 2],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[1, 0] + r[0, 1],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[2, 1] + r[1, 2],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[2, 0] + r[0, 2],
                    r[2, 1] + r[1, 2],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r.trace()],
            ]
        )
        / 3
    )
    _, v = np.linalg.eigh(k)
    q = v[:, -1]
    return q if q[3] >= 0 else -q


def node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    m[:3, :3] = quaternion_matrix(node.get("rotation", [0, 0, 0, 1])) @ np.diag(
        node.get("scale", [1, 1, 1])
    )
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def world_matrices(doc):
    nodes = doc["nodes"]
    parents = [-1] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parents[c] = i
    local = np.array([node_matrix(n) for n in nodes])
    worlds = {}

    def evaluate(i):
        if i not in worlds:
            worlds[i] = local[i] if parents[i] < 0 else evaluate(parents[i]) @ local[i]
        return worlds[i]

    return np.array([evaluate(i) for i in range(len(nodes))]), local, parents


def rotation_part(matrix):
    r = np.asarray(matrix)[..., :3, :3]
    return r / np.linalg.norm(r, axis=-2, keepdims=True)


def _ordering(parents, offsets):
    children = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    sizes = {}

    def size(i):
        sizes[i] = 1 + sum(size(c) for c in children[i])
        return sizes[i]

    root = parents.index(-1)
    size(root)
    for cs in children:
        cs.sort(key=lambda j: (-sizes[j], np.linalg.norm(offsets[j])))
    order = [root]
    for j in order:
        order.extend(children[j])
    return order


def prepare_document(
    doc, facing, left_joint="", right_joint="", name="character", blender_rest=None
):
    worlds, local, node_parents = world_matrices(doc)
    joints = list(doc["skins"][0]["joints"])
    names = [doc["nodes"][j].get("name", f"joint_{j}") for j in joints]
    if len(set(names)) != len(names):
        raise ValueError("Joint names must be unique")
    parents = []
    for j in joints:
        p = node_parents[j]
        while p >= 0 and p not in joints:
            p = node_parents[p]
        parents.append(joints.index(p) if p >= 0 else -1)
    # World-applied rest is the same coordinate frame as upstream after Z->Y.
    rest = worlds[joints].copy()
    if blender_rest is not None:
        for i, n in enumerate(names):
            rest[i] = np.asarray(blender_rest[n])
    rest_rotation = rotation_part(rest)
    offsets = np.empty((len(joints), 3))
    for i, p in enumerate(parents):
        offsets[i] = (
            rest[i, :3, 3]
            if p < 0
            else rest_rotation[p].T @ (rest[i, :3, 3] - rest[p, :3, 3])
        )
    order = _ordering(parents, offsets)
    joints = [joints[i] for i in order]
    names = [names[i] for i in order]
    parents = np.array(
        [-1 if parents[i] < 0 else order.index(parents[i]) for i in order],
        dtype=np.int64,
    )
    rest = rest[order]
    offsets = offsets[order]
    rest_rotation = rest_rotation[order]
    face_idxs = [-1, -1]
    if facing == "joint_pair":
        if (
            left_joint not in names
            or right_joint not in names
            or left_joint == right_joint
        ):
            raise ValueError("Select two distinct existing left/right joints")
        face_idxs = [names.index(right_joint), names.index(left_joint)]
        across = rest[face_idxs[0], :3, 3] - rest[face_idxs[1], :3, 3]
        forward = np.cross([0, 1, 0], across)
        if np.linalg.norm(forward) < 1e-8:
            raise ValueError("Facing pair has no horizontal separation")
        angle = -np.arctan2(forward[0], forward[2])
    else:
        angles = {"+Z": 0.0, "-Z": np.pi, "+X": -np.pi / 2, "-X": np.pi / 2}
        if facing not in angles:
            raise ValueError("Unknown facing direction")
        angle = angles[facing]
    c, s = np.cos(angle), np.sin(angle)
    facing_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    adjacency = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p >= 0:
            length = np.linalg.norm(offsets[i])
            adjacency[i].append((p, length))
            adjacency[p].append((i, length))

    def farthest(start):
        distances = {start: 0.0}
        queue = [start]
        for u in queue:
            for v, w in adjacency[u]:
                if v not in distances:
                    distances[v] = distances[u] + w
                    queue.append(v)
        endpoint = max(distances, key=distances.get)
        return endpoint, distances[endpoint]

    diameter = farthest(farthest(0)[0])[1]
    if diameter <= 1e-8:
        raise ValueError("Degenerate skeleton: zero geodesic diameter")
    scale = 2.0 / diameter
    transformed = rest[:, :3, 3] @ facing_matrix.T
    shift = np.array([transformed[0, 0], transformed[:, 1].min(), transformed[0, 2]])
    pos = (transformed - shift) * scale
    # Blender's upstream extractor normalizes armature object scale. Recover
    # the actual source->canonical similarity from original GLB head distances.
    actual = worlds[joints, :3, 3]
    source_lengths = np.linalg.norm(actual[1:] - actual[parents[1:]], axis=-1)
    canonical_lengths = np.linalg.norm(pos[1:] - pos[parents[1:]], axis=-1)
    valid = source_lengths > 1e-8
    source_scale = float(np.median(canonical_lengths[valid] / source_lengths[valid]))
    sim = np.eye(4)
    sim[:3, :3] = source_scale * facing_matrix
    sim[:3, 3] = pos[0] - source_scale * facing_matrix @ actual[0]
    restored = actual @ sim[:3, :3].T + sim[:3, 3]
    if not np.allclose(restored, pos, atol=2e-5, rtol=2e-5):
        raise ValueError(
            "Blender rest hierarchy is not reversibly related to the original GLB by a similarity"
        )
    global_rot = facing_matrix @ rest_rotation
    local_rot = global_rot.copy()
    for i, p in enumerate(parents):
        if p >= 0:
            local_rot[i] = global_rot[p].T @ global_rot[i]
    offsets *= scale
    offsets[0] = pos[0]
    tpos_offsets = pos.copy()
    tpos_offsets[1:] = pos[1:] - pos[parents[1:]]
    relations, distances = topology.compute_edge_relations_and_distances(parents)
    chains = topology.compute_kinematic_chains(parents)
    padded = np.full((len(chains), max(map(len, chains))), -1, dtype=np.int64)
    for i, chain in enumerate(chains):
        padded[i, : len(chain)] = chain
    cond = dict(
        object_type=np.array(name),
        parents=parents,
        offsets=offsets,
        tpos_offsets=tpos_offsets,
        joint_names=np.array(names),
        clean_joint_names=np.array(
            [post_process(clean_joint_name(n, name)) for n in names]
        ),
        tpos_first_frame=pos,
        tpos_local_rotations=np.array(
            [matrix_quaternion(r)[[3, 0, 1, 2]] for r in local_rot]
        ),
        tpos_global_rotations=np.array(
            [matrix_quaternion(r)[[3, 0, 1, 2]] for r in global_rot]
        ),
        joint_relations=relations,
        joint_graph_dists=distances,
        joint_depths=topology.compute_joint_depths(parents),
        edge_indexs=topology.compute_edge_indexs(parents),
        spectral_feats=topology.compute_laplacian_eigenvectors(parents)[0],
        kinematic_chains=padded,
        scale_factor=np.array(scale),
        ground_height_mode=np.array("per_motion"),
        face_joint_idxs=np.array(face_idxs),
    )
    mapping = dict(
        joint_indices=joints,
        source_to_canonical=sim.tolist(),
        adapter_revision=ADAPTER_REVISION,
        upstream_revision=UPSTREAM_REVISION,
        facing=facing,
        left_joint=left_joint,
        right_joint=right_joint,
        quaternion_order="wxyz conditioning; xyzw glTF",
    )
    return cond, mapping


def decode_features(features, parents):
    features = np.asarray(features, dtype=np.float64)
    raw = features[..., 3:9]
    x = raw[..., :3]
    y = raw[..., 3:]
    xn = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(xn < 1e-8):
        raise ValueError("Motion contains degenerate 6D rotation")
    x = x / xn
    z = np.cross(x, y)
    zn = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(zn < 1e-8):
        raise ValueError("Motion contains collinear 6D rotation axes")
    z = z / zn
    y = np.cross(z, x)
    hml = np.stack([x, y, z], axis=-1)
    rotations = np.broadcast_to(np.eye(3), (*features.shape[:2], 3, 3)).copy()
    for j, p in enumerate(parents[1:], 1):
        rotations[:, p] = hml[:, j]
    velocity = np.zeros((len(features), 3))
    velocity[1:, 0] = features[:-1, 0, 9]
    velocity[1:, 2] = features[:-1, 0, 11]
    root = np.cumsum(np.einsum("tji,tj->ti", hml[:, 0], velocity), axis=0)
    root[:, 1] = features[:, 0, 1]
    return rotations, root


def animate_document(source, cond, mapping, features):
    document, binary = parse_glb(source)
    document = copy.deepcopy(document)
    binary = bytearray(binary)
    worlds, local, node_parents = world_matrices(document)
    joints = mapping["joint_indices"]
    sim = np.asarray(mapping["source_to_canonical"])
    facing = rotation_part(sim)
    rotations, root = decode_features(features, cond["parents"])
    if len(joints) != features.shape[1]:
        raise ValueError("Motion joint count does not match prepared skeleton")

    def accessor(values, kind):
        a = np.asarray(values, dtype="<f4")
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(a.tobytes())
        document.setdefault("bufferViews", []).append(
            {"buffer": 0, "byteOffset": offset, "byteLength": a.nbytes}
        )
        entry = {
            "bufferView": len(document["bufferViews"]) - 1,
            "componentType": 5126,
            "count": len(a),
            "type": kind,
        }
        if kind == "SCALAR":
            entry.update(min=[float(a.min())], max=[float(a.max())])
        document.setdefault("accessors", []).append(entry)
        return len(document["accessors"]) - 1

    time = accessor(np.arange(60) / 30, "SCALAR")
    animation = {"name": "UniMate", "channels": [], "samplers": []}

    def channel(j, path, values, kind):
        output = accessor(values, kind)
        i = len(animation["samplers"])
        animation["samplers"].append(
            {"input": time, "output": output, "interpolation": "LINEAR"}
        )
        animation["channels"].append(
            {"sampler": i, "target": {"node": j, "path": path}}
        )

    for k, j in enumerate(joints):
        canonical_rest = facing @ rotation_part(worlds[j])
        rest_local = rotation_part(local[j])
        applied = rest_local @ canonical_rest.T @ rotations[:, k] @ canonical_rest
        quats = np.array([matrix_quaternion(r) for r in applied])
        for f in range(1, 60):
            if np.dot(quats[f - 1], quats[f]) < 0:
                quats[f] *= -1
        if "matrix" in document["nodes"][j]:
            node = document["nodes"][j]
            node.pop("matrix")
            node["translation"] = local[j, :3, 3].tolist()
            node["rotation"] = matrix_quaternion(rest_local).tolist()
            node["scale"] = np.linalg.norm(local[j, :3, :3], axis=0).tolist()
        channel(j, "rotation", quats, "VEC4")
    source_root = (
        np.concatenate([root, np.ones((60, 1))], axis=1) @ np.linalg.inv(sim).T
    )
    p = node_parents[joints[0]]
    if p >= 0:
        source_root = source_root @ np.linalg.inv(worlds[p]).T
    channel(joints[0], "translation", source_root[:, :3], "VEC3")
    document["animations"] = [animation]
    document["buffers"][0]["byteLength"] = len(binary)
    return pack_glb(document, bytes(binary))
