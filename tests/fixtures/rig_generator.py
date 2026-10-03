"""Original synthetic fixtures, MIT; no third-party geometry or textures."""

import json
import struct
import zlib
import numpy as np


def pack_glb(doc, binary):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary))
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def synthetic_glb(branching=True, *, legged=False):
    parents = [-1, 0, 0, 1, 2, 3, 4] if branching else [-1, 0, 1, 2, 3]
    positions = (
        [
            [0, 1, 0],
            [-0.35, 0.4, 0],
            [0.35, 0.4, 0],
            [0, 0.5, 0.05],
            [0, 0.5, 0.05],
            [0, 0.35, 0],
            [0, 0.35, 0],
        ]
        if branching
        else [
            [0, 0.5, 0],
            [0, 0.3, 0.15],
            [0.1, 0.3, 0],
            [0, 0.3, -0.15],
            [0.1, 0.3, 0],
        ]
    )
    j = len(parents)
    if legged:
        if not branching:
            raise ValueError('Legged fixture requires branching topology')
        positions = [[0, 1, 0], [-.3, -.1, 0], [.3, -.1, 0],
                     [0, -.5, .15], [0, -.5, .15], [0, -.4, -.15], [0, -.4, -.15]]
    names = ['Root', 'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee', 'LeftFoot', 'RightFoot']
    # Nonidentity bind rotations and a transformed uniform scene ancestor.
    angle = 0. if legged else 0.25
    q = [0, 0, float(np.sin(angle / 2)), float(np.cos(angle / 2))]
    nodes = [
        {
            "name": "SceneTransform",
            "translation": [2, 0, -3],
            "scale": [1.7] * 3,
            "children": [1, j + 1],
        }
    ]
    local, world = [], []
    for i, p in enumerate(parents):
        r = (
            np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            if i == 1
            else np.eye(3)
        )
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = positions[i]
        local.append(m)
        world.append(m if p < 0 else world[p] @ m)
        n = {
            "name": names[i] if legged else f"Joint_{i}",
            "translation": positions[i],
            "children": [k + 1 for k, x in enumerate(parents) if x == i],
        }
        if i == 1:
            n["rotation"] = q
        nodes.append(n)
    nodes.append({"name": "SkinnedTriangles", "mesh": 0, "skin": 0})
    vertices = []
    joints = []
    for i, w in enumerate(world):
        for d in [[-0.07, 0, 0], [0.07, 0, 0], [0, 0.1, 0.04]]:
            vertices.append((w @ np.array([*d, 1]))[:3])
            joints.append([i, max(parents[i], 0), 0, 0])
    doc = {
        "asset": {"version": "2.0", "generator": "UniMate original MIT synthetic rig"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "skins": [{"joints": list(range(1, j + 1)), "skeleton": 1}],
        "materials": [
            {
                "name": "FixtureBlue",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.15, 0.35, 0.8, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 0.7,
                },
            }
        ],
        "bufferViews": [],
        "accessors": [],
        "buffers": [{"byteLength": 0}],
    }
    binary = bytearray()

    def add(a, typ, ctype):
        binary.extend(b"\0" * (-len(binary) % 4))
        start = len(binary)
        binary.extend(a.tobytes())
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": start, "byteLength": a.nbytes}
        )
        ac = {
            "bufferView": len(doc["bufferViews"]) - 1,
            "componentType": ctype,
            "count": len(a),
            "type": typ,
        }
        if typ == "VEC3":
            ac.update(min=a.min(0).tolist(), max=a.max(0).tolist())
        doc["accessors"].append(ac)
        return len(doc["accessors"]) - 1

    pa = add(np.array(vertices, dtype="<f4"), "VEC3", 5126)
    ja = add(np.array(joints, dtype="<u2"), "VEC4", 5123)
    weights = np.tile(np.array([0.7, 0.3, 0, 0], dtype="<f4"), (len(vertices), 1))
    weights[:3] = [1, 0, 0, 0]
    wa = add(weights, "VEC4", 5126)
    ia = add(np.arange(len(vertices), dtype="<u2"), "SCALAR", 5123)
    uv = add(
        np.tile(np.array([[0, 0], [1, 0], [0.5, 1]], dtype="<f4"), (j, 1)), "VEC2", 5126
    )

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\x44\x88\xcc"))
        + chunk(b"IEND", b"")
    )
    binary.extend(b"\0" * (-len(binary) % 4))
    start = len(binary)
    binary.extend(png)
    doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": start, "byteLength": len(png)}
    )
    doc["images"] = [
        {"bufferView": len(doc["bufferViews"]) - 1, "mimeType": "image/png"}
    ]
    doc["textures"] = [{"source": 0}]
    doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    ba = add(
        np.array([np.linalg.inv(w).T.reshape(-1) for w in world], dtype="<f4"),
        "MAT4",
        5126,
    )
    doc["skins"][0]["inverseBindMatrices"] = ba
    doc["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": pa,
                        "JOINTS_0": ja,
                        "WEIGHTS_0": wa,
                        "TEXCOORD_0": uv,
                    },
                    "indices": ia,
                    "material": 0,
                    "mode": 4,
                }
            ]
        }
    ]
    doc["buffers"][0]["byteLength"] = len(binary)
    return pack_glb(doc, bytes(binary))


def legged_glb():
    """Original bent-leg seven-joint fixture with two ground contact joints."""
    return synthetic_glb(True, legged=True)
