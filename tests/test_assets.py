"""Hand-built glTF fixtures; no Blender or external assets are involved."""

import copy
import base64
import json
import struct
import zlib

import numpy as np
import pytest

from unimate_pack.assets import validate_glb


def pack_glb(doc, binary):
    encoded = json.dumps(doc, allow_nan=False, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, len(chunks) + 12) + chunks


def synthetic_doc():
    binary = bytearray()
    views, accessors = [], []

    def append(array, component, kind):
        binary.extend(b"\0" * (-len(binary) % 4))
        data = array.tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary.extend(data)
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": component,
                "count": len(array),
                "type": kind,
            }
        )
        return len(accessors) - 1

    position = append(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4"), 5126, "VEC3")
    joints = append(
        np.array([[0, 0, 0, 0], [1, 0, 0, 0], [4, 0, 0, 0]], "<u2"), 5123, "VEC4"
    )
    weights = append(np.array([[1, 0, 0, 0]] * 3, "<f4"), 5126, "VEC4")
    indices = append(np.array([0, 1, 2], "<u2"), 5123, "SCALAR")
    ibm = append(np.tile(np.eye(4, dtype="<f4").reshape(1, 16), (5, 1)), 5126, "MAT4")
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
        "nodes": [{"name": f"joint_{i}", "children": [i + 1]} for i in range(4)]
        + [{"name": "joint_4"}, {"name": "mesh", "mesh": 0, "skin": 0}],
        "skins": [
            {"joints": [0, 1, 2, 3, 4], "inverseBindMatrices": ibm, "skeleton": 0}
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position,
                            "JOINTS_0": joints,
                            "WEIGHTS_0": weights,
                        },
                        "indices": indices,
                    }
                ]
            }
        ],
        "scenes": [{"nodes": [0, 5]}],
        "scene": 0,
    }
    return doc, binary


def synthetic_glb():
    return pack_glb(*synthetic_doc())


def test_valid_rig_retains_original_joint_order_and_identity():
    result = validate_glb(synthetic_glb())
    assert result["skins"][0]["joints"] == [0, 1, 2, 3, 4]
    assert result["nodes"][4]["name"] == "joint_4"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["buffers"][0].update(uri="https://example.invalid/mesh.bin"),
        lambda d: d["bufferViews"][0].update(byteLength=100000),
        lambda d: d["accessors"][0].update(count=100000),
        lambda d: d["accessors"][0].update(bufferView=999),
        lambda d: d["skins"].append(copy.deepcopy(d["skins"][0])),
        lambda d: d["skins"][0].update(joints=[0, 1, 2, 3, 99]),
        lambda d: d["skins"][0].update(joints=[0, 1, 2, 3, 3]),
        lambda d: d["skins"][0].update(joints=[0, 1, 2, 3, 5]),
        lambda d: d["nodes"][4].update(children=[0]),
        lambda d: d["nodes"][4].update(scale=[-1, -1, -1]),
        lambda d: d["nodes"][4].update(scale=[1, 2, 1]),
        lambda d: d["nodes"][4].update(
            matrix=[1, 0, 0, 0, 0.2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        ),
        lambda d: d["meshes"][0]["primitives"][0].update(mode=1),
        lambda d: d["meshes"][0]["primitives"][0].update(targets=[{}]),
        lambda d: d["meshes"][0]["primitives"][0]["attributes"].pop("WEIGHTS_0"),
        lambda d: d.update(extensionsRequired=["KHR_draco_mesh_compression"]),
        lambda d: d["nodes"][0].update(extensions={"unrecognized": {}}),
        lambda d: d.update(images=[{"uri": "../secret.png"}]),
        lambda d: d.update(textures=[{"source": 99}]),
        lambda d: d["scenes"][0].update(nodes=[5]),
    ],
)
def test_malformed_and_unsupported_rigs_are_rejected(mutate):
    doc, binary = synthetic_doc()
    mutate(doc)
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


@pytest.mark.parametrize(
    "accessor, fmt, value",
    [(0, "<f", float("nan")), (1, "<H", 5), (2, "<f", -1.0), (3, "<H", 3)],
)
def test_binary_numeric_and_skin_indices_are_checked(accessor, fmt, value):
    doc, binary = synthetic_doc()
    view = doc["bufferViews"][doc["accessors"][accessor]["bufferView"]]
    struct.pack_into(fmt, binary, view["byteOffset"], value)
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b[:10],
        lambda b: b + b"junk",
        lambda b: b[:4] + struct.pack("<I", 1) + b[8:],
    ],
)
def test_malformed_glb_header_is_rejected(change):
    with pytest.raises(ValueError):
        validate_glb(change(synthetic_glb()))


def test_static_scene_ancestor_translation_uniform_scale_and_rotation_are_allowed():
    doc, binary = synthetic_doc()
    doc["nodes"].append(
        {
            "translation": [2, 4, -1],
            "scale": [1.7, 1.7, 1.7],
            "rotation": [0, 0, 1, 0],
            "children": [0, 5],
        }
    )
    doc["scenes"][0]["nodes"] = [6]
    result = validate_glb(pack_glb(doc, binary))
    assert result["nodes"][6]["translation"] == [2, 4, -1]
    assert result["nodes"][6]["scale"] == [1.7, 1.7, 1.7]


def test_interleaved_position_accessor_is_read_using_declared_stride():
    doc, binary = synthetic_doc()
    offset = len(binary)
    vertices = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], "<f4")
    binary.extend(vertices.tobytes())
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["bufferViews"][0] = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": vertices.nbytes,
        "byteStride": 16,
    }
    validate_glb(pack_glb(doc, binary))
    struct.pack_into("<f", binary, offset + 16, float("nan"))
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


def test_integer_normalized_skin_weights_are_supported():
    doc, binary = synthetic_doc()
    offset = len(binary)
    weights = np.array([[255, 0, 0, 0]] * 3, "u1")
    binary.extend(weights.tobytes())
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["bufferViews"][2] = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": weights.nbytes,
    }
    doc["accessors"][2].update(componentType=5121, normalized=True)
    validate_glb(pack_glb(doc, binary))


def animation_doc(path="translation"):
    doc, binary = synthetic_doc()
    for data, kind in (
        (np.array([[0], [1 / 30]], "<f4"), "SCALAR"),
        (np.array([[0, 0, 0], [1, 0, 0]], "<f4"), "VEC3"),
    ):
        offset = len(binary)
        binary.extend(data.tobytes())
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes}
        )
        doc["accessors"].append(
            {
                "bufferView": len(doc["bufferViews"]) - 1,
                "componentType": 5126,
                "type": kind,
                "count": 2,
            }
        )
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["animations"] = [
        {
            "samplers": [{"input": 5, "output": 6}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": path}}],
        }
    ]
    return doc, binary


def test_source_translation_animation_is_validated_but_does_not_replace_rest_pose():
    doc, binary = animation_doc()
    result = validate_glb(pack_glb(doc, binary))
    assert "translation" not in result["nodes"][0]
    assert result["animations"][0]["channels"][0]["target"]["path"] == "translation"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["animations"][0]["channels"][0]["target"].update(path="scale"),
        lambda d: d["animations"][0]["channels"][0]["target"].update(node=999),
        lambda d: d["animations"][0]["channels"][0].update(sampler=999),
        lambda d: d["animations"][0]["samplers"][0].update(output=999),
        lambda d: d["animations"][0]["samplers"][0].update(interpolation="unsafe"),
    ],
)
def test_invalid_animation_channels_are_rejected(change):
    doc, binary = animation_doc()
    change(doc)
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


def test_glb_size_limit_is_enforced_before_parsing(monkeypatch):
    from unimate_pack import assets

    monkeypatch.setattr(assets, "MAX_GLB_BYTES", 100)
    with pytest.raises(ValueError):
        validate_glb(synthetic_glb())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["accessors"][0].update(componentType=5126.0),
        lambda d: d["bufferViews"][0].update(buffer=False),
        lambda d: d["meshes"][0]["primitives"][0].update(mode=4.0),
        lambda d: d["nodes"][5].update(weights=[0]),
        lambda d: d["nodes"][0].update(translation=[1e308, 0, 0]),
    ],
)
def test_malformed_numeric_types_and_unsupported_node_values_are_rejected(change):
    doc, binary = synthetic_doc()
    change(doc)
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


@pytest.mark.parametrize(
    "attribute, index",
    [("NORMAL", 1), ("TANGENT", 0), ("TEXCOORD_0", 0), ("COLOR_0", 1)],
)
def test_vertex_attributes_use_supported_semantic_formats(attribute, index):
    doc, binary = synthetic_doc()
    doc["meshes"][0]["primitives"][0]["attributes"][attribute] = index
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


@pytest.mark.parametrize(
    "material",
    [
        {"pbrMetallicRoughness": {"baseColorFactor": "invalid"}},
        {"pbrMetallicRoughness": {"baseColorFactor": [1, 2, 0, 1]}},
        {"pbrMetallicRoughness": {"metallicFactor": -1}},
        {"pbrMetallicRoughness": {"roughnessFactor": 2}},
        {"emissiveFactor": [0, -1, 0]},
        {"alphaMode": "unknown"},
        {"doubleSided": 1},
    ],
)
def test_invalid_material_factors_are_rejected(material):
    doc, binary = synthetic_doc()
    doc["materials"] = [material]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


def png(width=1, height=1):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\x44\x88\xcc"))
        + chunk(b"IEND", b"")
    )


def with_image(doc, binary, raw, mime="image/png", embedded_uri=True):
    if embedded_uri:
        doc["images"] = [
            {"uri": f"data:{mime};base64," + base64.b64encode(raw).decode()}
        ]
    else:
        offset = len(binary)
        binary.extend(raw)
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        )
        doc["images"] = [{"bufferView": len(doc["bufferViews"]) - 1, "mimeType": mime}]
        doc["buffers"][0]["byteLength"] = len(binary)
    doc["textures"] = [{"source": 0}]


@pytest.mark.parametrize("embedded_uri", [True, False])
@pytest.mark.parametrize(
    "raw, mime",
    [
        (png(1_000_000_000, 1_000_000_000), "image/png"),
        (b"not a png", "image/png"),
        (b"not a jpeg", "image/jpeg"),
    ],
)
def test_image_dimensions_and_signatures_are_checked_before_decoding(
    embedded_uri, raw, mime
):
    doc, binary = synthetic_doc()
    with_image(doc, binary, raw, mime, embedded_uri)
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))


def test_valid_embedded_texture_requires_matching_vertex_texture_coordinates():
    doc, binary = synthetic_doc()
    with_image(doc, binary, png())
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))
    uv = np.array([[0, 0], [1, 0], [0, 1]], "<f4")
    offset = len(binary)
    binary.extend(uv.tobytes())
    doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": uv.nbytes}
    )
    doc["accessors"].append(
        {
            "bufferView": len(doc["bufferViews"]) - 1,
            "componentType": 5126,
            "type": "VEC2",
            "count": 3,
        }
    )
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 5
    validate_glb(pack_glb(doc, binary))
    doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"] = 1
    with pytest.raises(ValueError):
        validate_glb(pack_glb(doc, binary))
