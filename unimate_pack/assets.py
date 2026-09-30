"""Validate the supported, self-contained rigged GLB subset before Blender import.

Binary layout and accessor checks follow the Khronos glTF 2.0 specification:
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
No files, URLs, image decoders, or Blender modules are opened here.
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
import zlib

import numpy as np

MAX_GLB_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_TOTAL_IMAGE_PIXELS = 64 * 1024 * 1024
_COMPONENT = {
    5120: "i1",
    5121: "u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}
_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _index(value, values, label):
    _integer(value, label)
    if value >= len(values):
        raise ValueError(f"{label} index is out of range")
    return value


def _objects(doc, key, limit=4096):
    values = doc.get(key, [])
    if (
        type(values) is not list
        or len(values) > limit
        or any(type(v) is not dict for v in values)
    ):
        raise ValueError(f"Invalid or oversized glTF {key}")
    return values


def _vector(value, width, label):
    if type(value) is not list or len(value) != width:
        raise ValueError(f"Invalid {label} shape")
    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or abs(v) > float(np.finfo(np.float32).max)
        for v in value
    ):
        raise ValueError(f"Invalid or nonfinite {label}")
    return np.asarray(value, dtype=np.float64)


def _checked_json(value, depth=0):
    if depth > 64:
        raise ValueError("GLB JSON nesting exceeds limit")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("GLB contains nonfinite JSON values")
    if type(value) is dict:
        if value.get("extensions"):
            raise ValueError(
                "Unsupported glTF extensions; export standard uncompressed PBR glTF"
            )
        for item in value.values():
            _checked_json(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _checked_json(item, depth + 1)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate GLB JSON key")
        result[key] = value
    return result


def _parse(glb):
    if type(glb) is not bytes or not 20 <= len(glb) <= MAX_GLB_BYTES:
        raise ValueError("GLB must be bounded nonempty bytes")
    magic, version, length = struct.unpack_from("<III", glb)
    if magic != 0x46546C67 or version != 2 or length != len(glb):
        raise ValueError("Invalid GLB 2.0 header or length")
    chunks, offset = [], 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<II", glb, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise ValueError("Invalid GLB chunk alignment or bounds")
        chunks.append((kind, glb[offset : offset + size]))
        offset += size
    if len(chunks) != 2 or chunks[0][0] != 0x4E4F534A or chunks[1][0] != 0x004E4942:
        raise ValueError("Supported GLB requires exactly JSON then embedded BIN chunks")
    if len(chunks[0][1]) > MAX_JSON_BYTES:
        raise ValueError("GLB JSON exceeds size limit")
    try:
        doc = json.loads(
            chunks[0][1].decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid GLB JSON") from exc
    if (
        type(doc) is not dict
        or type(doc.get("asset")) is not dict
        or doc["asset"].get("version") != "2.0"
    ):
        raise ValueError("GLB asset version must be 2.0")
    if doc.get("extensionsUsed") or doc.get("extensionsRequired"):
        raise ValueError("Unsupported glTF extensions")
    _checked_json(doc)
    return doc, chunks[1][1]


def _accessors(doc, binary):
    buffers = _objects(doc, "buffers")
    if len(buffers) != 1 or "uri" in buffers[0]:
        raise ValueError("GLB must use one embedded buffer without external URI")
    size = _integer(buffers[0].get("byteLength"), "buffer byteLength", 1)
    if size > len(binary) or len(binary) - size > 3:
        raise ValueError("Embedded buffer byteLength does not match BIN chunk")
    views = _objects(doc, "bufferViews")
    for view in views:
        if type(view.get("buffer")) is not int or view["buffer"] != 0:
            raise ValueError("Invalid bufferView buffer index")
        start = _integer(view.get("byteOffset", 0), "bufferView byteOffset")
        count = _integer(view.get("byteLength"), "bufferView byteLength", 1)
        if start + count > size:
            raise ValueError("bufferView exceeds embedded buffer")
        if "byteStride" in view:
            stride = _integer(view["byteStride"], "bufferView byteStride", 4)
            if stride > 252 or stride % 4:
                raise ValueError("Invalid bufferView stride")
        if "target" in view and view["target"] not in (34962, 34963):
            raise ValueError("Invalid bufferView target")
    accessors, decoded, total = _objects(doc, "accessors"), [], 0
    for accessor in accessors:
        if "sparse" in accessor:
            raise ValueError(
                "Sparse accessors are unsupported; export dense glTF arrays"
            )
        view = views[_index(accessor.get("bufferView"), views, "accessor bufferView")]
        kind, component = accessor.get("type"), accessor.get("componentType")
        if (
            kind not in _WIDTH
            or type(component) is not int
            or component not in _COMPONENT
        ):
            raise ValueError("Unsupported accessor type or componentType")
        if kind.startswith("MAT") and component != 5126:
            raise ValueError("Only float32 matrix accessors are supported")
        dtype, width = np.dtype(_COMPONENT[component]), _WIDTH[kind]
        count = _integer(accessor.get("count"), "accessor count", 1)
        start = _integer(accessor.get("byteOffset", 0), "accessor byteOffset")
        item = width * dtype.itemsize
        stride = view.get("byteStride", item)
        absolute = view.get("byteOffset", 0) + start
        total += count * item
        if (
            total > MAX_DECODED_BYTES
            or stride < item
            or stride % dtype.itemsize
            or start % dtype.itemsize
            or absolute % dtype.itemsize
            or start + (count - 1) * stride + item > view["byteLength"]
        ):
            raise ValueError("Accessor alignment, stride, size, or bounds are invalid")
        normalized = accessor.get("normalized", False)
        if type(normalized) is not bool or normalized and component in (5125, 5126):
            raise ValueError("Invalid accessor normalization")
        array = np.ndarray(
            (count, width),
            dtype=dtype,
            buffer=binary,
            offset=absolute,
            strides=(stride, dtype.itemsize),
        )
        if component == 5126 and not np.isfinite(array).all():
            raise ValueError("Accessor contains nonfinite numeric values")
        for key in ("min", "max"):
            if key in accessor:
                _vector(accessor[key], width, f"accessor {key}")
        if (
            "min" in accessor
            and "max" in accessor
            and np.any(np.asarray(accessor["min"]) > accessor["max"])
        ):
            raise ValueError("Invalid accessor min/max bounds")
        decoded.append(array)
    return accessors, decoded, views


def _hierarchy(doc):
    nodes = _objects(doc, "nodes", 2048)
    if not nodes:
        raise ValueError("Rig GLB requires nodes")
    parents = [-1] * len(nodes)
    for parent, node in enumerate(nodes):
        if "weights" in node:
            raise ValueError("Node morph weights are unsupported")
        children = node.get("children", [])
        if type(children) is not list or len(set(children)) != len(children):
            raise ValueError("Invalid or duplicate node children")
        for child in children:
            _index(child, nodes, "node child")
            if child == parent or parents[child] != -1:
                raise ValueError(
                    "Node hierarchy contains multiple parents or self-cycle"
                )
            parents[child] = parent
        if "matrix" in node:
            if any(key in node for key in ("translation", "rotation", "scale")):
                raise ValueError("Node cannot combine matrix and TRS transforms")
            matrix = _vector(node["matrix"], 16, "node matrix").reshape(4, 4).T
            linear = matrix[:3, :3]
            scales = np.linalg.norm(linear, axis=0)
            if (
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
                or np.any(scales <= 1e-10)
                or np.linalg.det(linear) <= 0
                or not np.allclose(scales, scales[0], rtol=1e-5, atol=1e-7)
                or not np.allclose(
                    (linear / scales).T @ (linear / scales), np.eye(3), atol=1e-5
                )
            ):
                raise ValueError(
                    "Unsupported matrix shear, negative, nonuniform, or singular scale"
                )
        else:
            _vector(node.get("translation", [0, 0, 0]), 3, "translation")
            rotation = _vector(node.get("rotation", [0, 0, 0, 1]), 4, "rotation")
            scale = _vector(node.get("scale", [1, 1, 1]), 3, "scale")
            if not np.isclose(np.linalg.norm(rotation), 1, atol=1e-4):
                raise ValueError("Node rotation must be a unit xyzw quaternion")
            if np.any(scale <= 0) or not np.allclose(
                scale, scale[0], rtol=1e-5, atol=1e-7
            ):
                raise ValueError("Unsupported negative, zero, or nonuniform node scale")
    for node in range(len(nodes)):
        seen, cursor = set(), node
        while cursor != -1:
            if cursor in seen:
                raise ValueError("Cyclic node hierarchy")
            seen.add(cursor)
            cursor = parents[cursor]
    scenes = _objects(doc, "scenes", 128)
    selected = _index(doc.get("scene", 0), scenes, "scene")
    roots = scenes[selected].get("nodes", [])
    if type(roots) is not list or len(set(roots)) != len(roots):
        raise ValueError("Invalid scene root nodes")
    reachable, pending = set(), list(roots)
    for root in roots:
        _index(root, nodes, "scene root")
        if parents[root] != -1:
            raise ValueError("Scene root node has a parent")
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(nodes[node].get("children", []))
    return nodes, parents, reachable


def _shape(accessors, index, kind, components, label, count=None):
    _index(index, accessors, label)
    accessor = accessors[index]
    if (
        accessor.get("type") != kind
        or accessor.get("componentType") not in components
        or count is not None
        and accessor["count"] != count
    ):
        raise ValueError(f"Unsupported {label} accessor shape, dtype, or count")


def _vertex_attributes(attributes, accessors, arrays):
    for semantic, index in attributes.items():
        accessor = accessors[index]
        if semantic in ("POSITION", "JOINTS_0", "WEIGHTS_0"):
            continue
        if semantic in ("NORMAL", "TANGENT"):
            _shape(
                accessors,
                index,
                "VEC3" if semantic == "NORMAL" else "VEC4",
                (5126,),
                semantic,
            )
            if not np.allclose(
                np.linalg.norm(arrays[index][:, :3], axis=1), 1, atol=1e-3
            ):
                raise ValueError(f"{semantic} vectors must have unit length")
            if (
                semantic == "TANGENT"
                and not np.isin(arrays[index][:, 3], (-1, 1)).all()
            ):
                raise ValueError("Tangent handedness must be -1 or 1")
        elif re.fullmatch(r"TEXCOORD_[0-9]+", semantic) or re.fullmatch(
            r"COLOR_[0-9]+", semantic
        ):
            kind = "VEC2" if semantic.startswith("TEXCOORD_") else accessor["type"]
            if semantic.startswith("COLOR_") and kind not in ("VEC3", "VEC4"):
                raise ValueError("Vertex colors require VEC3 or VEC4")
            _shape(accessors, index, kind, (5121, 5123, 5126), semantic)
            if accessor["componentType"] != 5126 and not accessor.get(
                "normalized", False
            ):
                raise ValueError(f"Integer {semantic} attributes must be normalized")
            if semantic.startswith("COLOR_") and accessor["componentType"] == 5126:
                if np.any(arrays[index] < 0) or np.any(arrays[index] > 1):
                    raise ValueError("Vertex color components must be in [0,1]")
        elif not semantic.startswith("_"):
            raise ValueError(f"Unsupported vertex semantic {semantic}")


def _texture_infos(material):
    for owner, keys in (
        (material, ("normalTexture", "occlusionTexture", "emissiveTexture")),
        (
            material.get("pbrMetallicRoughness", {}),
            ("baseColorTexture", "metallicRoughnessTexture"),
        ),
    ):
        if type(owner) is not dict:
            raise ValueError("Invalid PBR material")
        for key in keys:
            if key in owner:
                if type(owner[key]) is not dict:
                    raise ValueError("Invalid material texture")
                yield key, owner[key]


def _skins_meshes(doc, accessors, arrays, nodes, parents, reachable):
    skins = _objects(doc, "skins")
    if len(skins) != 1:
        raise ValueError("Exactly one skin is supported")
    joints = skins[0].get("joints")
    if type(joints) is not list or not 5 <= len(joints) <= 70:
        raise ValueError("Supported rig requires 5–70 skin joints")
    for joint in joints:
        _index(joint, nodes, "skin joint")
        if joint not in reachable:
            raise ValueError("Skin joint is outside the active scene")
    if len(set(joints)) != len(joints):
        raise ValueError("Skin has duplicate joint identities")
    joint_set, roots = set(joints), []
    for joint in joints:
        cursor = parents[joint]
        while cursor != -1 and cursor not in joint_set:
            cursor = parents[cursor]
        if cursor == -1:
            roots.append(joint)
    if len(roots) != 1:
        raise ValueError("Skin joints must form one connected deforming skeleton")
    if "skeleton" in skins[0]:
        skeleton = _index(skins[0]["skeleton"], nodes, "skin skeleton")
        for joint in joints:
            cursor = joint
            while cursor != skeleton and cursor != -1:
                cursor = parents[cursor]
            if cursor == -1:
                raise ValueError("Skin skeleton must be ancestor of all skin joints")
    if "inverseBindMatrices" in skins[0]:
        index = skins[0]["inverseBindMatrices"]
        _shape(accessors, index, "MAT4", (5126,), "inverse bind matrices", len(joints))
        matrices = arrays[index].reshape(-1, 4, 4).transpose(0, 2, 1)
        if not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1e-5) or np.any(
            np.abs(np.linalg.det(matrices)) < 1e-12
        ):
            raise ValueError(
                "Inverse bind matrices must be finite invertible affine matrices"
            )
    meshes = _objects(doc, "meshes")
    material_count = len(_objects(doc, "materials"))
    used = set()
    for index, node in enumerate(nodes):
        if "skin" in node and (
            type(node["skin"]) is not int or node["skin"] != 0 or "mesh" not in node
        ):
            raise ValueError("Invalid node skin reference")
        if "mesh" in node:
            mesh = _index(node["mesh"], meshes, "node mesh")
            if node.get("skin") != 0 or index not in reachable:
                raise ValueError(
                    "All meshes must belong to the single skin in the active scene"
                )
            used.add(mesh)
    if not used or used != set(range(len(meshes))):
        raise ValueError("GLB requires skinned meshes with no unused mesh content")
    for mesh in meshes:
        if "weights" in mesh:
            raise ValueError("Morph weights are unsupported")
        primitives = mesh.get("primitives")
        if type(primitives) is not list or not primitives or len(primitives) > 2048:
            raise ValueError("Invalid mesh primitives")
        for primitive in primitives:
            if (
                type(primitive) is not dict
                or type(primitive.get("mode", 4)) is not int
                or primitive.get("mode", 4) != 4
                or "targets" in primitive
            ):
                raise ValueError("Only triangles without morph targets are supported")
            attributes = primitive.get("attributes")
            if (
                type(attributes) is not dict
                or not {"POSITION", "JOINTS_0", "WEIGHTS_0"} <= attributes.keys()
            ):
                raise ValueError(
                    "Skinned primitive requires POSITION, JOINTS_0, and WEIGHTS_0"
                )
            if any(
                key.startswith(("JOINTS_", "WEIGHTS_"))
                and key not in ("JOINTS_0", "WEIGHTS_0")
                for key in attributes
            ):
                raise ValueError("Only four skin influences per vertex are supported")
            position, joint, weight = (
                attributes[key] for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")
            )
            _shape(accessors, position, "VEC3", (5126,), "POSITION")
            count = accessors[position]["count"]
            for index in attributes.values():
                _index(index, accessors, "vertex attribute")
                if accessors[index]["count"] != count:
                    raise ValueError("Vertex attributes have differing vertex counts")
            _vertex_attributes(attributes, accessors, arrays)
            _shape(accessors, joint, "VEC4", (5121, 5123), "JOINTS_0", count)
            _shape(accessors, weight, "VEC4", (5121, 5123, 5126), "WEIGHTS_0", count)
            if accessors[joint].get("normalized", False) or np.any(
                arrays[joint] >= len(joints)
            ):
                raise ValueError("Skin vertex contains invalid joint index")
            weights = arrays[weight].astype(np.float64)
            component = accessors[weight]["componentType"]
            if component != 5126:
                if not accessors[weight].get("normalized", False):
                    raise ValueError("Integer skin weights must be normalized")
                weights /= 255 if component == 5121 else 65535
            if (
                np.any(weights < 0)
                or np.any(weights > 1)
                or not np.allclose(weights.sum(axis=1), 1, atol=1e-3)
            ):
                raise ValueError("Skin weights must be nonnegative and sum to one")
            if "indices" in primitive:
                index = primitive["indices"]
                _shape(
                    accessors, index, "SCALAR", (5121, 5123, 5125), "triangle indices"
                )
                if (
                    accessors[index].get("normalized", False)
                    or accessors[index]["count"] % 3
                    or np.any(arrays[index] >= count)
                ):
                    raise ValueError("Triangle indices are out of bounds or incomplete")
            elif count % 3:
                raise ValueError(
                    "Unindexed triangle vertex count must be divisible by three"
                )
            if "material" in primitive:
                _index(
                    primitive["material"], range(material_count), "primitive material"
                )
                for key, info in _texture_infos(
                    doc["materials"][primitive["material"]]
                ):
                    texcoord = _integer(info.get("texCoord", 0), "material texCoord")
                    if f"TEXCOORD_{texcoord}" not in attributes:
                        raise ValueError(
                            f"{key} references missing vertex TEXCOORD_{texcoord}"
                        )


def _image_dimensions(raw, mime):
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Embedded image exceeds encoded size limit")
    if mime == "image/png":
        if (
            len(raw) < 45
            or raw[:8] != b"\x89PNG\r\n\x1a\n"
            or raw[8:16] != b"\0\0\0\rIHDR"
            or struct.unpack_from(">I", raw, 29)[0] != zlib.crc32(raw[12:29])
        ):
            raise ValueError("Invalid embedded PNG signature or IHDR")
        width, height, depth, color, compression, filtering, interlace = (
            struct.unpack_from(">IIBBBBB", raw, 16)
        )
        allowed_depths = {
            0: (1, 2, 4, 8, 16),
            2: (8, 16),
            3: (1, 2, 4, 8),
            4: (8, 16),
            6: (8, 16),
        }
        if (
            depth not in allowed_depths.get(color, ())
            or compression != 0
            or filtering != 0
            or interlace not in (0, 1)
        ):
            raise ValueError("Unsupported or malformed PNG header")
    elif mime == "image/jpeg":
        if len(raw) < 11 or raw[:2] != b"\xff\xd8" or raw[-2:] != b"\xff\xd9":
            raise ValueError("Invalid embedded JPEG signature")
        offset, dimensions = 2, None
        while offset < len(raw) - 2:
            if raw[offset] != 0xFF:
                raise ValueError("Malformed JPEG segment")
            while offset < len(raw) and raw[offset] == 0xFF:
                offset += 1
            if offset >= len(raw):
                raise ValueError("Truncated JPEG marker")
            marker = raw[offset]
            offset += 1
            if marker in (0xD9, 0xDA):
                break
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(raw):
                raise ValueError("Truncated JPEG segment length")
            length = struct.unpack_from(">H", raw, offset)[0]
            if length < 2 or offset + length > len(raw):
                raise ValueError("JPEG segment exceeds image bytes")
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                if length < 8 or dimensions is not None:
                    raise ValueError("Invalid JPEG frame header")
                height, width = struct.unpack_from(">HH", raw, offset + 3)
                dimensions = (width, height)
            offset += length
        if dimensions is None:
            raise ValueError("JPEG frame dimensions are missing")
        width, height = dimensions
    else:
        raise ValueError("Only embedded PNG and JPEG textures are supported")
    if (
        not 0 < width <= 16384
        or not 0 < height <= 16384
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise ValueError("Embedded texture exceeds decoded dimension or pixel limit")
    return width * height


def _scalar(value, label, lower=0, upper=1):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < lower
        or upper is not None
        and value > upper
    ):
        raise ValueError(f"Invalid {label} numeric range")


def _appearance(doc, views, binary):
    images = _objects(doc, "images")
    total_pixels = 0
    for image in images:
        if "uri" in image:
            if "bufferView" in image or type(image["uri"]) is not str:
                raise ValueError("Invalid image source")
            uri = image["uri"]
            if not uri.startswith(
                ("data:image/png;base64,", "data:image/jpeg;base64,")
            ):
                raise ValueError(
                    "Textures must be embedded PNG or JPEG; external URIs are forbidden"
                )
            try:
                raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("Invalid embedded image data") from exc
            mime = "image/png" if uri.startswith("data:image/png;") else "image/jpeg"
        else:
            view = views[_index(image.get("bufferView"), views, "image bufferView")]
            if image.get("mimeType") not in ("image/png", "image/jpeg"):
                raise ValueError("Only embedded PNG and JPEG textures are supported")
            start = view.get("byteOffset", 0)
            if view["byteLength"] > MAX_IMAGE_BYTES:
                raise ValueError("Embedded image exceeds encoded size limit")
            raw = binary[start : start + view["byteLength"]]
            mime = image["mimeType"]
        total_pixels += _image_dimensions(raw, mime)
        if total_pixels > MAX_TOTAL_IMAGE_PIXELS:
            raise ValueError("Combined textures exceed decoded pixel budget")
    samplers = _objects(doc, "samplers")
    textures = _objects(doc, "textures")
    for texture in textures:
        _index(texture.get("source"), images, "texture image")
        if "sampler" in texture:
            _index(texture["sampler"], samplers, "texture sampler")
    for material in _objects(doc, "materials"):
        pbr = material.get("pbrMetallicRoughness", {})
        if type(pbr) is not dict:
            raise ValueError("Invalid PBR material")
        for owner, name, width in (
            (pbr, "baseColorFactor", 4),
            (material, "emissiveFactor", 3),
        ):
            if name in owner:
                factors = _vector(owner[name], width, name)
                if np.any(factors < 0) or np.any(factors > 1):
                    raise ValueError(f"{name} components must be in [0,1]")
        for name in ("metallicFactor", "roughnessFactor"):
            if name in pbr:
                _scalar(pbr[name], name)
        if material.get("alphaMode", "OPAQUE") not in ("OPAQUE", "MASK", "BLEND"):
            raise ValueError("Invalid material alphaMode")
        if "alphaCutoff" in material:
            _scalar(material["alphaCutoff"], "alphaCutoff", upper=None)
        if "doubleSided" in material and type(material["doubleSided"]) is not bool:
            raise ValueError("Material doubleSided must be boolean")
        for key, info in _texture_infos(material):
            _index(info.get("index"), textures, "material texture")
            _integer(info.get("texCoord", 0), "texture texCoord")
            if key == "occlusionTexture" and "strength" in info:
                _scalar(info["strength"], "occlusion strength")
            if key == "normalTexture" and "scale" in info:
                if type(info["scale"]) not in (int, float) or not math.isfinite(
                    info["scale"]
                ):
                    raise ValueError(
                        "Normal texture scale must be finite numeric value"
                    )


def _animations(doc, accessors, arrays, nodes):
    for animation in _objects(doc, "animations", 128):
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if (
            type(samplers) is not list
            or type(channels) is not list
            or not samplers
            or not channels
            or len(samplers) > 4096
            or len(channels) > 4096
        ):
            raise ValueError("Invalid animation structure")
        seen = set()
        for channel in channels:
            if type(channel) is not dict or type(channel.get("target")) is not dict:
                raise ValueError("Invalid animation channel")
            target = channel["target"]
            node = _index(target.get("node"), nodes, "animation target")
            path = target.get("path")
            if path not in ("translation", "rotation"):
                raise ValueError("Scale animation and morph animation are unsupported")
            if "matrix" in nodes[node]:
                raise ValueError("Animated nodes must use TRS transforms")
            if (node, path) in seen:
                raise ValueError("Duplicate animation target")
            seen.add((node, path))
            sampler = samplers[
                _index(channel.get("sampler"), samplers, "animation sampler")
            ]
            if type(sampler) is not dict:
                raise ValueError("Invalid animation sampler")
            source, output = sampler.get("input"), sampler.get("output")
            _shape(accessors, source, "SCALAR", (5126,), "animation timestamps")
            times = arrays[source][:, 0]
            if np.any(times < 0) or np.any(np.diff(times) <= 0):
                raise ValueError(
                    "Animation timestamps must be nonnegative and strictly increasing"
                )
            interpolation = sampler.get("interpolation", "LINEAR")
            if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE"):
                raise ValueError("Unsupported animation interpolation")
            count = len(times) * (3 if interpolation == "CUBICSPLINE" else 1)
            _shape(
                accessors,
                output,
                "VEC4" if path == "rotation" else "VEC3",
                (5126,),
                "animation output",
                count,
            )
            if path == "rotation":
                rotations = (
                    arrays[output][1::3]
                    if interpolation == "CUBICSPLINE"
                    else arrays[output]
                )
                if not np.allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-3):
                    raise ValueError(
                        "Animation rotations must be unit xyzw quaternions"
                    )


def validate_glb(glb: bytes) -> dict:
    """Return the original parsed glTF metadata after supported-rig validation."""
    try:
        doc, binary = _parse(glb)
        accessors, arrays, views = _accessors(doc, binary)
        nodes, parents, reachable = _hierarchy(doc)
        _skins_meshes(doc, accessors, arrays, nodes, parents, reachable)
        _appearance(doc, views, binary)
        _animations(doc, accessors, arrays, nodes)
    except (
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        struct.error,
        RecursionError,
    ) as exc:
        raise ValueError("Malformed or unsupported rigged GLB") from exc
    return doc
