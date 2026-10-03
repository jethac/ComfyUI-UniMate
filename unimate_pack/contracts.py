"""Versioned, portable socket values. This module does not import ComfyUI or torch."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import stat
import zipfile

import numpy as np

MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def _bytes(value, limit, label):
    if type(value) is not bytes or not value or len(value) > limit:
        raise ValueError(f"{label} must be nonempty bytes within {limit} bytes")


def _digest(value, label):
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"Invalid {label} SHA-256 digest")


def _name(value, extension):
    if (
        type(value) is not str
        or not value
        or len(value) > 255
        or any(c in value for c in "\\/:\0")
        or value in (".", "..")
        or not value.lower().endswith(extension)
    ):
        raise ValueError(f"Name must be a portable {extension} basename")


def _record(value, schema, fields):
    if type(value) is not dict or set(value) != set(fields) | {"schema"}:
        raise ValueError(f"Invalid fields for {schema}")
    if value["schema"] != schema:
        raise ValueError(f"Unsupported schema; expected {schema}")


def _json(value, depth=0, text=False):
    if depth > 32:
        raise ValueError("JSON metadata exceeds nesting limit")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON metadata contains nonfinite values")
        return
    if type(value) is str:
        if (
            "\0" in value
            or not text
            and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/", "\\\\")))
        ):
            raise ValueError("Local paths are not portable metadata")
        return
    if type(value) is list:
        for item in value:
            _json(item, depth + 1, text)
        return
    if type(value) is dict:
        for key, item in value.items():
            if (
                type(key) is not str
                or key.lower() in ("path", "paths", "local_path")
                or key.lower().endswith("_path")
            ):
                raise ValueError(
                    "Metadata keys must be strings without local path fields"
                )
            _json(item, depth + 1, text or key == "prompt")
        return
    raise ValueError("Metadata must contain only portable JSON values")


def _json_bytes(value):
    _json(value)
    data = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON metadata is too large")
    return data


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON manifest key")
        result[key] = value
    return result


def _array(array):
    if (
        not isinstance(array, np.ndarray)
        or array.dtype.kind not in "biufU"
        or array.dtype.hasobject
        or array.ndim > 8
        or array.nbytes > MAX_ARRAY_BYTES
        or array.dtype.itemsize > 4096
    ):
        raise ValueError(
            "Arrays must have bounded numeric or Unicode dtype, never object dtype"
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("Array contains nonfinite values")


def _members(archive, total_limit, member_limit, count_limit=256):
    infos = archive.infolist()
    if not infos or len(infos) > count_limit:
        raise ValueError("Invalid archive member count")
    seen, total = set(), 0
    for info in infos:
        name = info.filename
        parts = name.split("/")
        mode = info.external_attr >> 16
        if (
            name in seen
            or not name
            or "\\" in name
            or ":" in name
            or "\0" in name
            or any(part in ("", ".", "..") for part in parts)
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError("Unsafe or duplicate archive member")
        if info.file_size > member_limit or info.file_size < 0:
            raise ValueError("Archive member is too large")
        total += info.file_size
        if total > total_limit:
            raise ValueError("Archive expands beyond size limit")
        seen.add(name)
    return infos


def encode_arrays(**arrays) -> bytes:
    if not arrays or len(arrays) > 128:
        raise ValueError("Numeric archive requires 1–128 arrays")
    for name, array in arrays.items():
        if not _ARRAY_NAME.fullmatch(name):
            raise ValueError("Invalid array name")
        _array(array)
    if sum(a.nbytes for a in arrays.values()) > MAX_ARRAY_BYTES:
        raise ValueError("Numeric archive is too large")
    output = io.BytesIO()
    np.savez(output, **arrays)
    result = output.getvalue()
    decode_arrays(result)
    return result


def decode_arrays(payload: bytes) -> dict[str, np.ndarray]:
    _bytes(payload, MAX_ARRAY_BYTES, "NPZ payload")
    result = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = _members(archive, MAX_ARRAY_BYTES, MAX_ARRAY_BYTES, 128)
            for info in infos:
                if not info.filename.endswith(".npy") or not _ARRAY_NAME.fullmatch(
                    info.filename[:-4]
                ):
                    raise ValueError("Numeric archives contain only named .npy arrays")
                raw = archive.read(info)
                stream = io.BytesIO(raw)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY version")
                if (
                    dtype.hasobject
                    or dtype.kind not in "biufU"
                    or dtype.itemsize > 4096
                    or len(shape) > 8
                    or any(type(n) is not int or n < 0 for n in shape)
                ):
                    raise ValueError("Unsafe NPY dtype or shape")
                expected = math.prod(shape) * dtype.itemsize
                if expected > MAX_ARRAY_BYTES or len(raw) - stream.tell() != expected:
                    raise ValueError("NPY shape exceeds its bounded member data")
                array = (
                    np.frombuffer(raw, dtype=dtype, offset=stream.tell())
                    .reshape(shape, order="F" if fortran else "C")
                    .copy()
                )
                _array(array)
                result[info.filename[:-4]] = array
    except (
        OSError,
        EOFError,
        zipfile.BadZipFile,
        RuntimeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError("Invalid numeric archive") from exc
    return result


def make_asset(glb: bytes, name: str) -> dict:
    from .assets import MAX_GLB_BYTES

    _bytes(glb, MAX_GLB_BYTES, "GLB")
    value = {
        "schema": "unimate.asset.v1",
        "glb": glb,
        "sha256": hashlib.sha256(glb).hexdigest(),
        "name": name,
    }
    validate_asset(value)
    return value


def validate_asset(value: dict) -> None:
    _asset_metadata(value)


def _asset_metadata(value):
    from .assets import MAX_GLB_BYTES, validate_glb

    _record(value, "unimate.asset.v1", ("glb", "sha256", "name"))
    _bytes(value["glb"], MAX_GLB_BYTES, "GLB")
    _name(value["name"], ".glb")
    _digest(value["sha256"], "asset")
    if hashlib.sha256(value["glb"]).hexdigest() != value["sha256"]:
        raise ValueError("Asset digest mismatch")
    return validate_glb(value["glb"])


def _rig_digest(asset, conditioning, mapping):
    digest = hashlib.sha256(b"unimate.rig.v1\0")
    for part in (asset["sha256"].encode(), conditioning, _json_bytes(mapping)):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def make_rig(asset: dict, conditioning: bytes, mapping: dict) -> dict:
    validate_asset(asset)
    _bytes(conditioning, MAX_ARRAY_BYTES, "conditioning")
    value = {
        "schema": "unimate.rig.v1",
        "asset": copy.deepcopy(asset),
        "conditioning": conditioning,
        "mapping": copy.deepcopy(mapping),
        "rig_id": _rig_digest(asset, conditioning, mapping),
    }
    validate_rig(value)
    return value


def _rig_mapping(mapping, asset_metadata):
    required = {
        "joint_indices",
        "source_to_canonical",
        "adapter_revision",
        "upstream_revision",
        "facing",
        "quaternion_order",
    }
    missing = required - mapping.keys()
    if missing:
        raise ValueError(
            f"Rig mapping is missing required fields: {', '.join(sorted(missing))}"
        )
    if mapping["adapter_revision"] != "gltf-preserving-v1":
        raise ValueError(
            "Unsupported rig adapter revision; prepare the source asset again"
        )
    if mapping["upstream_revision"] != "5d6aabedd947297b5ba6706d8e9113e68c0c3e4f":
        raise ValueError("Rig mapping must identify the pinned UniMate revision")
    if mapping["quaternion_order"] != "wxyz conditioning; xyzw glTF":
        raise ValueError("Unsupported rig quaternion conventions")
    if mapping["facing"] not in ("+Z", "-Z", "+X", "-X", "joint_pair"):
        raise ValueError("Rig mapping has unsupported facing direction")
    if "body_axis" in mapping and type(mapping["body_axis"]) is not bool:
        raise ValueError("Rig body_axis must be boolean")
    if mapping["facing"] == "joint_pair":
        left, right = mapping.get("left_joint"), mapping.get("right_joint")
        names = {
            asset_metadata["nodes"][joint].get("name", f"joint_{joint}")
            for joint in asset_metadata["skins"][0]["joints"]
        }
        if (
            type(left) is not str
            or type(right) is not str
            or left == right
            or left not in names
            or right not in names
        ):
            raise ValueError(
                "Rig joint-pair facing requires two distinct source joint names"
            )
    values = mapping["source_to_canonical"]
    limit = float(np.finfo(np.float32).max)
    if (
        type(values) is not list
        or len(values) != 4
        or any(type(row) is not list or len(row) != 4 for row in values)
        or any(
            type(item) not in (int, float)
            or abs(item) > limit
            or not math.isfinite(item)
            for row in values
            for item in row
        )
    ):
        raise ValueError("Rig source_to_canonical requires finite numeric 4x4 matrix")
    transform = np.asarray(values, dtype=np.float64)
    linear = transform[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or np.any(scales <= 0)
        or not np.allclose(scales / scales[0], 1, atol=1e-7, rtol=1e-5)
    ):
        raise ValueError(
            "Rig source_to_canonical must be invertible affine uniform similarity"
        )
    rotation = linear / scales
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("Rig source_to_canonical contains shear or reflection")
    try:
        reverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Rig source_to_canonical is singular") from exc
    if not np.isfinite(reverse).all() or np.any(np.abs(reverse) > limit):
        raise ValueError(
            "Rig source_to_canonical cannot be reversed within glTF numeric limits"
        )


def validate_rig(value: dict) -> None:
    _record(value, "unimate.rig.v1", ("asset", "conditioning", "mapping", "rig_id"))
    asset_metadata = _asset_metadata(value["asset"])
    if type(value["mapping"]) is not dict:
        raise ValueError("Rig mapping must be JSON object")
    _json_bytes(value["mapping"])
    _rig_mapping(value["mapping"], asset_metadata)
    arrays = decode_arrays(value["conditioning"])
    parents = arrays.get("parents")
    if (
        parents is None
        or parents.ndim != 1
        or parents.dtype.kind not in "iu"
        or not 5 <= len(parents) <= 70
        or parents[0] != -1
    ):
        raise ValueError("Rig requires 5–70 ordered joint parents with root -1")
    for child, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < child:
            raise ValueError("Rig parents must precede children without cycles")
    joint_count = len(parents)
    if joint_count != len(asset_metadata["skins"][0]["joints"]):
        raise ValueError("Prepared rig must retain every source skin joint")
    for name in ("joint_names", "clean_joint_names", "joint_depths"):
        if name in arrays and arrays[name].shape != (joint_count,):
            raise ValueError(f"Rig {name} shape does not match joint count")
    for name in ("joint_names", "clean_joint_names"):
        if name in arrays and arrays[name].dtype.kind != "U":
            raise ValueError(f"Rig {name} requires Unicode strings")
    for name in ("offsets", "tpos_offsets", "tpos_first_frame"):
        if name in arrays and (
            arrays[name].shape != (joint_count, 3) or arrays[name].dtype.kind != "f"
        ):
            raise ValueError(f"Rig {name} shape does not match joint count")
    for name in ("tpos_local_rotations", "tpos_global_rotations"):
        if name in arrays and (
            arrays[name].shape != (joint_count, 4)
            or arrays[name].dtype.kind != "f"
            or not np.allclose(np.linalg.norm(arrays[name], axis=1), 1, atol=1e-4)
        ):
            raise ValueError(f"Rig {name} requires unit quaternions shaped (J,4)")
    for name in ("joint_relations", "joint_graph_dists"):
        if name in arrays and (
            arrays[name].shape != (joint_count, joint_count)
            or arrays[name].dtype.kind not in "iuf"
            or np.any(arrays[name] < 0)
        ):
            raise ValueError(f"Rig {name} requires nonnegative (J,J) numeric matrix")
    if "joint_depths" in arrays:
        depths = arrays["joint_depths"]
        if (
            depths.dtype.kind not in "iu"
            or np.any(depths < 0)
            or np.any(depths >= joint_count)
        ):
            raise ValueError("Rig joint_depths contains invalid tree depths")
    if "edge_indexs" in arrays:
        edges = arrays["edge_indexs"]
        if (
            edges.shape != (2, 2 * (joint_count - 1))
            or edges.dtype.kind not in "iu"
            or np.any(edges < 0)
            or np.any(edges >= joint_count)
        ):
            raise ValueError("Rig edge_indexs contains invalid skeleton indices")
    if "kinematic_chains" in arrays:
        chains = arrays["kinematic_chains"]
        if (
            chains.ndim != 2
            or not chains.size
            or chains.shape[1] > joint_count
            or chains.dtype.kind not in "iu"
            or np.any(chains < -1)
            or np.any(chains >= joint_count)
        ):
            raise ValueError(
                "Rig kinematic_chains contains invalid padded skeleton indices"
            )
    if "face_joint_idxs" in arrays:
        face = arrays["face_joint_idxs"]
        if (
            face.shape not in ((2,), (4,))
            or face.dtype.kind not in "iu"
            or not ((face.shape == (2,) and np.all(face == -1)) or np.all((face >= 0) & (face < joint_count)))
        ):
            raise ValueError(
                "Rig face_joint_idxs requires two or four valid indices or -1 sentinel"
            )
    if "spectral_feats" in arrays and (
        arrays["spectral_feats"].shape != (joint_count, 8)
        or arrays["spectral_feats"].dtype.kind != "f"
    ):
        raise ValueError("Rig spectral_feats requires floating-point (J,8) array")
    if "scale_factor" in arrays and (
        arrays["scale_factor"].shape != ()
        or arrays["scale_factor"].dtype.kind not in "iuf"
        or arrays["scale_factor"].item() <= 0
    ):
        raise ValueError("Rig scale_factor requires positive scalar")
    indices = value["mapping"]["joint_indices"]
    if (
        type(indices) is not list
        or len(indices) != joint_count
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != joint_count
    ):
        raise ValueError("Rig mapping joint identities do not match conditioning")
    if set(indices) != set(asset_metadata["skins"][0]["joints"]):
        raise ValueError("Rig mapping must identify the exact original skin joints")
    source_parents = [-1] * len(asset_metadata["nodes"])
    for parent, node in enumerate(asset_metadata["nodes"]):
        for child in node.get("children", []):
            source_parents[child] = parent
    prepared_index = {source: index for index, source in enumerate(indices)}
    for prepared, source in enumerate(indices):
        ancestor = source_parents[source]
        while ancestor != -1 and ancestor not in prepared_index:
            ancestor = source_parents[ancestor]
        expected = prepared_index.get(ancestor, -1)
        if parents[prepared] != expected:
            raise ValueError("Prepared parents do not preserve source joint hierarchy")
    _digest(value["rig_id"], "rig")
    if value["rig_id"] != _rig_digest(
        value["asset"], value["conditioning"], value["mapping"]
    ):
        raise ValueError("Prepared rig digest mismatch")


def make_model(bundle: bytes, name: str) -> dict:
    _bytes(bundle, MAX_MODEL_BYTES, "Model bundle")
    value = {
        "schema": "unimate.model.v1",
        "bundle": bundle,
        "sha256": hashlib.sha256(bundle).hexdigest(),
        "name": name,
    }
    validate_model(value)
    return value


def validate_model(value: dict) -> None:
    _record(value, "unimate.model.v1", ("bundle", "sha256", "name"))
    _bytes(value["bundle"], MAX_MODEL_BYTES, "Model bundle")
    _name(value["name"], ".unimate")
    _digest(value["sha256"], "model")
    if hashlib.sha256(value["bundle"]).hexdigest() != value["sha256"]:
        raise ValueError("Model bundle digest mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(value["bundle"])) as archive:
            infos = _members(archive, MAX_MODEL_BYTES, MAX_MODEL_BYTES)
            if "manifest.json" not in archive.namelist():
                raise ValueError("Model bundle requires manifest.json")
            if archive.getinfo("manifest.json").file_size > MAX_JSON_BYTES:
                raise ValueError("Model manifest exceeds size limit")
            manifest = json.loads(
                archive.read("manifest.json"), object_pairs_hook=_unique_json_pairs
            )
            _json_bytes(manifest)
            if (
                type(manifest) is not dict
                or manifest.get("schema") != "unimate.bundle.v1"
                or type(manifest.get("files")) is not dict
            ):
                raise ValueError("Unsupported model bundle manifest")
            files = manifest["files"]
            if set(files) != {info.filename for info in infos} - {"manifest.json"}:
                raise ValueError("Model manifest must identify every bundled file")
            for name, entry in files.items():
                basename = name.rsplit("/", 1)[-1]
                if not (
                    name.lower().endswith(
                        (".json", ".npz", ".safetensors", ".model", ".txt", ".md")
                    )
                    or basename in ("LICENSE", "NOTICE")
                ):
                    raise ValueError(
                        "Model bundle contains unsafe executable or pickle artifacts"
                    )
                expected = entry.get("sha256") if type(entry) is dict else entry
                _digest(expected, "model member")
                info = archive.getinfo(name)
                if type(entry) is dict and (
                    type(entry.get("size")) is not int
                    or entry["size"] != info.file_size
                ):
                    raise ValueError("Model member size differs from manifest")
                if name.lower().endswith(".npz"):
                    if info.file_size > MAX_ARRAY_BYTES:
                        raise ValueError("Model numeric archive exceeds size limit")
                    decode_arrays(archive.read(info))
                hasher = hashlib.sha256()
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != expected:
                    raise ValueError("Model member digest mismatch")
    except (
        OSError,
        zipfile.BadZipFile,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError("Invalid model bundle") from exc


def make_motion(rig_id: str, features: bytes, metadata: dict) -> dict:
    value = {
        "schema": "unimate.motion.v1",
        "rig_id": rig_id,
        "features": features,
        "fps": 30,
        "metadata": copy.deepcopy(metadata),
    }
    validate_motion(value)
    return value


def validate_motion(value: dict, rig_id: str | None = None) -> None:
    _record(value, "unimate.motion.v1", ("rig_id", "features", "fps", "metadata"))
    _digest(value["rig_id"], "rig")
    if rig_id is not None and value["rig_id"] != rig_id:
        raise ValueError("Motion and prepared rig identities do not match")
    if type(value["fps"]) is not int or value["fps"] != 30:
        raise ValueError("UniMate motion requires fps 30")
    if type(value["metadata"]) is not dict:
        raise ValueError("Motion metadata must be JSON object")
    _json_bytes(value["metadata"])
    arrays = decode_arrays(value["features"])
    features = arrays.get("features")
    if (
        set(arrays) != {"features"}
        or features.dtype != np.dtype("float32")
        or features.ndim != 3
        or features.shape[0] < 1
        or features.shape[2] != 12
        or not 5 <= features.shape[1] <= 70
    ):
        raise ValueError("Motion requires float32 features shaped (T>=1, 5–70, 12)")
