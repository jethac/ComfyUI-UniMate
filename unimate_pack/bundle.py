"""Bounded, hash-checked inference bundle; this module never deserializes pickle."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import stat
import zipfile

UPSTREAM_REVISION = "5d6aabedd947297b5ba6706d8e9113e68c0c3e4f"
SOLVER = {"method": "dopri5", "num_steps": 50, "atol": 1e-6, "rtol": 1e-3}
MAX_BUNDLE_BYTES = 4 * 1024**3
REQUIRED = frozenset(
    {
        "config.json",
        "stats.npz",
        "denoiser.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/spiece.model",
        "text_encoder/tokenizer_config.json",
        "text_encoder/special_tokens_map.json",
        "text_encoder/NOTICE",
        "text_encoder/LICENSE",
    }
)
ALLOWED = REQUIRED | {"text_encoder/tokenizer.json"}


def _json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key in model bundle")
            out[key] = value
        return out

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON")
        ),
    )


def inspect_bundle(payload: bytes) -> dict:
    """Validate every member before extraction, including unknown and duplicate files."""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BUNDLE_BYTES:
        raise ValueError("Model bundle exceeds the 4 GiB limit or is empty")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [entry.filename for entry in members]
            if len(names) != len(set(names)) or not 1 <= len(names) <= 16:
                raise ValueError("Duplicate or excess bundle members")
            if set(names) - (ALLOWED | {"manifest.json"}) or not REQUIRED <= set(names):
                raise ValueError("Unsafe, missing, or unsupported model bundle member")
            total = 0
            for entry in members:
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or entry.flag_bits & 1 or entry.is_dir():
                    raise ValueError(
                        "Links, encryption, and directories are forbidden in bundles"
                    )
                total += entry.file_size
                limit = (
                    1024**2
                    if entry.filename == "manifest.json"
                    else 8 * 1024**2
                    if entry.filename.endswith(".json")
                    else MAX_BUNDLE_BYTES
                )
                if entry.file_size > limit or total > MAX_BUNDLE_BYTES:
                    raise ValueError("Expanded model bundle exceeds size limit")
            manifest = _json(archive.read("manifest.json"))
            if (
                manifest.get("schema") != "unimate.bundle.v1"
                or manifest.get("weights") != "ema"
                or manifest.get("upstream_revision") != UPSTREAM_REVISION
                or manifest.get("solver") != SOLVER
            ):
                raise ValueError(
                    "Incompatible UniMate bundle revision, EMA selection, or solver"
                )
            encoder = manifest.get("text_encoder", {})
            if (
                encoder.get("id") != "google/flan-t5-base"
                or not re.fullmatch(r"[0-9a-f]{40}", encoder.get("revision", ""))
                or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("model_revision", ""))
            ):
                raise ValueError(
                    "Bundle must identify pinned model and FLAN-T5-base revisions"
                )
            entries = manifest.get("files")
            if not isinstance(entries, dict) or set(entries) != set(names) - {
                "manifest.json"
            }:
                raise ValueError("Manifest file inventory mismatch")
            for name, expected in entries.items():
                digest = hashlib.sha256()
                size = 0
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(1024**2), b""):
                        size += len(chunk)
                        digest.update(chunk)
                if expected != {"sha256": digest.hexdigest(), "size": size}:
                    raise ValueError(f"Model bundle integrity check failed: {name}")
            return manifest
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        AttributeError,
        UnicodeError,
        json.JSONDecodeError,
        RuntimeError,
    ) as error:
        raise ValueError("Corrupt UniMate model bundle") from error


def extract_bundle(payload: bytes, destination: Path) -> dict:
    manifest = inspect_bundle(payload)
    # This is a private new TemporaryDirectory, never an input-controlled directory.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in manifest["files"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024**2), b""):
                    output.write(chunk)
    return manifest
