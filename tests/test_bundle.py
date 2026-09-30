import hashlib
import io
import json
import zipfile

import pytest

from unimate_pack.bundle import inspect_bundle, UPSTREAM_REVISION, SOLVER


def archive(extra=None, manifest_update=None):
    files = {
        "config.json": b"{}",
        "stats.npz": b"stats",
        "denoiser.safetensors": b"weights",
        "text_encoder/config.json": b"{}",
        "text_encoder/model.safetensors": b"text",
        "text_encoder/spiece.model": b"tokens",
        "text_encoder/tokenizer_config.json": b"{}",
        "text_encoder/special_tokens_map.json": b"{}",
        "text_encoder/NOTICE": b"Google FLAN-T5-base",
        "text_encoder/LICENSE": b"Apache-2.0",
    }
    files.update(extra or {})
    manifest = dict(
        schema="unimate.bundle.v1",
        upstream_revision=UPSTREAM_REVISION,
        weights="ema",
        model_revision="a" * 40,
        text_encoder=dict(id="google/flan-t5-base", revision="b" * 40),
        solver=SOLVER,
        files={
            name: dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
            for name, data in files.items()
        },
    )
    manifest.update(manifest_update or {})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            z.writestr(name, data)
    return output.getvalue()


def test_manifest_integrity():
    assert inspect_bundle(archive())["weights"] == "ema"


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/escape",
        "text_encoder/../escape",
        "text_encoder\\escape",
        "code.py",
    ],
)
def test_reject_unsafe_members(name):
    with pytest.raises(ValueError):
        inspect_bundle(archive({name: b"x"}))


@pytest.mark.parametrize(
    "update",
    [
        {"weights": "raw"},
        {"upstream_revision": "a" * 40},
        {"solver": {"method": "euler"}},
    ],
)
def test_reject_incompatible_bundle(update):
    with pytest.raises(ValueError):
        inspect_bundle(archive(manifest_update=update))


def test_corrupt_archive_rejected():
    with pytest.raises(ValueError):
        inspect_bundle(b"invalid")


def test_checksum_mismatch_rejected():
    raw = archive()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(output, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, b"bad" if name == "stats.npz" else src.read(name))
    with pytest.raises(ValueError, match="integrity"):
        inspect_bundle(output.getvalue())


def test_duplicate_members_rejected():
    output = io.BytesIO(archive())
    with zipfile.ZipFile(output, "a") as target:
        with pytest.warns(UserWarning):
            target.writestr("stats.npz", b"bad")
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_bundle(output.getvalue())


def test_symlink_members_rejected():
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(archive())) as source,
        zipfile.ZipFile(output, "w") as target,
    ):
        for item in source.infolist():
            if item.filename == "stats.npz":
                item.external_attr = 0o120777 << 16
            target.writestr(item, source.read(item.filename))
    with pytest.raises(ValueError, match="Links"):
        inspect_bundle(output.getvalue())


def test_missing_selected_model(tmp_path):
    from unimate_pack.inference import load_model_bundle

    with pytest.raises(ValueError, match="installed"):
        load_model_bundle(tmp_path / "missing model.unimate")
