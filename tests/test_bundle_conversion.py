import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from tools import build_bundle


def test_trusted_stats_uses_authenticated_snapshot_if_source_changes(
    tmp_path, monkeypatch
):
    legacy = {
        family: {
            part: np.arange(12, dtype=float) + 1
            for part in ("mean_root", "std_root", "mean_local", "std_local")
        }
        for family in ("objaverse", "mixamo", "truebones")
    }
    data = io.BytesIO()
    np.save(data, legacy)
    payload = data.getvalue()
    source = tmp_path / "trusted stats.npy"
    source.write_bytes(payload)
    monkeypatch.setattr(
        build_bundle, "OFFICIAL_STATS_SHA256", hashlib.sha256(payload).hexdigest()
    )
    original_read = Path.read_bytes

    def replace_after_read(path):
        snapshot = original_read(path)
        path.write_bytes(b"untrusted replacement must never be loaded")
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    stats = build_bundle.load_stats(source, trust_legacy_stats=True)
    np.testing.assert_array_equal(stats["objaverse_mean_root"], np.arange(12) + 1)
    assert original_read(source).startswith(b"untrusted")


@pytest.mark.parametrize("trust", [False, True])
def test_unknown_legacy_stats_never_reaches_pickle(tmp_path, monkeypatch, trust):
    source = tmp_path / "not official.npy"
    source.write_bytes(b"untrusted")

    def forbidden(*args, **kwargs):
        pytest.fail("Unverified stats reached numpy deserialization")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="exact pinned official"):
        build_bundle.load_stats(source, trust)


def test_mixamo_only_numeric_stats_round_trip(tmp_path):
    source = tmp_path / "mixamo.npz"
    values = {f"mixamo_{part}": np.ones(12) for part in
              ("mean_root", "std_root", "mean_local", "std_local")}
    np.savez(source, **values)
    assert set(build_bundle.load_stats(source)) == set(values)
