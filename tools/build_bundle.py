"""Explicit local conversion of pinned UniMate raw/EMA and FLAN-T5 artifacts.

No downloads. Checkpoints use torch.load(weights_only=True). The optional legacy
stats conversion accepts ONLY the pinned official file with its known SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import shutil
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unimate_pack.bundle import UPSTREAM_REVISION, SOLVER, inspect_bundle

OFFICIAL_STATS_SHA256 = (
    "c13ecfe8317c5e7b4a04089b71817d0787494d8f3f1df66b0ac8f496842d2c89"
)
OTHER_RELEASED_STATS_SHA256 = frozenset({
    "9a3d946da35da85866d828dacbccc4d587503e103882f18364547a3f3765f675",
    "f449bd747ec65eeedc5e1723790dc988792c1875ec355af770b1c669b45f9b54",
})


def load_stats(stats_path: Path, trust_legacy_stats: bool = False) -> dict:
    import numpy as np
    from unimate_pack.upstream import validate_stats

    if stats_path.suffix.lower() == ".npy":
        verified_stats = stats_path.read_bytes()
        if (
            not trust_legacy_stats
            or hashlib.sha256(verified_stats).hexdigest() not in
            OTHER_RELEASED_STATS_SHA256 | {OFFICIAL_STATS_SHA256}
        ):
            raise ValueError(
                "Legacy stats require --trust-legacy-stats and the exact pinned official SHA-256"
            )
        # Trusted adapter boundary: digest checked BEFORE enabling legacy pickle.
        # Deserialize the authenticated snapshot, never reopen a replaceable path.
        legacy = np.load(io.BytesIO(verified_stats), allow_pickle=True).item()
        stats = {
            f"{family}_{part}": np.asarray(legacy[family][part])
            for family in legacy
            for part in ("mean_root", "std_root", "mean_local", "std_local")
        }
    else:
        with np.load(stats_path, allow_pickle=False) as source:
            stats = {key: source[key] for key in source.files}
    validate_stats(stats)
    return stats


def build_bundle(
    checkpoint: Path,
    config_path: Path,
    stats_path: Path,
    encoder_dir: Path,
    output: Path,
    model_revision: str,
    text_revision: str,
    trust_legacy_stats: bool = False,
    weights: str = 'ema',
) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import T5EncoderModel, T5Tokenizer
    from unimate_pack.upstream import create_denoiser, validate_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    stats = load_stats(stats_path, trust_legacy_stats)
    model = create_denoiser(config)
    from unimate_pack.training_initialization import select_checkpoint_weights
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    select_checkpoint_weights(model, checkpoint_data, weights)
    del checkpoint_data
    if output.suffix.lower() != ".unimate":
        raise ValueError("Output must use the .unimate extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing bundle: {output}")
    with tempfile.TemporaryDirectory(
        prefix="unimate-build-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        (root / "config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        np.savez(root / "stats.npz", **stats)
        # The spectral encoder is shared by all attention blocks. Materialize
        # aliases so strict state_dict reload preserves every expected key.
        save_file(
            {
                key: value.detach().cpu().contiguous().clone()
                for key, value in model.state_dict().items()
            },
            str(root / "denoiser.safetensors"),
        )
        del model
        encoder, loading = T5EncoderModel.from_pretrained(
            str(encoder_dir),
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or loading.get("error_msgs")
        ):
            raise ValueError(
                "Local FLAN-T5 encoder weights are incomplete or incompatible"
            )
        encoder = encoder.float().eval()
        tokenizer = T5Tokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
        encoder.save_pretrained(root / "text_encoder", safe_serialization=True)
        # Preserve the upstream tokenizer files byte-for-byte; transformers 5
        # save_pretrained rewrites SentencePiece tokenizers into a new layout.
        for name in (
            "spiece.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        ):
            source = encoder_dir / name
            if source.is_file():
                shutil.copyfile(source, root / "text_encoder" / name)
        (root / "text_encoder" / "NOTICE").write_text(
            "FLAN-T5-base, Google, Apache-2.0. https://huggingface.co/google/flan-t5-base\n"
            f"Source revision: {text_revision}\nEncoder-only safe conversion; tokenizer unchanged.\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            Path(__file__).parent / "licenses" / "Apache-2.0.txt",
            root / "text_encoder" / "LICENSE",
        )
        del encoder, tokenizer
        files = {}
        for path in root.rglob("*"):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024**2), b""):
                        digest.update(chunk)
                files[path.relative_to(root).as_posix()] = {
                    "sha256": digest.hexdigest(),
                    "size": path.stat().st_size,
                }
        manifest = dict(
            schema="unimate.bundle.v1",
            upstream_revision=UPSTREAM_REVISION,
            weights=weights,
            model_revision=model_revision,
            text_encoder={"id": "google/flan-t5-base", "revision": text_revision},
            solver=SOLVER,
            files=files,
        )
        candidate = root / "bundle.unimate"
        with zipfile.ZipFile(
            candidate, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            for name in sorted(files):
                archive.write(root / name, name)
        inspect_bundle(candidate.read_bytes())
        candidate.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("checkpoint", "config", "stats", "text-encoder", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--trust-legacy-stats", action="store_true")
    parser.add_argument("--weights", choices=['raw', 'ema'], default='ema')
    args = parser.parse_args()
    build_bundle(
        args.checkpoint,
        args.config,
        args.stats,
        args.text_encoder,
        args.output,
        args.model_revision,
        args.text_revision,
        args.trust_legacy_stats,
        args.weights,
    )
    print(f"Created {args.output.name}")


if __name__ == "__main__":
    main()
