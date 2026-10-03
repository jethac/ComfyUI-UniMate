"""Portable bundles with private, ComfyUI-managed runtime models."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import tempfile
import threading

from .bundle import extract_bundle, inspect_bundle, MAX_BUNDLE_BYTES, _json

_LOCK = threading.RLock()
_RUNTIMES = OrderedDict()


def _release_rope_cache(patcher, unpatch_all):
    """Upstream RopeND is a plain object: its lazy tensors are not buffers."""
    for name in ("rope_t", "rope_j"):
        rope = getattr(patcher.model, name, None)
        for axis in range(getattr(rope, "nd", 0)):
            for prefix in ("cos_", "sin_"):
                key = f"{prefix}{axis}"
                if hasattr(rope, key):
                    delattr(rope, key)


def load_model_bundle(path: str | Path) -> dict:
    from .contracts import make_model

    path = Path(path)
    if path.suffix.lower() != ".unimate" or not path.is_file():
        raise ValueError("Select an installed .unimate model bundle")
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Model bundle exceeds the 4 GiB limit")
    payload = path.read_bytes()
    inspect_bundle(payload)
    return make_model(payload, path.name)


class _Runtime:
    def __init__(self, model):
        import torch
        from safetensors.torch import load_file
        from transformers import T5EncoderModel, T5Tokenizer
        from comfy import model_management as mm
        from comfy.model_patcher import ModelPatcher
        from comfy.patcher_extension import CallbacksMP
        from .contracts import decode_arrays
        from .upstream import create_denoiser, validate_stats

        self.directory = tempfile.TemporaryDirectory(prefix="unimate-model-")
        try:
            root = Path(self.directory.name)
            self.manifest = extract_bundle(model["bundle"], root)
            self.config = _json((root / "config.json").read_bytes())
            self.stats = decode_arrays((root / "stats.npz").read_bytes())
            validate_stats(self.stats)
            mm.throw_exception_if_processing_interrupted()
            # Model construction initializes parameters; fork_rng prevents changes to
            # ComfyUI's global generator. Sampling below uses its own Generator.
            with torch.random.fork_rng(devices=[]):
                denoiser = create_denoiser(self.config).float().eval()
            state = load_file(str(root / "denoiser.safetensors"))
            if any(
                value.is_floating_point() and not torch.isfinite(value).all()
                for value in state.values()
            ):
                raise ValueError("Denoiser weights contain nonfinite values")
            denoiser.load_state_dict(state, strict=True)
            del state
            denoiser.requires_grad_(False)
            self.denoiser = ModelPatcher(
                denoiser,
                load_device=mm.get_torch_device(),
                offload_device=mm.unet_offload_device(),
            )
            self.denoiser.add_callback(CallbacksMP.ON_DETACH, _release_rope_cache)
            mm.throw_exception_if_processing_interrupted()
            text_config = _json((root / "text_encoder/config.json").read_bytes())
            expected = dict(
                model_type="t5",
                d_model=768,
                d_ff=2048,
                d_kv=64,
                num_heads=12,
                num_layers=12,
                vocab_size=32128,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=128,
                feed_forward_proj="gated-gelu",
                layer_norm_epsilon=1e-6,
            )
            if any(text_config.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    "Bundle must contain the FLAN-T5-base encoder architecture"
                )
            with torch.random.fork_rng(devices=[]):
                encoder, loading = T5EncoderModel.from_pretrained(
                    str(root / "text_encoder"),
                    local_files_only=True,
                    use_safetensors=True,
                    output_loading_info=True,
                )
                if any(
                    loading.get(key)
                    for key in (
                        "missing_keys",
                        "unexpected_keys",
                        "mismatched_keys",
                        "error_msgs",
                    )
                ):
                    raise ValueError(
                        "Bundle text encoder weights are incomplete or incompatible"
                    )
                encoder = encoder.float().eval().requires_grad_(False)
            if any(
                not torch.isfinite(parameter).all()
                for parameter in encoder.parameters()
            ):
                raise ValueError("Text encoder weights contain nonfinite values")
            self.tokenizer = T5Tokenizer.from_pretrained(
                str(root / "text_encoder"), local_files_only=True
            )

            # Transformers exposes a read-only .device property; ModelPatcher
            # writes .device during load/offload. A registered wrapper adapts
            # that interface while retaining normal module/parameter traversal.
            class ManagedEncoder(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.encoder = module

                def forward(self, **inputs):
                    return self.encoder(**inputs)

            self.encoder = ModelPatcher(
                ManagedEncoder(encoder),
                load_device=mm.text_encoder_device(),
                offload_device=mm.text_encoder_offload_device(),
            )
            self.embeddings = OrderedDict()
        except BaseException:
            self.directory.cleanup()
            raise

    def encode(self, texts):
        import numpy as np
        import torch
        from comfy import model_management as mm

        key = tuple(texts)
        if key in self.embeddings:
            self.embeddings.move_to_end(key)
            return self.embeddings[key]
        mm.throw_exception_if_processing_interrupted()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("Text exceeds the local encoder's 512-token limit")
        for i, text in enumerate(texts):
            if text == "":
                inputs["attention_mask"][i] = 0
        mm.load_models_gpu([self.encoder], force_full_load=True)
        inputs = inputs.to(self.encoder.load_device)
        with torch.inference_mode():
            hidden = self.encoder.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            # Joint embeddings use upstream pooled_from_hidden arithmetic. Prompt
            # embeddings use trimmed rows' mean, matching sample._encode_prompt.
            pooled = (
                (
                    (hidden * mask.unsqueeze(-1)).sum(-2)
                    / mask.sum(-1, keepdim=True).clamp(min=1)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            hidden = hidden.cpu().numpy().astype(np.float32)
            counts = mask.sum(-1).cpu().tolist()
            sequences = [
                hidden[i, :n] if n else np.zeros((1, 768), dtype=np.float32)
                for i, n in enumerate(counts)
            ]
        mm.throw_exception_if_processing_interrupted()
        self.embeddings[key] = (sequences, pooled)
        while len(self.embeddings) > 16:
            self.embeddings.popitem(last=False)
        return sequences, pooled


def _get_runtime(model):
    key = model["sha256"]
    if key not in _RUNTIMES:
        _RUNTIMES[key] = _Runtime(model)
        while len(_RUNTIMES) > 1:
            _, previous = _RUNTIMES.popitem(last=False)
            previous.directory.cleanup()
    _RUNTIMES.move_to_end(key)
    return _RUNTIMES[key]


def generate_motion(
    model: dict,
    rig: dict,
    prompt: str,
    seed: int,
    guidance: float,
    normalization: str = "objaverse",
    *, reference: dict | None = None, constraint_mode: str | None = None,
    selection: str = "",
) -> dict:
    import math
    import numpy as np
    import torch
    from comfy import model_management as mm
    from comfy.utils import ProgressBar
    from .contracts import (
        validate_model,
        validate_rig,
        decode_arrays,
        encode_arrays,
        make_motion,
        validate_motion,
    )
    from .upstream import build_condition, sample_flow, sample_constrained_flow

    validate_model(model)
    validate_rig(rig)
    if not isinstance(prompt, str) or len(prompt) > 4096:
        raise ValueError("Prompt must contain at most 4096 characters")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if (
        not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
        or not 1 <= guidance <= 10
    ):
        raise ValueError("Guidance must be between 1 and 10")
    if normalization not in ("objaverse", "mixamo", "truebones"):
        raise ValueError("Choose objaverse, mixamo, or truebones normalization")
    arrays = decode_arrays(rig["conditioning"])
    reference_features = None
    keep = None
    if reference is not None or constraint_mode is not None:
        from .motion_selection import frame_mask, joint_mask
        if reference is None or constraint_mode not in ("inbetween", "edit"):
            raise ValueError("Select a reference motion and inbetween or edit mode")
        validate_motion(reference, rig["rig_id"])
        reference_features = decode_arrays(reference["features"])["features"]
        if reference_features.shape[1] != len(arrays["parents"]) or not 1 <= len(reference_features) <= 60:
            raise ValueError("Reference motion must match the rig and fit the 60-frame window")
        if guidance <= 1:
            raise ValueError("Constrained generation requires guidance greater than 1")
        if constraint_mode == "inbetween":
            keep = frame_mask(selection, len(reference_features), 60)
        else:
            aliases = [f"{raw}|{clean}" for raw, clean in zip(
                arrays["joint_names"], arrays.get("clean_joint_names", arrays["joint_names"])
            )]
            keep = joint_mask(selection, aliases, 71)
    with _LOCK:
        mm.throw_exception_if_processing_interrupted()
        runtime = _get_runtime(model)
        names = arrays.get("clean_joint_names", arrays["joint_names"]).tolist()
        prompt_tokens = runtime.encode([prompt])[0][0]
        joint_embeddings = runtime.encode(names)[1]
        cond = build_condition(
            arrays,
            runtime.config,
            runtime.stats,
            normalization,
            prompt_tokens,
            joint_embeddings,
        )
        if reference_features is not None:
            frames = len(reference_features)
            cond["motion_length"] = torch.full_like(cond["motion_length"], frames)
            cond["lengths_mask"] = (
                torch.arange(60, device=cond["lengths_mask"].device)[None, :] < frames
            ).reshape(cond["lengths_mask"].shape)
        mm.load_models_gpu([runtime.denoiser], force_full_load=True)
        device = runtime.denoiser.load_device
        cond = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in cond.items()
        }
        progress = ProgressBar(100)
        with torch.inference_mode():
            if reference_features is None:
                samples = sample_flow(
                    runtime.denoiser.model, cond, seed, guidance, device,
                    mm.throw_exception_if_processing_interrupted,
                    lambda value: progress.update_absolute(value, 100),
                )
            else:
                known = torch.zeros((1, 71, 12, 60), device=device)
                raw = torch.from_numpy(reference_features).to(device)
                count = raw.shape[1]
                normalized = (raw - cond["mean"][0, :count]) / cond["std"][0, :count]
                known[0, :count, :, :len(raw)] = normalized.permute(1, 2, 0)
                samples = sample_constrained_flow(
                    runtime.denoiser.model, cond, known,
                    torch.from_numpy(keep).to(device), seed, guidance,
                    mm.throw_exception_if_processing_interrupted,
                    lambda value: progress.update_absolute(value, 100),
                )
            joints = len(arrays["parents"])
            features = samples[0, :joints].permute(2, 0, 1)
            features = features * cond["std"][0, :joints] + cond["mean"][0, :joints]
            features = features.cpu().numpy().astype(np.float32)
            if reference_features is not None:
                features = features[:len(reference_features)]
                # Preserve the input features exactly across normalization roundoff.
                if constraint_mode == "inbetween":
                    selected = keep[0, 0, 0, :len(features)]
                    features[selected] = reference_features[selected]
                else:
                    selected = keep[0, :joints, 0, 0]
                    features[:, selected] = reference_features[:, selected]
        mm.throw_exception_if_processing_interrupted()
        if not np.isfinite(features).all():
            raise ValueError("UniMate produced nonfinite motion features")
        progress.update_absolute(100, 100)
        metadata = dict(
            prompt=prompt,
            seed=seed,
            guidance=float(guidance),
            normalization=normalization,
            solver=runtime.manifest["solver"] if reference_features is None else {"method": "euler", "num_steps": 50},
            model_sha256=model["sha256"],
            model_revision=runtime.manifest["model_revision"],
            weights="ema",
            text_encoder=runtime.manifest["text_encoder"],
            upstream_revision=runtime.manifest["upstream_revision"],
            adapter_revision="unimate.inference.v1",
            precision="float32",
            frames=len(features),
            fps=30,
        )
        if reference is not None:
            if "canonical_root_origin" in reference["metadata"]:
                metadata["canonical_root_origin"] = reference["metadata"]["canonical_root_origin"]
            metadata.update(constraint_mode=constraint_mode, selection=selection,
                            reference_features_sha256=hashlib.sha256(reference["features"]).hexdigest())
        return make_motion(rig["rig_id"], encode_arrays(features=features), metadata)
