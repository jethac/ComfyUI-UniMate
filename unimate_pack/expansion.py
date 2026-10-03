"""Prompt-chain expansion through free and replacement-style UniMate sampling."""

import hashlib
import math

import numpy as np

from .contracts import decode_arrays, encode_arrays, make_motion


def expand_motion(model, rig, prompts, seed, guidance, normalization="objaverse", overlap=10):
    from .inference import generate_motion

    if type(prompts) is not list or not 1 <= len(prompts) <= 64 or any(
        not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096 for prompt in prompts
    ):
        raise ValueError("Expansion requires 1–64 nonempty text prompts, at most 4096 characters each")
    if type(overlap) is not int or not 0 < overlap < 60:
        raise ValueError("Expansion overlap must be between 1 and 59 frames")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or not 1 < guidance <= 10:
        raise ValueError("Expansion requires guidance greater than 1 and at most 10")
    segments = []
    parts = []
    previous = None
    for index, prompt in enumerate(prompts):
        kwargs = {}
        if previous is not None:
            known = np.zeros_like(previous)
            known[:overlap] = previous[-overlap:]
            reference = make_motion(rig["rig_id"], encode_arrays(features=known), {})
            kwargs = dict(reference=reference, constraint_mode="inbetween",
                          selection=",".join(str(frame) for frame in range(overlap)))
        segment = generate_motion(model, rig, prompt, (seed + index) % 2**64,
                                  guidance, normalization, **kwargs)
        values = decode_arrays(segment["features"])["features"]
        if len(values) != 60:
            raise ValueError("Expansion requires 60-frame model segments")
        if previous is not None and not np.array_equal(values[:overlap], previous[-overlap:]):
            raise ValueError("Expansion failed to preserve the preceding overlap")
        parts.append(values if previous is None else values[overlap:])
        segments.append(dict(segment["metadata"], features_sha256=hashlib.sha256(segment["features"]).hexdigest()))
        previous = values
    combined = np.concatenate(parts, axis=0)
    return make_motion(rig["rig_id"], encode_arrays(features=combined), dict(
        mode="expansion", prompts=prompts, seed=seed, seed_policy="increment_mod_uint64",
        guidance=float(guidance), normalization=normalization, overlap=overlap,
        segments=segments, frames=len(combined), fps=30,
    ))
