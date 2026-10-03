"""Bounded case expansion with stable per-case seeds and portable outputs."""

import math


def generate_batch(model, rigs, prompts, repetitions, seed, guidance, normalization="objaverse"):
    from .inference import generate_motion
    if not isinstance(rigs, list) or not rigs:
        raise ValueError("Select at least one prepared rig")
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 32
            or any(not isinstance(p, str) or len(p) > 4096 for p in prompts)):
        raise ValueError("Provide 1–32 prompts of at most 4096 characters")
    if type(repetitions) is not int or not 1 <= repetitions <= 64:
        raise ValueError("Repetitions must be 1–64")
    if len(rigs) * len(prompts) * repetitions > 256:
        raise ValueError("Batch exceeds the 256-case limit")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or not 1 <= guidance <= 10:
        raise ValueError("Guidance must be between 1 and 10")
    if normalization not in ("objaverse", "mixamo", "truebones"):
        raise ValueError("Select a supported normalization family")
    motions = []
    for rig in rigs:
        for prompt in prompts:
            for _ in range(repetitions):
                motion = generate_motion(model, rig, prompt, (seed + len(motions)) % 2**64,
                                         guidance, normalization)
                motions.append(motion)
    return motions


def generate_batch_with_rigs(model, rigs, prompts, repetitions, seed, guidance, normalization="objaverse"):
    """Return parallel execution lists for motion generation and rig-aware consumers."""
    motions = generate_batch(model, rigs, prompts, repetitions, seed, guidance, normalization)
    matched = [rig for rig in rigs for _ in prompts for _ in range(repetitions)]
    return motions, matched
