"""Adapters for released UniMate f60 models using 5d6aabe math, float32.

Topology, collate, normalization and denoiser operations use pinned MIT sources.
The flow sampler is the upstream velocity/Linear ODE specialization; no training
loss, Motion package, dataset loader, captioning provider or download is imported.
"""

from __future__ import annotations

import numpy as np

from .bundle import SOLVER


def validate_config(config: dict) -> None:
    dataset, model = config.get("dataset", {}), config.get("model", {})
    expected_data = dict(
        feature_len=12,
        max_motion_length=60,
        topology_condition_type="tpos",
        motion_repr="unimate",
    )
    expected_model = dict(
        latent_dim=512,
        ff_size=2048,
        num_heads=8,
        dropout=0.0,
        use_spectral_rope=True,
        max_freqs=8,
        use_signnet=True,
        cond_mode="text",
        cond_mask_prob=0.1,
        use_joint_name_emb=True,
        use_graph_emb=False,
        use_depth_emb=True,
        concat_parent_features=True,
        num_tpos_queries=4,
        inject_tpos_to_adaln=True,
        use_graph_attn_bias=True,
        share_graph_attn_bias=False,
        gradient_checkpointing=False,
        text_encoder_type="t5",
        text_encoder_version="google/flan-t5-base",
    )
    if (
        any(dataset.get(k) != v for k, v in expected_data.items())
        or any(model.get(k) != v for k, v in expected_model.items())
        or (dataset.get("max_joints"), dataset.get("max_depth"),
            model.get("num_layers"), model.get("attention"), model.get("text_cond"))
        not in {(22, 7, 6, "graph", "adaln"), (61, 19, 10, "graph", "adaln"),
                (71, 19, 10, "graph", "adaln"), (71, 19, 10, "full", "cross_attn")}
        or config.get("training", {}).get("diff_model") != "flow"
        or config.get("training", {}).get("use_ema") is not True
    ):
        raise ValueError(
            "This adapter requires a released UniMate f60 EMA configuration"
        )


def create_denoiser(config: dict):
    validate_config(config)
    from ._vendor import denoiser

    dataset = config["dataset"]
    keys = (
        "latent_dim",
        "ff_size",
        "num_layers",
        "num_heads",
        "dropout",
        "use_spectral_rope",
        "max_freqs",
        "use_signnet",
        "cond_mode",
        "cond_mask_prob",
        "use_joint_name_emb",
        "use_graph_emb",
        "use_depth_emb",
        "concat_parent_features",
        "num_tpos_queries",
        "inject_tpos_to_adaln",
        "use_graph_attn_bias",
        "share_graph_attn_bias",
        "gradient_checkpointing",
    )
    constructor = (denoiser.UniMateGraphAdaLN if config["model"]["attention"] == "graph"
                   else denoiser.UniMateFullCrossAttn)
    if config["model"]["attention"] == "full":
        keys = tuple(k for k in keys if k not in (
            "use_graph_attn_bias", "share_graph_attn_bias", "gradient_checkpointing"))
    return constructor(
        **{k: config["model"][k] for k in keys},
        text_dim=768,
        **{
            k: dataset[k]
            for k in ("feature_len", "max_motion_length", "max_joints", "max_depth")
        },
    )


def validate_stats(stats: dict) -> None:
    required = {
        f"{family}_{part}"
        for family in ("objaverse", "mixamo", "truebones")
        for part in ("mean_root", "std_root", "mean_local", "std_local")
    }
    mixamo = {f"mixamo_{part}" for part in ("mean_root", "std_root", "mean_local", "std_local")}
    if set(stats) not in (required, mixamo):
        raise ValueError("Bundle requires complete released normalization families")
    for name, value in stats.items():
        if (
            value.shape != (12,)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid normalization statistics")
        if "_std_" in name and (value <= 0).any():
            raise ValueError("Normalization standard deviation must be positive")


def build_condition(
    arrays: dict,
    config: dict,
    stats: dict,
    normalization: str,
    caption_tokens: np.ndarray,
    joint_embeddings: np.ndarray,
):
    from ._vendor.collate import mixture_batch_collate
    from ._vendor.transforms import apply_normalization, build_parent_features
    from ._vendor import topology_utils as topology

    validate_config(config)
    validate_stats(stats)
    if normalization not in ("objaverse", "mixamo", "truebones"):
        raise ValueError("Choose objaverse, mixamo, or truebones normalization")
    if f"{normalization}_mean_root" not in stats:
        raise ValueError("Selected normalization is absent from this checkpoint")
    parents = arrays["parents"].astype(np.int64)
    joints = len(parents)
    if (
        parents.shape != (joints,)
        or not 5 <= joints <= config["dataset"]["max_joints"]
        or parents[0] != -1
        or any(not 0 <= parents[j] < j for j in range(1, joints))
    ):
        raise ValueError("Conditioning requires connected joints within this checkpoint's capacity")
    depth = topology.compute_joint_depths(parents)
    if depth.max() > config["dataset"]["max_depth"]:
        raise ValueError("Skeleton exceeds this checkpoint's maximum depth")
    tpos = np.asarray(arrays["tpos_first_frame"], dtype=np.float64)
    if tpos.shape != (joints, 3) or not np.isfinite(tpos).all():
        raise ValueError("Expected finite canonical rest positions (J,3)")
    # Quaternions.id(1).rotation_matrix(cont6d=True) in the pinned reference:
    # first two matrix columns, flattened as [col0, col1].
    padded = np.zeros((joints, 12), dtype=np.float64)
    padded[:, :3] = tpos
    padded[:, 3:9] = [1, 0, 0, 0, 1, 0]
    mean = np.empty((joints, 12), dtype=np.float64)
    std = np.empty_like(mean)
    for name, target in (("mean", mean), ("std", std)):
        target[0] = stats[f"{normalization}_{name}_root"]
        target[1:] = stats[f"{normalization}_{name}_local"]
    normalized = apply_normalization(padded, mean, std)
    relations, distances = topology.compute_edge_relations_and_distances(parents)
    spectral = arrays.get("spectral_feats")
    if spectral is None:
        spectral = topology.compute_laplacian_eigenvectors(parents)[0]
    if spectral.shape != (joints, 8) or not np.isfinite(spectral).all():
        raise ValueError("Invalid spectral topology conditioning")
    if (
        joint_embeddings.shape != (joints, 768)
        or caption_tokens.ndim != 2
        or caption_tokens.shape[1] != 768
    ):
        raise ValueError("Invalid local T5 embeddings")
    batch = dict(
        motion=np.zeros((60, joints, 12)),
        max_motion_length=60,
        motion_length=60,
        max_joints=config["dataset"]["max_joints"],
        parents=parents,
        edge_indexs=topology.compute_edge_indexs(parents),
        tpos_first_frame=normalized,
        **build_parent_features(normalized, parents),
        joint_graph_dist=distances,
        joint_relations=relations,
        joint_depths=depth,
        spectral_feats=spectral,
        joint_names_emb=joint_embeddings,
        start_idx=0,
        mean=mean,
        std=std,
        caption_emb=caption_tokens.mean(axis=0),
        caption_tokens=caption_tokens,
    )
    return mixture_batch_collate([batch])[1]


def sample_constrained_flow(model, cond, known, mask, seed, guidance, check_cancel, progress=None):
    import torch
    from .constrained import sample_replacement

    if not 1 < guidance <= 10:
        raise ValueError("Constrained generation requires guidance greater than 1")
    generator = torch.Generator(device=known.device).manual_seed(seed)
    noise = torch.randn(known.shape, generator=generator, device=known.device, dtype=known.dtype)

    def velocity(value, time, cond=None):
        conditional = model(value, time, cond)
        check_cancel()
        unconditional = model(value, time, cond, force_mask=True)
        return unconditional + guidance * (conditional - unconditional)

    return sample_replacement(
        velocity, cond, known, mask, noise, check_cancel=check_cancel,
        progress=(lambda index, total: progress(int(index * 100 / total))) if progress else None,
    )


def sample_flow(
    model, cond, seed: int, guidance: float, device, check_cancel, progress=None
):
    """Upstream Linear/velocity flow, same torchdiffeq defaults and CFG=1 semantics."""
    import torch
    from torchdiffeq import odeint

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, cond["mean"].shape[1], 12, cond["lengths_mask"].shape[-1]),
        device=device, generator=generator, dtype=torch.float32
    )

    def velocity(t, x):
        check_cancel()
        if progress:
            progress(min(99, max(0, int(float(t) * 100))))
        timesteps = torch.ones(x.size(0), device=device) * t
        if guidance == 1:
            return model(x, timesteps, cond, force_mask=True)
        conditional = model(x, timesteps, cond)
        check_cancel()
        unconditional = model(x, timesteps, cond, force_mask=True)
        return unconditional + guidance * (conditional - unconditional)

    # num_steps means saved output times, not an integration step count.
    times = torch.linspace(0, 1, SOLVER["num_steps"]).to(device)
    return odeint(
        velocity,
        noise,
        times,
        method=SOLVER["method"],
        atol=[SOLVER["atol"]],
        rtol=[SOLVER["rtol"]],
    )[-1]
