"""Optional pinned-checkout reference checks; never download test artifacts."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REFERENCE = os.environ.get("UNIMATE_REFERENCE")
pytestmark = pytest.mark.skipif(
    not REFERENCE, reason="Set UNIMATE_REFERENCE to the pinned local checkout"
)


def reference_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, Path(REFERENCE) / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "parents", [np.array([-1, 0, 1, 1, 3]), np.array([-1, 0, 0, 1, 1, 2, 2])]
)
def test_pinned_conditioning_and_masks(parents):
    import torch
    from unimate_pack.upstream import build_condition
    from unimate_pack._vendor import topology_utils as actual_topology

    ref_topology = reference_module(
        "unimate/utils/topology_utils.py", "reference_topology"
    )
    for operation in (
        "compute_edge_indexs",
        "compute_joint_depths",
        "compute_edge_relations_and_distances",
        "compute_laplacian_eigenvectors",
    ):
        expected = getattr(ref_topology, operation)(parents)
        actual = getattr(actual_topology, operation)(parents)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
    config_path = os.environ.get("UNIMATE_REFERENCE_CONFIG")
    if not config_path:
        pytest.skip("Set UNIMATE_REFERENCE_CONFIG to official config.json")
    config = json.loads(Path(config_path).read_text())
    joints = len(parents)
    tpos = np.arange(joints * 3).reshape(joints, 3) / 10
    arrays = dict(parents=parents, tpos_first_frame=tpos)
    stats = {
        f"{family}_{kind}_{position}": np.full(12, 0.3 if kind == "mean" else 1.7)
        for family in ("objaverse", "mixamo", "truebones")
        for kind in ("mean", "std")
        for position in ("root", "local")
    }
    tokens = np.ones((3, 768), dtype=np.float32)
    names = np.ones((joints, 768), dtype=np.float32)
    cond = build_condition(arrays, config, stats, "objaverse", tokens, names)
    transforms = reference_module(
        "unimate/dataset/transforms.py", "reference_transforms"
    )
    collate = reference_module(
        "unimate/dataset/mixture/collate.py", "reference_collate"
    )
    raw = np.zeros((joints, 12))
    raw[:, :3] = tpos
    raw[:, 3:9] = [1, 0, 0, 0, 1, 0]
    mean = np.full((joints, 12), 0.3)
    std = np.full((joints, 12), 1.7)
    normalized = transforms.apply_normalization(raw, mean, std)
    relation, distance = ref_topology.compute_edge_relations_and_distances(parents)
    batch = dict(
        motion=np.zeros((60, joints, 12)),
        max_joints=71,
        motion_length=60,
        start_idx=0,
        parents=parents,
        edge_indexs=ref_topology.compute_edge_indexs(parents),
        tpos_first_frame=normalized,
        mean=mean,
        std=std,
        **transforms.build_parent_features(normalized, parents),
        joint_depths=ref_topology.compute_joint_depths(parents),
        joint_relations=relation,
        joint_graph_dist=distance,
        spectral_feats=ref_topology.compute_laplacian_eigenvectors(parents)[0],
        joint_names_emb=names,
        caption_emb=tokens.mean(0),
        caption_tokens=tokens,
    )
    expected = collate.mixture_batch_collate([batch])[1]
    assert cond.keys() == expected.keys()
    for key in cond:
        if torch.is_tensor(cond[key]):
            torch.testing.assert_close(cond[key], expected[key], atol=0, rtol=0)
        elif key == "parents":
            np.testing.assert_array_equal(cond[key][0], expected[key][0])
        else:
            torch.testing.assert_close(cond[key][0], expected[key][0], atol=0, rtol=0)


def test_solver_reference_and_rng_isolation():
    import torch
    from unimate_pack.upstream import sample_flow

    sys.path.insert(0, str(REFERENCE))
    from unimate.models.factory import create_transport
    from unimate.models.flow.transport import Sampler
    from unimate.inference.generate import ClassifierFreeSampleModel
    from types import SimpleNamespace

    class Velocity(torch.nn.Module):
        cond_mask_prob = 0.1

        def forward(self, x, t, cond=None, force_mask=False):
            return x * 0.1 + (0.0 if force_mask else 0.02)

    model = Velocity()
    transport = create_transport(
        training_config=SimpleNamespace(lambda_geo=0, lambda_smooth=0)
    )
    state = torch.random.get_rng_state().clone()
    for guidance in (1, 3):
        actual = sample_flow(model, {}, 42, guidance, torch.device("cpu"), lambda: None)
        noise = torch.randn(
            (1, 71, 12, 60), generator=torch.Generator().manual_seed(42)
        )
        if guidance == 1:

            def reference(x, t, **kw):
                return model(x, t, force_mask=True)
        else:
            reference = ClassifierFreeSampleModel(model, guidance)
        expected = Sampler(transport).sample_ode()(noise, reference, cond={})[-1]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(state, torch.random.get_rng_state())


def test_cancellation_between_solver_evaluations():
    import torch
    from unimate_pack.upstream import sample_flow

    checks = []

    def cancel():
        checks.append(1)
        if len(checks) == 3:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        sample_flow(lambda x, *a, **kw: x * 0.1, {}, 42, 3, torch.device("cpu"), cancel)
    assert len(checks) == 3


def test_normalization_family_and_root_local_remain_distinct():
    from unimate_pack.upstream import build_condition

    config_path = os.environ.get("UNIMATE_REFERENCE_CONFIG")
    if not config_path:
        pytest.skip("Set UNIMATE_REFERENCE_CONFIG")
    config = json.loads(Path(config_path).read_text())
    stats = {}
    for i, family in enumerate(("objaverse", "mixamo", "truebones")):
        for j, part in enumerate(("mean_root", "std_root", "mean_local", "std_local")):
            stats[f"{family}_{part}"] = (
                np.arange(12, dtype=np.float64) + 1 + 20 * i + 3 * j
            )
    arrays = dict(
        parents=np.array([-1, 0, 0, 1, 2]),
        tpos_first_frame=np.arange(15).reshape(5, 3).astype(float),
    )
    raw = np.zeros((5, 12))
    raw[:, :3] = arrays["tpos_first_frame"]
    raw[:, 3:9] = [1, 0, 0, 0, 1, 0]
    for family in ("objaverse", "mixamo", "truebones"):
        cond = build_condition(
            arrays,
            config,
            stats,
            family,
            np.ones((1, 768), np.float32),
            np.ones((5, 768), np.float32),
        )
        for joint, kind in ((0, "root"), (1, "local"), (4, "local")):
            expected = (raw[joint] - stats[f"{family}_mean_{kind}"]) / stats[
                f"{family}_std_{kind}"
            ]
            np.testing.assert_array_equal(
                cond["tpos_first_frame"][0, joint].numpy(), expected.astype(np.float32)
            )
            np.testing.assert_array_equal(
                cond["mean"][0, joint].numpy(),
                stats[f"{family}_mean_{kind}"].astype(np.float32),
            )
