"""Real offline EMA/T5/ComfyUI integration, opt-in via installed local artifacts."""

import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import pytest

BUNDLE = os.environ.get("UNIMATE_TEST_BUNDLE")
REFERENCE = os.environ.get("UNIMATE_REFERENCE")
CHECKPOINT = os.environ.get("UNIMATE_TEST_CHECKPOINT")
COMFY = os.environ.get("UNIMATE_TEST_COMFY")
pytestmark = pytest.mark.skipif(
    not all((BUNDLE, REFERENCE, CHECKPOINT, COMFY)),
    reason="Requires explicit local bundle, checkpoint, upstream and ComfyUI checkout",
)


def test_real_ema_offline_reference_and_comfy_unload(monkeypatch):
    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(REFERENCE))
    import torch

    torch.set_num_threads(4)
    from comfy import model_management as mm
    from unimate_pack.inference import load_model_bundle, _Runtime
    from unimate_pack.upstream import build_condition, sample_flow
    from unimate.configs.schema import MainConfig
    from unimate.models.factory import create_model, create_transport
    from unimate.models.flow.transport import Sampler
    from unimate.inference.generate import ClassifierFreeSampleModel

    def no_network(*args, **kwargs):
        raise AssertionError("Inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    started = time.perf_counter()
    model = load_model_bundle(BUNDLE)
    runtime = _Runtime(model)
    # Both local encoders' outputs match before/after safe bundle conversion.
    from transformers import T5EncoderModel, T5Tokenizer

    original_encoder = Path(os.environ["UNIMATE_TEST_TEXT_ENCODER"])
    texts = ["A character walks forward.", "root", "left knee", ""]
    sequence, pooled = runtime.encode(texts)
    tokenizer = T5Tokenizer.from_pretrained(
        str(original_encoder), local_files_only=True
    )
    text_model = (
        T5EncoderModel.from_pretrained(
            str(original_encoder), local_files_only=True, use_safetensors=True
        )
        .float()
        .eval()
        .to(runtime.encoder.load_device)
    )
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(
        runtime.encoder.load_device
    )
    tokens["attention_mask"][-1] = 0
    with torch.inference_mode():
        hidden = text_model(**tokens).last_hidden_state
        expected_pool = (
            (
                (hidden * tokens["attention_mask"].unsqueeze(-1)).sum(-2)
                / tokens["attention_mask"].sum(-1, keepdim=True).clamp(min=1)
            )
            .cpu()
            .numpy()
        )
    np.testing.assert_array_equal(pooled, expected_pool)
    assert np.count_nonzero(sequence[-1]) == 0
    del text_model
    config = MainConfig.from_json(str(Path(CHECKPOINT).parents[1] / "config.json"))
    reference = create_model(config.dataset, config.model).float().eval()
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    reference.load_state_dict(source["model_state_dict"], strict=True)
    with torch.no_grad():
        for parameter, ema in zip(
            reference.parameters(), source["ema_state_dict"]["shadow_params"]
        ):
            parameter.copy_(ema)
    del source
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(
            value, runtime.denoiser.model.state_dict()[key], atol=0, rtol=0
        )
    parents = np.array([-1, 0, 0, 1, 2])
    arrays = dict(
        parents=parents,
        tpos_first_frame=np.array(
            [[0, 1, 0], [-0.2, 0.6, 0], [0.2, 0.6, 0], [-0.2, 0, 0], [0.2, 0, 0]]
        ),
    )
    names = ["root", "left leg", "right leg", "left foot", "right foot"]
    cond = build_condition(
        arrays,
        runtime.config,
        runtime.stats,
        "objaverse",
        sequence[0],
        runtime.encode(names)[1],
    )
    mm.load_models_gpu([runtime.denoiser], force_full_load=True)
    device = runtime.denoiser.load_device
    reference.to(device)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(
            value, runtime.denoiser.model.state_dict()[key], atol=0, rtol=0
        )
    cond = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond.items()}
    global_rng = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        probe = torch.randn(
            (1, 71, 12, 60),
            device=device,
            generator=torch.Generator(device=device).manual_seed(123),
        )
        timestep = torch.full((1,), 0.3, device=device)
        direct = runtime.denoiser.model(probe, timestep, cond)
        direct_ref = reference(probe, timestep, cond)
        direct_repeat = runtime.denoiser.model(probe, timestep, cond)
        torch.testing.assert_close(direct, direct_repeat, atol=0, rtol=0)
        for key, buffer in reference.named_buffers():
            torch.testing.assert_close(
                buffer,
                dict(runtime.denoiser.model.named_buffers())[key],
                atol=0,
                rtol=0,
            )
        print("FORWARD_MAX_ERROR", float((direct - direct_ref).abs().max()), flush=True)
        torch.testing.assert_close(direct, direct_ref, atol=2e-5, rtol=2e-5)
        actual = sample_flow(
            runtime.denoiser.model,
            cond,
            123,
            3.0,
            device,
            mm.throw_exception_if_processing_interrupted,
        )
        noise = torch.randn(
            (1, 71, 12, 60),
            device=device,
            generator=torch.Generator(device=device).manual_seed(123),
        )
        sampler = Sampler(create_transport(training_config=config.training))
        expected = sampler.sample_ode()(
            noise, ClassifierFreeSampleModel(reference, 3.0), cond=cond
        )[-1]
        # Isolate the adapter solver from ModelPatcher allocation/layout: both
        # solvers execute the very same original upstream model instance.
        same_model_adapter = sample_flow(
            reference,
            cond,
            123,
            3.0,
            device,
            mm.throw_exception_if_processing_interrupted,
        )
        crossover_error = float((same_model_adapter - expected).abs().max())
        print("SAME_UPSTREAM_MODEL_SOLVER_ERROR", crossover_error, flush=True)
        torch.testing.assert_close(same_model_adapter, expected, atol=0, rtol=0)
    max_error = float((actual - expected).abs().max())
    print(
        "SAMPLE_MAX_ERROR",
        max_error,
        "VALID_JOINT_ERROR",
        float((actual[:, :5] - expected[:, :5]).abs().max()),
        flush=True,
    )
    # Independently allocated original and Comfy-managed CUDA networks differ
    # by ~6e-6 in one forward despite bit-identical state tensors. Adaptive
    # float32 integration accumulates that error; retain a bounded 2e-4 gate.
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    assert torch.isfinite(actual).all()
    assert torch.equal(global_rng, torch.random.get_rng_state())
    mm.unload_all_models()
    assert (
        next(runtime.denoiser.model.parameters()).device
        == runtime.denoiser.offload_device
    )
    assert (
        next(runtime.encoder.model.parameters()).device
        == runtime.encoder.offload_device
    )
    assert not hasattr(runtime.denoiser.model.rope_t, "cos_0")
    assert not hasattr(runtime.denoiser.model.rope_t, "sin_0")
    mm.load_models_gpu([runtime.denoiser], force_full_load=True)
    assert next(runtime.denoiser.model.parameters()).device == device
    mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            sample_flow(
                runtime.denoiser.model,
                cond,
                123,
                3,
                device,
                mm.throw_exception_if_processing_interrupted,
            )
    finally:
        mm.interrupt_current_processing(False)
        mm.unload_all_models()
    report = dict(
        torch=torch.__version__,
        device=str(device),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        precision="float32",
        max_sample_error=max_error,
        max_valid_joint_error=float((actual[:, :5] - expected[:, :5]).abs().max()),
        max_forward_error=float((direct - direct_ref).abs().max()),
        sample_tolerance=2e-4,
        identical_model_solver_error=crossover_error,
        repeated_managed_forward_exact=True,
        elapsed_seconds=time.perf_counter() - started,
        offline=True,
        ema_exact=True,
        text_embeddings_exact=True,
        cancellation=True,
        comfy_unload_reload=True,
        model_sha256=model["sha256"],
    )
    report_path = os.environ.get("UNIMATE_TEST_REPORT")
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
