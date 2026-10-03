import pytest

from unimate_pack.batch import generate_batch


def test_batch_order_and_uint64_seed_wrap_are_stable(monkeypatch):
    from unimate_pack import inference
    calls = []

    def sample(model, rig, prompt, seed, guidance, normalization):
        calls.append((rig, prompt, seed))
        return {"case": calls[-1]}

    monkeypatch.setattr(inference, "generate_motion", sample)
    result = generate_batch({}, ["rig-a", "rig-b"], ["walk", "sit"], 2, 2**64-2, 3)
    assert len(result) == 8
    assert calls == [(rig, prompt, (2**64-2 + i) % 2**64)
                     for i, (rig, prompt) in enumerate(
                         (r, p) for r in ["rig-a", "rig-b"] for p in ["walk", "sit"] for _ in range(2))]


@pytest.mark.parametrize("prompts,repetitions", [([], 1), (["walk"], 0), ([1], 1), (["walk"]*32, 64)])
def test_invalid_or_excessive_batch_is_rejected_before_sampling(monkeypatch, prompts, repetitions):
    from unimate_pack import inference
    monkeypatch.setattr(inference, "generate_motion", lambda *a: pytest.fail("Invalid batch sampled"))
    with pytest.raises(ValueError):
        generate_batch({}, ["rig"], prompts, repetitions, 0, 3)


def test_batch_returns_matching_rigs_in_case_order(monkeypatch):
    from unimate_pack import batch, inference
    rigs = [{'rig_id': 'a'}, {'rig_id': 'b'}]
    monkeypatch.setattr(inference, 'generate_motion',
        lambda model, rig, prompt, seed, guidance, normalization: {'rig_id': rig['rig_id'], 'prompt': prompt, 'seed': seed})
    motions, matched = batch.generate_batch_with_rigs({}, rigs, ['walk', 'sit'], 2, 9, 3)
    assert matched == [rigs[0]] * 4 + [rigs[1]] * 4
    assert all(motion['rig_id'] == rig['rig_id'] for motion, rig in zip(motions, matched))
    assert [motion['seed'] for motion in motions] == list(range(9, 17))
