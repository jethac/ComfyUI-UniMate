import numpy as np
import pytest

from unimate_pack.contracts import make_motion, encode_arrays, decode_arrays
from unimate_pack.expansion import expand_motion


def test_expansion_pins_previous_tail_and_omits_duplicate_overlap(monkeypatch):
    from unimate_pack import inference
    calls = []
    rig = {"rig_id": "a" * 64}
    def generate(model, target, prompt, seed, guidance, normalization, **kwargs):
        index = len(calls)
        values = np.full((60, 5, 12), index + 1, np.float32)
        if index:
            reference = decode_arrays(kwargs["reference"]["features"])["features"]
            np.testing.assert_array_equal(reference[:10], calls[-1][1][-10:])
            assert kwargs["selection"] == ",".join(str(i) for i in range(10))
            values[:10] = reference[:10]
        calls.append((prompt, values, seed))
        return make_motion(target["rig_id"], encode_arrays(features=values), {"prompt": prompt})
    monkeypatch.setattr(inference, "generate_motion", generate)
    result = expand_motion({}, rig, ["stand", "walk", "turn"], 7, 3, overlap=10)
    features = decode_arrays(result["features"])["features"]
    assert features.shape == (160, 5, 12)
    np.testing.assert_array_equal(features[:60], 1)
    np.testing.assert_array_equal(features[60:110], 2)
    np.testing.assert_array_equal(features[110:], 3)
    assert [call[2] for call in calls] == [7, 8, 9]
    assert result["metadata"]["frames"] == 160


@pytest.mark.parametrize("prompts,overlap", [([], 10), (["walk"], 0), (["walk"], 60), ([3], 10), ("walk", 10)])
def test_bad_expansion_requests_fail_before_inference(prompts, overlap):
    with pytest.raises(ValueError):
        expand_motion({}, {}, prompts, 0, 3, overlap=overlap)
