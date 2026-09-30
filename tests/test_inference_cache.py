from types import SimpleNamespace

from unimate_pack.inference import _release_rope_cache


def test_detach_drops_unregistered_upstream_rope_tensors():
    temporal = SimpleNamespace(
        nd=2, cos_0=object(), sin_0=object(), cos_1=object(), sin_1=object()
    )
    spatial = SimpleNamespace(nd=1, cos_0=object(), sin_0=object())
    patcher = SimpleNamespace(model=SimpleNamespace(rope_t=temporal, rope_j=spatial))
    _release_rope_cache(patcher, True)
    assert vars(temporal) == {"nd": 2}
    assert vars(spatial) == {"nd": 1}
    _release_rope_cache(patcher, True)
