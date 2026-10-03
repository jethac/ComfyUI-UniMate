"""Actual batch workflow must collect a mapped sample list and restore a batch."""


def test_batch_workflow_collects_list_and_restores_portable_value():
    from tools.training_batch_workflow import workflow_graph
    for mode in ('build','restore','reload'):
        graph,inputs,outputs=workflow_graph(mode)
        assert outputs==[dict(key='batch',type_name='UNIMATE_TRAINING_BATCH')]
        if mode=='restore':
            assert inputs==[dict(key='batch',type_name='UNIMATE_TRAINING_BATCH')]
            assert 'collate' not in graph
        else:
            assert inputs==[dict(key='samples',type_name='UNIMATE_TRAINING_SAMPLE')]
            assert graph['collate']['class_type']=='UniMateCollateTrainingSamples'
            assert graph['collate']['inputs']==dict(samples=['in_samples',0],workspace_mib=512)


def test_cache_provenance_uses_full_identity():
    import pytest
    from tools.training_batch_workflow import check_sample_cache
    from test_training_text import encoder,IDENTITY
    from unimate_pack.training_text import build_text_cache
    from unimate_pack.training_dataset_samples import text_cache_identity
    cache=build_text_cache(['walk'],encoder,IDENTITY)
    identity=text_cache_identity(cache)
    assert identity!=cache['sha256']
    check_sample_cache([{'provenance':{'text_cache_sha256':identity}}],cache)
    with pytest.raises(ValueError):
        check_sample_cache([{'provenance':{'text_cache_sha256':cache['sha256']}}],cache)


def test_worker_and_server_share_managed_partition_root(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    import tools.dataset_workflow as server
    from tools.training_batch_workflow import verify
    monkeypatch.setenv('COMFY_PARTITION_ROOT','previous-root')
    def check(args,**kwargs):
        assert os.environ['COMFY_PARTITION_ROOT']==str(tmp_path.resolve()/'partition')
    monkeypatch.setattr(server,'verify',check)
    verify(SimpleNamespace(workdir=tmp_path))
    assert os.environ['COMFY_PARTITION_ROOT']=='previous-root'
