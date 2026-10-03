"""Dataset workflow coverage must include all public nodes and statistics modes."""

def test_dataset_workflows_cover_collection_restoration_reload_and_all_modes():
    from tools.dataset_workflow import workflow_graph
    public = set()
    for mode in ('build', 'restore', 'reload'):
        graph, inputs, outputs = workflow_graph(mode)
        public.update(node['class_type'] for node in graph.values())
        assert {item['type_name'] for item in outputs} == {'UNIMATE_DATASET', 'UNIMATE_STATISTICS', 'UNIMATE_SAMPLING'}
        assert {item['key'] for item in outputs} >= {f'plan_{i}' for i in range(4)}
        if mode != 'restore':
            stats = [node['inputs'] for node in graph.values()
                     if node['class_type'] == 'UniMateDatasetStatistics']
            assert {(s['per_dataset'], s['balanced'], s['tie_std']) for s in stats} == {
                (p, b, t) for p in (False, True) for b in (False, True) for t in (False, True)}
        if mode == 'build':
            assert {item['type_name'] for item in inputs} == {'UNIMATE_RIG', 'UNIMATE_MOTION'}
            assert graph['dataset_build']['inputs']['rigs'] == ['in_rig', 0]
            assert graph['dataset_build']['inputs']['motions'] == ['in_motion', 0]
            assert graph['dataset']['class_type'] == 'UniMateSplitDataset'
        if mode == 'reload':
            assert not inputs
            assert graph['dataset']['inputs']['archive'] == 'reloaded.unimatedata'
            assert graph['statistics']['inputs']['archive'] == 'reloaded.unimatestats'
    assert public >= {'UniMateBuildDataset', 'UniMateDatasetStatistics', 'UniMateLoadDataset',
                      'UniMateSaveDataset', 'UniMateLoadStatistics', 'UniMateSaveStatistics',
                      'UniMateSplitDataset', 'UniMatePlanSampling'}
