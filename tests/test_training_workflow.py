"""Training workflow covers producers, operations and portable restore."""

def test_training_workflow_graph_covers_all_operations_modes_and_restore():
    from tools.training_workflow import workflow_graph,CASES
    assert {(case['augmentation'],case['mode']) for case in CASES} >= {
        (operation,mode) for operation in ('none','addition','addition_linear','removal','pooling','perturbation','random')
        for mode in ('tpos','first_frame')}
    assert any(case['addition_policy']=='neutral_fk' for case in CASES)
    for mode in ('build','restore','reload'):
        graph,inputs,outputs=workflow_graph(mode)
        assert len([o for o in outputs if o['type_name']=='UNIMATE_TRAINING_SAMPLE'])==len(CASES)
        classes={node['class_type'] for node in graph.values()}
        for node in graph.values():
            if node['class_type']=='UniMatePrepareTrainingSample':
                assert {'workspace_mib','embedding_policy','ground_rest','realign_feature',
                        'max_freqs','start_idx'} <= node['inputs'].keys()
        if mode=='build':
            assert {'UniMateModelLoader','UniMateBuildTextCache','UniMatePrepareTrainingSample'}<=classes
        elif mode=='restore':
            assert 'UniMatePrepareTrainingSample' in classes
            assert 'UniMateBuildTextCache' not in classes
        else:
            assert 'UniMatePrepareTrainingSample' not in classes
        assert {'UniMateSaveDataset','UniMateSaveStatistics'}<=classes
