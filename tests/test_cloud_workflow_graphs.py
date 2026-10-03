from tools.cloud_workflow import add_capture_nodes, restored_graph


def test_preprocessing_worker_graph_has_explicit_typed_boundaries():
    graph = {'2': {'class_type': 'UniMatePrepareRig'}, '4': {'class_type': 'UniMateLoadMotion'},
             'recover_fk': {}, 'recover_ric': {}, 'preview_fk': {}, 'preview_ric': {}}
    outputs = add_capture_nodes(graph)
    assert any(node.get('class_type') == 'UniMateExportFBX' for node in graph.values())
    assert {item['type_name'] for item in outputs} == {
        'UNIMATE_ASSET', 'UNIMATE_RIG', 'UNIMATE_MOTION', 'UNIMATE_CONDITIONING',
        'UNIMATE_SKELETON', 'IMAGE'}
    keys = {item['key'] for item in outputs}
    assert len(keys) == len(outputs)
    restored = restored_graph(outputs)
    assert any(node['class_type'] == 'UniMateExportFBX' for node in restored.values())
    assert {node['inputs']['boundary_key'] for node in restored.values()
            if node['class_type'] == 'CloudPartitionInput'} == keys
    assert any(node['class_type'] == 'UniMateExtractMotion' for node in restored.values())
    assert any(node['class_type'] == 'UniMatePrepareRig' and node['inputs']['asset'] == ['in_asset', 0]
               for node in restored.values())
    assert any(node['class_type'] == 'SaveImage' and node['inputs']['images'] == ['in_image_fk', 0]
               for node in restored.values())


def test_execution_list_graph_consumes_and_captures_mapped_motion_cases():
    from tools.cloud_workflow import execution_list_graph
    graph, outputs = execution_list_graph()
    assert {node['inputs']['boundary_key'] for node in graph.values()
            if node['class_type'] == 'CloudPartitionInput'} == {'rig', 'motion'}
    assert {item['key'] for item in outputs} == {'motion', 'skeleton_fk', 'skeleton_ric'}
    for method in ('fk', 'ric'):
        assert graph['recover_' + method]['inputs']['motion'] == ['in_motion', 0]
        assert graph['out_skeleton_' + method]['inputs']['value'] == ['recover_' + method, 0]


def test_batch_worker_restores_both_paired_execution_lists():
    from tools.cloud_batch_workflow import batch_boundaries, restored_batch_graph
    boundaries = batch_boundaries()
    assert boundaries == [{'key': 'motions', 'type_name': 'UNIMATE_MOTION'},
                          {'key': 'rigs', 'type_name': 'UNIMATE_RIG'}]
    graph = restored_batch_graph()
    assert graph['export']['inputs'] == {'rig': ['in_rigs', 0], 'motion': ['in_motions', 0],
                                        'filename_prefix': 'verified/restored-batch'}
    for key in ('motions', 'rigs'):
        assert graph['out_' + key]['inputs']['value'] == ['in_' + key, 0]
