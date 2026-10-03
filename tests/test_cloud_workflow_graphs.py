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
