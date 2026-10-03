def test_foot_graph_corrects_before_export_preview_and_numeric_save():
    from tools.foot_workflow import configure_foot_graph
    graph = {'4': {'class_type': 'UniMateLoadMotion', 'inputs': {'archive': 'reference.npz'}},
             '5': {'class_type': 'UniMateExportGLB', 'inputs': {'rig': ['2', 0], 'motion': ['4', 0]}},
             'recover_fk': {'class_type': 'UniMateRecoverSkeleton', 'inputs': {'motion': ['4', 0]}}}
    configure_foot_graph(graph)
    assert graph['reference']['class_type'] == 'UniMateLoadMotion'
    assert graph['4']['class_type'] == 'UniMateFootLockMotion'
    assert graph['4']['inputs']['motion'] == ['reference', 0]
    assert graph['4']['inputs']['rig'] == ['2', 0]
    assert graph['5']['inputs']['motion'] == ['4', 0]
    assert graph['recover_fk']['inputs']['motion'] == ['4', 0]
    assert graph['save_corrected']['class_type'] == 'UniMateSaveMotion'
    assert graph['save_corrected']['inputs']['motion'] == ['4', 0]


def test_archive_evidence_requires_correction_and_authenticated_source():
    import copy
    import pytest
    from test_foot_lock import portable_legged_motion
    from unimate_pack.foot_lock import lock_motion
    from unimate_pack.motion_io import dump_motion
    from tools.foot_workflow import validate_foot_archive
    rig, motion, _ = portable_legged_motion()
    corrected, report = lock_motion(rig, motion)
    source = dump_motion(motion)
    assert validate_foot_archive(source, dump_motion(corrected)) == report
    with pytest.raises(ValueError, match='correction'):
        validate_foot_archive(source, source)
    corrupt = copy.deepcopy(corrected)
    corrupt['metadata']['foot_lock']['source_features_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source'):
        validate_foot_archive(source, dump_motion(corrupt))
