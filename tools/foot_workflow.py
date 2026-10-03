"""Foot-lock correction graph for archive-based integration verification."""

import hashlib

import numpy as np


def configure_foot_graph(graph):
    graph['reference'] = graph['4']
    graph['4'] = {'class_type': 'UniMateFootLockMotion', 'inputs': {
        'rig': ['2', 0], 'motion': ['reference', 0], 'joint_names': ''}}
    graph['save_corrected'] = {'class_type': 'UniMateSaveMotion', 'inputs': {
        'motion': ['4', 0], 'filename_prefix': 'verified/foot-locked'}}


def validate_foot_archive(source, saved):
    """Require an actual correction with unchanged root and rig identity."""
    from unimate_pack.contracts import decode_arrays
    from unimate_pack.motion_io import load_motion

    reference = load_motion(source)
    corrected = load_motion(saved, reference['rig_id'])
    report = corrected['metadata'].get('foot_lock')
    if (not isinstance(report, dict) or not report.get('segments')
            or not any(not segment.get('skipped', True) for segment in report['segments'])):
        raise ValueError('Saved motion has no active foot correction evidence')
    if report.get('source_features_sha256') != hashlib.sha256(reference['features']).hexdigest():
        raise ValueError('Foot correction source digest does not match reference')
    before = decode_arrays(reference['features'])['features']
    after = decode_arrays(corrected['features'])['features']
    if before.shape != after.shape or np.array_equal(before, after):
        raise ValueError('Saved motion has no changed same-length correction')
    if (not np.array_equal(before[:, 0], after[:, 0])
            or reference['metadata'].get('canonical_root_origin', [0, 0, 0])
            != corrected['metadata'].get('canonical_root_origin', [0, 0, 0])):
        raise ValueError('Foot correction changed root motion or origin')
    return report
