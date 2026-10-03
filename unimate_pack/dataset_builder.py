"""Collect strictly paired prepared rigs and motions into numeric shards."""

import hashlib
import json
from pathlib import PurePosixPath

from .contracts import MAX_JSON_BYTES, _unique_json_pairs, validate_motion, validate_rig
from .dataset_contracts import MAX_ENTRIES, make_dataset


def build_dataset(rigs, motions, labels, default_dataset, *, cancel=None):
    if (type(rigs) is not list or type(motions) is not list
            or not 1 <= len(rigs) <= MAX_ENTRIES or len(rigs) != len(motions)):
        raise ValueError('Dataset requires equal nonempty paired rig/motion lists')
    if type(labels) is not str or len(labels.encode('utf-8')) > MAX_JSON_BYTES:
        raise ValueError('Dataset labels require bounded JSON text')
    metadata = json.loads(labels, object_pairs_hook=_unique_json_pairs)
    if type(metadata) is not list or len(metadata) not in (0, len(rigs)):
        raise ValueError('Labels must be empty or match the paired clip count')
    allowed = {'id', 'dataset_type', 'object_type', 'caption', 'split'}
    if any(type(entry) is not dict or not set(entry) <= allowed for entry in metadata):
        raise ValueError('Unsupported dataset label fields')
    files, topologies, clips = {}, {}, []
    for index, (rig, motion) in enumerate(zip(rigs, motions)):
        if cancel is not None:
            cancel()
        validate_rig(rig)
        validate_motion(motion, rig)
        conditioning_id = hashlib.sha256(rig['conditioning']).hexdigest()
        feature_id = hashlib.sha256(motion['features']).hexdigest()
        files[conditioning_id] = rig['conditioning']
        files[feature_id] = motion['features']
        if conditioning_id not in topologies:
            topologies[conditioning_id] = dict(id=conditioning_id, source_rig_id=rig['rig_id'],
                                               conditioning=conditioning_id)
        elif topologies[conditioning_id]['source_rig_id'] != rig['rig_id']:
            topologies[conditioning_id]['source_rig_id'] = None
        entry = dict(id=f'clip{index + 1}', dataset_type=default_dataset,
                     object_type=PurePosixPath(rig['asset']['name']).stem,
                     caption=motion['metadata'].get('prompt', ''), split='train')
        if metadata:
            entry.update(metadata[index])
        clips.append(dict(entry, topology_id=conditioning_id, features=feature_id,
                          source_rig_id=rig['rig_id'],
                          origin=motion['metadata'].get('canonical_root_origin', [0, 0, 0]), fps=30))
    return make_dataset(dict(topologies=list(topologies.values()), clips=clips), files, cancel=cancel)
