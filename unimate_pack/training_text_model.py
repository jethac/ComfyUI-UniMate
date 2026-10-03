"""Installed ComfyUI-managed text encoder to portable dataset text cache."""
import hashlib

from .contracts import decode_arrays,validate_model
from .dataset_contracts import manifest_bytes,validate_dataset
from .training_text import build_text_cache
from .training_transforms import _check


def installed_encoder_identity(manifest):
    inventory={key:value for key,value in manifest['files'].items() if key.startswith('text_encoder/')}
    if not inventory or type(manifest.get('text_encoder')) is not dict:
        raise ValueError('Missing installed encoder inventory')
    return dict(encoder_type='t5',encoder_version=manifest['text_encoder']['id'],
        artifact_sha256=hashlib.sha256(manifest_bytes(dict(files=inventory,encoder=manifest['text_encoder']))).hexdigest())


def build_dataset_text_cache(model,dataset,*,existing=None,chunk_size=256,cancel=None):
    from .bundle import inspect_bundle
    from .inference import _LOCK,_get_runtime
    _check(cancel)
    validate_model(model)
    validate_dataset(dataset,cancel=cancel)
    texts=[]
    for entry in dataset['manifest']['topologies']:
        _check(cancel)
        cond=decode_arrays(dataset['files'][entry['conditioning']])
        clean=cond['clean_joint_names'].tolist()
        names=clean if all(str(name).strip() for name in clean) else cond['joint_names'].tolist()
        texts.extend(names)
    texts.extend(clip['caption'] for clip in dataset['manifest']['clips'] if clip['caption'])
    texts=list(dict.fromkeys(texts))
    manifest=inspect_bundle(model['bundle'])
    identity=installed_encoder_identity(manifest)
    def encode(chunk):
        _check(cancel)
        return _get_runtime(model).encode(chunk)
    with _LOCK:
        return build_text_cache(texts,encode,identity,existing=existing,
                                chunk_size=chunk_size,cancel=cancel)
