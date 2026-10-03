"""Offline trained bundles binding selected weights, statistics and encoder."""
import hashlib
import io
import re
import zipfile

from .bundle import REQUIRED,ALLOWED,MAX_BUNDLE_BYTES,_json
from .contracts import _digest,make_model,validate_model
from .dataset_contracts import _fields,manifest_bytes
from .inference_weights import validate_inference_weights,DEFAULT_WORKSPACE
from .statistics import UPSTREAM_REVISION,validate_statistics
from .training_checkpoint import _exact
from .training_dataset_samples import statistics_identity,text_cache_identity
from .training_text import validate_text_cache,_encoder_identity,_texts,FIELDS as CACHE_FIELDS
from .training_text_model import installed_encoder_identity
from .trained_sampling import sampling_options
from .training_transforms import _check,_integer

BASE={'weights.json','denoiser.safetensors','statistics.json','stats.npz','text_cache.json'}
ENCODER_REQUIRED={name for name in REQUIRED if name.startswith('text_encoder/')}
ENCODER_ALLOWED={name for name in ALLOWED if name.startswith('text_encoder/')}
MANIFEST_FIELDS={'schema','upstream_revision','weights','model_revision','job_sha256',
    'weights_identity_sha256','statistics_identity_sha256','text_cache_identity_sha256',
    'text_encoder','solver','fps','files'}


def _budget(size,workspace):
    _integer(workspace,'max_workspace_bytes',1)
    if size*8+64*1024**2>workspace:
        raise ValueError('Trained bundle exceeds workspace budget')


def _requires_encoder(options):
    return options['cond_mode']=='text' or options['use_joint_name_emb']


def _cache_metadata(value):
    _fields(value,CACHE_FIELDS-{'arrays'})
    if value['schema']!='unimate.text_cache.v1' or value['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Invalid trained cache source/schema')
    _encoder_identity(value['encoder'])
    _texts(value['texts'])
    if value['texts']!=sorted(set(value['texts'])):
        raise ValueError('Cache texts must be sorted and unique')
    _digest(value['sha256'],'text arrays')


def _binding(weights,statistics,cache):
    binding=dict(dataset_sha256=statistics['dataset_sha256'],
        statistics_sha256=statistics_identity(statistics),text_cache_sha256=text_cache_identity(cache))
    if binding not in weights['job']['datasets']:
        raise ValueError('Trained statistics/cache binding differs from job')
    return binding


def _encoder(manifest,cache,options,names):
    encoded={name for name in names if name.startswith('text_encoder/')}
    if manifest['text_encoder'] is None:
        if encoded or _requires_encoder(options):
            raise ValueError('Trained model requires bound encoder assets')
        return
    encoder=manifest['text_encoder']
    _fields(encoder,{'id','revision'})
    if (encoder['id']!='google/flan-t5-base' or type(encoder['revision']) is not str
        or not re.fullmatch('[0-9a-f]{40}',encoder['revision'])
        or not ENCODER_REQUIRED<=encoded or encoded-ENCODER_ALLOWED
        or options['text_dim']!=768 or installed_encoder_identity(manifest)!=cache['encoder']):
        raise ValueError('Trained encoder inventory/identity/dimension mismatch')


def validate_trained_archive(manifest,archive,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    """Called after common archive safety and every file digest have passed."""
    _check(cancel)
    _fields(manifest,MANIFEST_FIELDS)
    if manifest['schema']!='unimate.bundle.v2' or manifest['upstream_revision']!=UPSTREAM_REVISION:
        raise ValueError('Invalid trained bundle schema/source')
    names=set(manifest['files'])
    if not BASE<=names or names-(BASE|ENCODER_ALLOWED):
        raise ValueError('Invalid trained bundle inventory')
    if type(manifest['fps']) is not int or manifest['fps']!=30:
        raise ValueError('Trained motion requires 30 fps')
    weights_meta=_json(archive.read('weights.json'))
    if type(weights_meta) is not dict or 'tensors' in weights_meta:
        raise ValueError('Invalid trained weights metadata')
    weights=dict(weights_meta,tensors=archive.read('denoiser.safetensors'))
    validate_inference_weights(weights,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    stats_meta=_json(archive.read('statistics.json'))
    if type(stats_meta) is not dict or 'arrays' in stats_meta:
        raise ValueError('Invalid trained statistics metadata')
    statistics=dict(stats_meta,arrays=archive.read('stats.npz'))
    validate_statistics(statistics)
    cache=_json(archive.read('text_cache.json'))
    _cache_metadata(cache)
    binding=_binding(weights,statistics,cache)
    expected=dict(weights=weights['weights'],model_revision=weights['source_checkpoint_sha256'],
        job_sha256=weights['job']['sha256'],weights_identity_sha256=weights['identity_sha256'],
        statistics_identity_sha256=binding['statistics_sha256'],text_cache_identity_sha256=binding['text_cache_sha256'])
    if any(not _exact(manifest[key],value) for key,value in expected.items()):
        raise ValueError('Trained bundle metadata binding mismatch')
    if not _exact(sampling_options(weights['job']['paradigm'],manifest['solver']),manifest['solver']):
        raise ValueError('Noncanonical trained sampler options')
    _encoder(manifest,cache,weights['job']['model'],names)
    _check(cancel)
    return weights,statistics


def assemble_trained_bundle(weights,statistics,text_cache,encoder_model=None,sampling=None,*,
    cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    from .bundle import inspect_bundle
    _check(cancel)
    size=0
    for value,key in ((weights,'tensors'),(statistics,'arrays'),(text_cache,'arrays')):
        if type(value) is not dict or type(value.get(key)) is not bytes:
            raise ValueError('Invalid trained numeric payload')
        size+=len(value[key])
    if encoder_model is not None:
        if type(encoder_model) is not dict or type(encoder_model.get('bundle')) is not bytes:
            raise ValueError('Invalid installed encoder model')
        size+=len(encoder_model['bundle'])
    _budget(size,max_workspace_bytes)
    validate_inference_weights(weights,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    validate_statistics(statistics)
    arrays=validate_text_cache(text_cache)
    if arrays['tokens'].shape[1]!=weights['job']['model']['text_dim']:
        raise ValueError('Trained cache text dimension mismatch')
    binding=_binding(weights,statistics,text_cache)
    solver=sampling_options(weights['job']['paradigm'],sampling)
    files={'weights.json':manifest_bytes({k:v for k,v in weights.items() if k!='tensors'}),
        'denoiser.safetensors':weights['tensors'],
        'statistics.json':manifest_bytes({k:v for k,v in statistics.items() if k!='arrays'}),
        'stats.npz':statistics['arrays'],
        'text_cache.json':manifest_bytes({k:v for k,v in text_cache.items() if k!='arrays'})}
    encoder=None
    if encoder_model is not None:
        validate_model(encoder_model,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
        donor=inspect_bundle(encoder_model['bundle'],cancel=cancel,max_workspace_bytes=max_workspace_bytes)
        encoder=donor['text_encoder']
        _encoder(donor,text_cache,weights['job']['model'],set(donor['files']))
        with zipfile.ZipFile(io.BytesIO(encoder_model['bundle'])) as archive:
            for name in donor['files']:
                if name.startswith('text_encoder/'):
                    _check(cancel)
                    files[name]=archive.read(name)
    manifest=dict(schema='unimate.bundle.v2',upstream_revision=UPSTREAM_REVISION,
        weights=weights['weights'],model_revision=weights['source_checkpoint_sha256'],
        job_sha256=weights['job']['sha256'],weights_identity_sha256=weights['identity_sha256'],
        statistics_identity_sha256=binding['statistics_sha256'],text_cache_identity_sha256=binding['text_cache_sha256'],
        text_encoder=encoder,solver=solver,fps=30,
        files={name:dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw)) for name,raw in files.items()})
    _encoder(manifest,text_cache,weights['job']['model'],set(files))
    if sum(map(len,files.values()))>MAX_BUNDLE_BYTES:
        raise ValueError('Trained bundle exceeds size limit')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in [('manifest.json',manifest_bytes(manifest)),*sorted(files.items())]:
            _check(cancel)
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3
            info.external_attr=0o100644<<16
            archive.writestr(info,raw)
    payload=output.getvalue()
    inspect_bundle(payload,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    _check(cancel)
    model=make_model(payload,'trained.unimate',cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    _check(cancel)
    return model


def read_trained_components(payload,*,cancel=None,max_workspace_bytes=DEFAULT_WORKSPACE):
    from .bundle import inspect_bundle
    manifest=inspect_bundle(payload,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    if manifest['schema']!='unimate.bundle.v2':
        raise ValueError('Expected complete trained model bundle')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return validate_trained_archive(manifest,archive,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
