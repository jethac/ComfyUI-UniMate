"""Portable training jobs bound to actual prepared artifacts and epoch plans."""
from copy import deepcopy
import hashlib

from .contracts import decode_arrays,_digest
from .dataset_contracts import _fields,manifest_bytes,validate_dataset
from .dataset_selection import sampling_plan,validate_sampling,_number
from .statistics import UPSTREAM_REVISION,validate_statistics
from .training_checkpoint import _exact,_json_value
from .training_dataset_samples import (_preflight,dataset_identity,statistics_identity,
    text_cache_identity,validate_training_conditioning)
from .training_diffusion import create_diffusion_schedule
from .training_loss import create_flow_schedule
from .training_model import model_options
from .training_session import session_options
from .training_text import validate_text_cache
from .training_initialization import inspect_initialization,validate_initialization
from .training_transforms import _check,_integer

FIELDS={'schema','upstream_revision','model','optimizer','paradigm','loss','sample',
    'sampling','batch_size','drop_last','seed','datasets','sha256'}
SAMPLE_DEFAULTS=dict(mode='tpos',augmentation='none',realign_feature=True,ground_rest=True,
    embedding_policy='cached',addition_policy='released',enabled=[])


def job_identity(value):
    return hashlib.sha256(manifest_bytes({k:v for k,v in value.items() if k!='sha256'})).hexdigest()


def _sample_options(sample):
    if type(sample) is not dict or sample.keys()-SAMPLE_DEFAULTS.keys():
        raise ValueError('Invalid sample options')
    sample={**deepcopy(SAMPLE_DEFAULTS),**sample}
    for name,choices in [('mode',('tpos','first_frame')),
        ('augmentation',('none','addition','addition_linear','removal','pooling','perturbation','random')),
        ('embedding_policy',('cached','fresh')),('addition_policy',('released','neutral_fk'))]:
        if sample[name] not in choices:
            raise ValueError('Invalid sample '+name)
    if any(type(sample[k]) is not bool for k in ('realign_feature','ground_rest')):
        raise ValueError('Invalid sample flags')
    enabled=sample['enabled']
    if (type(enabled) is not list or any(type(x) is not str or x not in
        ('addition','removal','pooling','perturbation') for x in enabled) or len(set(enabled))!=len(enabled)):
        raise ValueError('Invalid enabled augmentations')
    return sample


def validate_training_job_metadata(value):
    """Validate portable configuration; actual artifact/epoch validation is separate."""
    _fields(value,FIELDS|({'initialization'} if type(value) is dict and 'initialization' in value else set()))
    _json_value(value)
    _digest(value['sha256'],'training job')
    if (value['schema']!='unimate.training_job.v1' or value['upstream_revision']!=UPSTREAM_REVISION
        or job_identity(value)!=value['sha256']):
        raise ValueError('Training job schema/source/digest mismatch')
    if not _exact(model_options(value['model']),value['model']) or not _exact(session_options(value['optimizer']),value['optimizer']):
        raise ValueError('Noncanonical training configuration')
    if value['paradigm'] not in ('flow','diffusion') or type(value['loss']) is not dict:
        raise ValueError('Invalid training paradigm/loss')
    (create_flow_schedule if value['paradigm']=='flow' else create_diffusion_schedule)(value['loss'])
    _integer(value['batch_size'],'batch_size',1)
    _integer(value['seed'],'seed')
    if value['batch_size']>4096 or value['seed']>=2**64 or type(value['drop_last']) is not bool:
        raise ValueError('Invalid training batch/seed/drop_last')
    if not _exact(_sample_options(value['sample']),value['sample']):
        raise ValueError('Noncanonical training sample options')
    _fields(value['sampling'],{'alpha','dataset_alpha'})
    _number(value['sampling']['alpha'])
    if value['sampling']['dataset_alpha'] is not None:
        _number(value['sampling']['dataset_alpha'])
    if 'initialization' in value:
        validate_initialization(value['initialization'])
    if type(value['datasets']) is not list or not 1<=len(value['datasets'])<=4096:
        raise ValueError('Invalid training dataset identities')
    for data in value['datasets']:
        _fields(data,{'dataset_sha256','statistics_sha256','text_cache_sha256'})
        for name,digest in data.items():
            _digest(digest,name)


def make_training_job(dataset,statistics,cache,options=None,*,initialization=None,cancel=None,
        max_workspace_bytes=512*1024*1024):
    """Bind a prepared dataset, which may contain multiple source dataset labels.

    This contract configures execution; it does not instantiate a model or claim
    distributed/runtime support. Epoch sampling uses the released balanced sampler.
    """
    _check(cancel)
    _integer(max_workspace_bytes,'max_workspace_bytes',1)
    options={} if options is None else options
    if type(options) is not dict or options.keys()-{
        'model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed','initialization'}:
        raise ValueError('Invalid training job options')
    _json_value(options)
    options=deepcopy(options)
    _preflight(dataset,statistics,cache,max_workspace_bytes)
    validate_dataset(dataset,cancel=cancel)
    validate_statistics(statistics)
    arrays=validate_text_cache(cache)
    identity=dataset_identity(dataset)
    if statistics['dataset_sha256']!=identity:
        raise ValueError('Statistics dataset binding mismatch')
    model=options.get('model',{})
    if type(model) is not dict:
        raise ValueError('Invalid training architecture')
    selection=options.get('initialization')
    if selection is not None:
        validate_initialization(selection)
    derived=None
    if initialization is not None:
        descriptor,derived,_=inspect_initialization(initialization,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
        if selection is not None and not _exact(selection,descriptor):
            raise ValueError('Training initialization identity mismatch')
        selection=descriptor
        if any(not _exact(value,derived[key]) for key,value in model.items() if key in derived):
            raise ValueError('Selected initialization architecture differs from options')
        model={**derived,**model}
    dimension=arrays['tokens'].shape[1]
    if 'text_dim' in model and model['text_dim']!=dimension:
        raise ValueError('Model text dimension differs from validated cache')
    model=model_options({**model,'text_dim':dimension})
    optimizer=session_options(options.get('optimizer'))
    paradigm=options.get('paradigm','flow')
    if paradigm not in ('flow','diffusion'):
        raise ValueError('Invalid training paradigm')
    loss=options.get('loss',{})
    if loss is None:
        loss={}
    (create_flow_schedule if paradigm=='flow' else create_diffusion_schedule)(loss)
    batch_size=options.get('batch_size',1)
    _integer(batch_size,'batch_size',1)
    if batch_size>4096:
        raise ValueError('Training batch size exceeds bound')
    seed=options.get('seed',0)
    _integer(seed,'seed')
    if seed>=2**64:
        raise ValueError('Training seed exceeds uint64')
    drop_last=options.get('drop_last',True)
    if type(drop_last) is not bool:
        raise ValueError('Invalid drop_last flag')
    sample=_sample_options(options.get('sample',{}))
    sampling=options.get('sampling',{})
    if type(sampling) is not dict or sampling.keys()-{'alpha','dataset_alpha'}:
        raise ValueError('Invalid sampling options')
    sampling={**dict(alpha=.5,dataset_alpha=None),**sampling}
    sampling_plan(dataset,**sampling,epoch=0,cancel=cancel)
    texts=set(cache['texts'])
    topologies={entry['id']:entry for entry in dataset['manifest']['topologies']}
    seen=set()
    for clip in dataset['manifest']['clips']:
        _check(cancel)
        if clip['split']!='train':
            continue
        if not clip['caption'] or clip['caption'] not in texts or clip['dataset_type'] not in statistics['datasets']:
            raise ValueError('Unusable training caption or statistics coverage')
        key=clip['topology_id']
        if key in seen:
            continue
        seen.add(key)
        cond=decode_arrays(dataset['files'][topologies[key]['conditioning']])
        validate_training_conditioning(cond,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
        clean=cond['clean_joint_names'].tolist()
        names=clean if all(str(n).strip() for n in clean) else cond['joint_names'].tolist()
        if any(name not in texts for name in names):
            raise ValueError('Text cache misses training joint names')
        if len(cond['parents'])>model['max_joints']:
            raise ValueError('Training topology exceeds model joint capacity')
        insertion=sample['augmentation'] in ('addition','addition_linear') or (
            sample['augmentation']=='random' and 'addition' in sample['enabled'])
        if insertion and len(cond['parents'])>=model['max_joints']:
            raise ValueError('Training augmentation exceeds model joint capacity')
    value=dict(schema='unimate.training_job.v1',upstream_revision=UPSTREAM_REVISION,
        model=model,optimizer=optimizer,paradigm=paradigm,loss=loss,sample=sample,sampling=sampling,
        batch_size=batch_size,drop_last=drop_last,seed=seed,datasets=[dict(dataset_sha256=identity,
        statistics_sha256=statistics_identity(statistics),text_cache_sha256=text_cache_identity(cache))])
    if selection is not None:
        value['initialization']=selection
    value['sha256']=job_identity(value)
    _check(cancel)
    return value


def validate_training_job(value,dataset,statistics,cache,*,cancel=None,max_workspace_bytes=512*1024*1024):
    validate_training_job_metadata(value)
    options={k:value[k] for k in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    if 'initialization' in value:
        options['initialization']=value['initialization']
    expected=make_training_job(dataset,statistics,cache,options,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    if not _exact(value,expected):
        raise ValueError('Training job artifact binding or canonical configuration mismatch')
    return expected


def plan_training_epoch(job,dataset,statistics,cache,*,epoch,cancel=None,max_workspace_bytes=512*1024*1024):
    job=validate_training_job(job,dataset,statistics,cache,cancel=cancel,max_workspace_bytes=max_workspace_bytes)
    plan=sampling_plan(dataset,**job['sampling'],epoch=epoch,cancel=cancel)
    validate_sampling(plan,dataset)
    arrays=decode_arrays(plan['arrays'])
    ids=[plan['clip_ids'][int(i)] for i in arrays['indices']]
    size=job['batch_size']
    if job['drop_last']:
        ids=ids[:len(ids)//size*size]
    batches=[ids[i:i+size] for i in range(0,len(ids),size)]
    if not batches:
        raise ValueError('Training epoch is empty after drop_last')
    value=dict(schema='unimate.training_epoch.v1',job_sha256=job['sha256'],epoch=epoch,
        sampling_sha256=plan['sha256'],batches=batches,batches_per_epoch=len(batches))
    value['sha256']=job_identity(value)
    _check(cancel)
    return value
