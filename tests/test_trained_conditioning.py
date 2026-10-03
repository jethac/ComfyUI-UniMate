"""Training capacities and numeric statistics drive inference conditioning."""
import numpy as np
import pytest
import torch

from test_training_execution import job
from test_training_session import assert_tree
from test_training_collation import reference,assert_equal  # noqa: F401 fixture
from unimate_pack import upstream
from unimate_pack.contracts import decode_arrays
from unimate_pack.training_model import create_training_model


@pytest.mark.parametrize('attention,text_cond',[
    ('graph','adaln'),('graph','cross_attn'),('full','adaln'),('full','cross_attn')])
@pytest.mark.parametrize('frames,freqs,text_dim',[(9,4,7),(17,12,11)])
def test_trained_capacity_condition_and_actual_forward(attention,text_cond,frames,freqs,text_dim):
    value,_,stats,_=job()
    options={**value['model'],'attention':attention,'text_cond':text_cond,
        'max_motion_length':frames,'max_joints':11,'max_freqs':freqs,
        'text_dim':text_dim,'use_spectral_rope':True,'use_joint_name_emb':True,
        'use_depth_emb':True,'concat_parent_features':True,
        'num_tpos_queries':2,'inject_tpos_to_adaln':True}
    parents=np.array([-1,0,0,1,2])
    arrays=dict(parents=parents,tpos_first_frame=np.arange(15).reshape(5,3)/10)
    tokens=np.arange(3*text_dim,dtype=np.float32).reshape(3,text_dim)/10
    names=np.ones((5,text_dim),np.float32)
    cond=upstream.build_training_condition(arrays,options,stats,stats['datasets'][0],tokens,names)
    assert cond['lengths_mask'].shape[-1]==frames
    assert cond['spectral_feats'].shape[-1]==freqs
    assert cond['mean'].shape==(1,11,12)
    assert cond['joint_mask'].sum()==5
    np.testing.assert_array_equal(cond['caption_tokens'][0],tokens)
    numbers=decode_arrays(stats['arrays'])
    np.testing.assert_array_equal(cond['mean'][0,0],numbers['mean_root'][0].astype(np.float32))
    np.testing.assert_array_equal(cond['std'][0,1],numbers['std_local'][0].astype(np.float32))
    model=create_training_model(options).eval().requires_grad_(False)
    with torch.no_grad():
        output=model(torch.zeros(1,11,12,frames),torch.tensor([.5]),cond)
    assert output.shape==(1,11,12,frames) and torch.isfinite(output).all()


@pytest.mark.parametrize('kind',['label','tokens','names','parents','depth','spectral','normalization_overflow','pooled_overflow'])
def test_trained_condition_rejects_invalid_inputs(kind):
    value,_,stats,_=job()
    options=value['model']
    arrays=dict(parents=np.array([-1,0,0,1,2]),tpos_first_frame=np.zeros((5,3)))
    tokens=np.ones((3,7),np.float32)
    names=np.ones((5,7),np.float32)
    label=stats['datasets'][0]
    if kind=='label':
        label='missing'
    elif kind=='tokens':
        tokens=np.empty((0,7),np.float32)
    elif kind=='names':
        names[0,0]=np.nan
    elif kind=='parents':
        arrays['parents'][2]=2
    elif kind=='depth':
        options={**options,'max_depth':0}
    elif kind=='spectral':
        arrays['spectral_feats']=np.full((5,8),np.nan)
    elif kind=='pooled_overflow':
        tokens.fill(3e38)
    else:
        import hashlib
        from unimate_pack.contracts import encode_arrays
        numbers=decode_arrays(stats['arrays'])
        numbers['mean_root'].fill(1e40)
        stats={**stats,'arrays':encode_arrays(**numbers)}
        stats['sha256']=hashlib.sha256(stats['arrays']).hexdigest()
    with pytest.raises(ValueError):
        upstream.build_training_condition(arrays,options,stats,label,tokens,names)


def test_trained_condition_matches_released_condition():
    import json
    from pathlib import Path
    from unimate_pack.statistics import UPSTREAM_REVISION
    from unimate_pack.statistics import ARRAY_FIELDS
    from unimate_pack.contracts import encode_arrays
    import hashlib
    config=json.loads((Path(__file__).parent/'fixtures/model_configs/unimate_uniml3d_f60_v2.json').read_text(encoding='utf-8-sig'))
    options={k:v for k,v in config['model'].items() if not k.startswith('text_encoder_')}
    options.update({k:config['dataset'][k] for k in ('feature_len','max_motion_length','max_joints','max_depth')})
    parents=np.array([-1,0,0,1,2])
    arrays=dict(parents=parents,tpos_first_frame=np.arange(15).reshape(5,3)/10)
    values={name:np.full((1,12),.3 if name.startswith('mean') else 1.7) for name in ARRAY_FIELDS}
    payload=encode_arrays(**values)
    stats=dict(schema='unimate.statistics.v1',dataset_sha256='1'*64,clip_count=1,datasets=['custom'],
        options=dict(per_dataset=True,balanced=False,tie_std=False),upstream_revision=UPSTREAM_REVISION,
        arrays=payload,sha256=hashlib.sha256(payload).hexdigest())
    released={f'{family}_{name}':values[name][0] for family in ('mixamo','truebones','objaverse') for name in values}
    tokens=np.ones((3,768),np.float32)
    names=np.ones((5,768),np.float32)
    actual=upstream.build_training_condition(arrays,options,stats,'custom',tokens,names)
    expected=upstream.build_condition(arrays,config,released,'mixamo',tokens,names)
    np.testing.assert_array_equal(actual.pop('parents')[0],expected.pop('parents')[0])
    assert_tree(actual,expected)


@pytest.mark.parametrize('reason',['budget','cancel'])
def test_workspace_and_cancel_before_topology_allocation(reason,monkeypatch):
    from unimate_pack._vendor import topology_utils
    value,_,stats,_=job()
    options={**value['model'],'max_joints':4096,'max_motion_length':4096}
    arrays=dict(parents=np.array([-1,0,0,1,2]),tpos_first_frame=np.zeros((5,3)))
    def forbidden(*args,**kwargs):
        raise AssertionError('topology allocation before budget/cancel')
    monkeypatch.setattr(topology_utils,'compute_joint_depths',forbidden)
    def cancel():
        raise InterruptedError('cancelled')
    kwargs={'max_workspace_bytes':512*1024**2} if reason=='budget' else {'cancel':cancel}
    with pytest.raises(ValueError if reason=='budget' else InterruptedError):
        upstream.build_training_condition(arrays,options,stats,stats['datasets'][0],
            np.ones((3,7),np.float32),np.ones((5,7),np.float32),**kwargs)


@pytest.mark.parametrize('frames,freqs,text_dim',[(9,4,7),(17,12,11)])
def test_trained_condition_matches_pinned_numeric_pipeline(reference,frames,freqs,text_dim):  # noqa: F811 fixture
    import importlib.util
    import os
    from pathlib import Path
    root=Path(os.environ['UNIMATE_DATASET_REFERENCE'])
    def module(relative,name):
        path=root/relative
        spec=importlib.util.spec_from_file_location(name,path)
        result=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    topo=module('unimate/utils/topology_utils.py','trained_condition_reference_topology')
    transforms=module('unimate/dataset/transforms.py','trained_condition_reference_transforms')
    value,_,stats,_=job()
    options={**value['model'],'max_motion_length':frames,'max_joints':11,
        'max_freqs':freqs,'text_dim':text_dim}
    parents=np.array([-1,0,0,1,2])
    tpos=np.arange(15).reshape(5,3)/10
    tokens=np.arange(3*text_dim,dtype=np.float32).reshape(3,text_dim)/10
    names=np.ones((5,text_dim),np.float32)
    raw=np.zeros((5,12))
    raw[:,:3]=tpos
    raw[:,3:9]=[1,0,0,0,1,0]
    numbers=decode_arrays(stats['arrays'])
    mean=np.stack([numbers['mean_root'][0]]+[numbers['mean_local'][0]]*4)
    std=np.stack([numbers['std_root'][0]]+[numbers['std_local'][0]]*4)
    normalized=transforms.apply_normalization(raw,mean,std)
    relations,distances=topo.compute_edge_relations_and_distances(parents)
    batch=dict(motion=np.zeros((frames,5,12)),max_joints=11,
        motion_length=frames,start_idx=0,parents=parents,
        edge_indexs=topo.compute_edge_indexs(parents),tpos_first_frame=normalized,
        mean=mean,std=std,**transforms.build_parent_features(normalized,parents),
        joint_depths=topo.compute_joint_depths(parents),joint_relations=relations,
        joint_graph_dist=distances,spectral_feats=topo.compute_laplacian_eigenvectors(parents,max_freqs=freqs)[0],
        joint_names_emb=names,caption_emb=tokens.mean(0),caption_tokens=tokens)
    actual=upstream.build_training_condition(dict(parents=parents,tpos_first_frame=tpos),
        options,stats,stats['datasets'][0],tokens,names)
    assert_equal(actual,reference([batch])[1])
