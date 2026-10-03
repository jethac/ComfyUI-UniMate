"""Job-driven normalized sampling compared with independent pinned schedules."""
import ast
import os
from pathlib import Path

import pytest
import torch
from torchdiffeq import odeint

from test_training_execution import job,cpu_residency
from test_training_loss import reference as flow_reference  # noqa: F401 fixture
from test_training_diffusion import reference as diffusion_reference,source_schedule  # noqa: F401 fixture
from unimate_pack import trained_sampling as sampling
from unimate_pack.training_job import make_training_job
from unimate_pack.training_dataset_samples import produce_training_sample
from unimate_pack.training_batch_contracts import collate_training_samples,validate_training_batch


class Predictor(torch.nn.Module):
    def __init__(self,variance=False):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.03,dtype=torch.float32),requires_grad=False)
        self.variance=variance
        self.eval()
    def forward(self,x,t,cond=None,force_mask=False):
        result=x*self.weight+t[:,None,None,None]*.001+(.02 if force_mask else .04)
        return torch.cat((result,result*.1),dim=1) if self.variance else result


def inputs(paradigm='flow',loss=None):
    value,dataset,stats,cache=job()
    opts={key:value[key] for key in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    opts.update(paradigm=paradigm,loss=loss or dict(lambda_geo=0.,lambda_smooth=0.))
    value=make_training_job(dataset,stats,cache,opts)
    sample=produce_training_sample(dataset,stats,cache,'a',max_motion_length=8,max_joints=16)
    _,cond=validate_training_batch(collate_training_samples([sample]))
    return value,cond


def guided(model,scale):
    def call(x,t,cond=None):
        if scale==1:
            return model(x,t,cond,force_mask=True)
        conditional=model(x,t,cond)
        uncond=model(x,t,cond,force_mask=True)
        return uncond+scale*(conditional-uncond)
    return call


def test_flow_result_owns_only_final_motion_storage():
    value,cond=inputs()
    result=sampling.sample_trained_model(Predictor(),cond,value,seed=17,
        options=dict(method='euler',num_steps=7))
    assert result.untyped_storage().nbytes()==result.numel()*result.element_size()


@pytest.mark.parametrize('path',['Linear','GVP','VP'])
@pytest.mark.parametrize('prediction',['velocity','noise','score'])
@pytest.mark.parametrize('guidance',[1.,3.])
@pytest.mark.parametrize('method',['euler','dopri5'])
def test_flow_paths_match_pinned_sampling(flow_reference,path,prediction,guidance,method):  # noqa: F811 fixture
    ref,_=flow_reference
    source=Path(os.environ['UNIMATE_DATASET_REFERENCE'])/'unimate/models/flow/integrators.py'
    import hashlib
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='0ed70897c18f346a1498c45b395888d3a1dae5f2ded3c408254d7a9e7c012c4b'
    tree=ast.parse(raw)
    tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    ctx=dict(th=torch,odeint=odeint)
    exec(compile(tree,str(source),'exec'),ctx)
    ref.Sampler.sample_ode.__globals__['ode']=ctx['ode']
    loss=dict(path_type=path,prediction=prediction,lambda_geo=0.,lambda_smooth=0.)
    value,cond=inputs(loss=loss)
    model=Predictor()
    options=dict(method=method,num_steps=7)
    initial=torch.get_rng_state().clone()
    from unimate_pack.training_loss import create_flow_schedule
    interval=create_flow_schedule(loss)
    transport=ref.Transport(model_type=getattr(ref.ModelType,prediction.upper()),
        path_type=getattr(ref.PathType,{'Linear':'LINEAR','GVP':'GVP','VP':'VP'}[path]),
        loss_type=ref.WeightType.NONE,train_eps=interval.train_eps,sample_eps=interval.sample_eps)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(17)
        noise=torch.randn(1,16,12,8)
        source_call=guided(model,guidance)
        def checked_source(x,t,cond=None):
            output=source_call(x,t,cond)
            if not torch.isfinite(output).all():
                raise FloatingPointError('Pinned sampler produced nonfinite model input/output')
            return output
        if (method,path,prediction,guidance)==('dopri5','VP','noise',3.):
            # Adaptive stages leave this path's valid time domain with this
            # fixture. Prove the source failure rather than accepting NaNs.
            with pytest.raises(FloatingPointError):
                ref.Sampler(transport).sample_ode(sampling_method=method,num_steps=7)(
                    noise,checked_source,cond=cond)
            with pytest.raises(ValueError,match='nonfinite'):
                sampling.sample_trained_model(model,cond,value,seed=17,guidance=guidance,options=options)
            return
        expected=ref.Sampler(transport).sample_ode(sampling_method=method,num_steps=7)(
            noise,checked_source,cond=cond)[-1]
    actual=sampling.sample_trained_model(model,cond,value,seed=17,guidance=guidance,options=options)
    assert torch.equal(torch.get_rng_state(),initial)
    assert torch.equal(actual,expected) and torch.isfinite(actual).all()


@pytest.mark.parametrize('method',['ancestral','ddim'])
@pytest.mark.parametrize('predict,small,rescale,variance',[
    (True,True,False,False),(False,False,True,False),(True,False,True,True),(False,True,False,True)])
@pytest.mark.parametrize('respacing',['','ddim5'])
def test_diffusion_sampling_matches_pinned(diffusion_reference,method,predict,small,rescale,variance,respacing):  # noqa: F811 fixture
    loss=dict(diffusion_steps=10,noise_schedule='cosine',lambda_geo=0.,
        predict_xstart=predict,sigma_small=small,rescale_timesteps=rescale,
        learn_sigma=variance,timestep_respacing=respacing)
    value,cond=inputs('diffusion',loss)
    model=Predictor(variance)
    options=dict(method=method,clip_denoised=False)
    if method=='ddim':
        options['eta']=.3
    initial=torch.get_rng_state().clone()
    actual=sampling.sample_trained_model(model,cond,value,seed=17,guidance=3.,options=options)
    assert torch.equal(torch.get_rng_state(),initial)
    schedule=source_schedule(diffusion_reference,loss)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(17)
        noise=torch.randn(1,16,12,8)
        fn=schedule.p_sample_loop if method=='ancestral' else schedule.ddim_sample_loop
        extra={} if method=='ancestral' else dict(eta=.3)
        expected=fn(guided(model,3.),noise.shape,noise=noise,clip_denoised=False,
            model_kwargs=dict(cond=cond),device=torch.device('cpu'),**extra)
    assert torch.equal(actual,expected) and torch.isfinite(actual).all()


@pytest.mark.parametrize('attention,text_cond',[
    ('graph','adaln'),('graph','cross_attn'),('full','adaln'),('full','cross_attn')])
@pytest.mark.parametrize('paradigm',['flow','diffusion'])
@pytest.mark.parametrize('selection',['raw','ema'])
def test_actual_trained_selected_model_generates(attention,text_cond,paradigm,selection):
    from unimate_pack.training_execution import run_training_job
    from unimate_pack.inference_weights import make_inference_weights,create_inference_model
    value,dataset,stats,cache=job()
    opts={key:value[key] for key in ('model','optimizer','paradigm','loss','sample','sampling','batch_size','drop_last','seed')}
    opts['model'].update(attention=attention,text_cond=text_cond)
    opts.update(paradigm=paradigm,loss=dict(lambda_geo=0.) if paradigm=='diffusion' else dict(lambda_geo=0.,lambda_smooth=0.))
    if paradigm=='diffusion':
        opts['loss']['diffusion_steps']=10
    value=make_training_job(dataset,stats,cache,opts)
    checkpoint,_=run_training_job(value,dataset,stats,cache,updates=2,residency=cpu_residency)
    model=create_inference_model(make_inference_weights(checkpoint,selection))
    sample=produce_training_sample(dataset,stats,cache,'a',max_motion_length=8,max_joints=16)
    _,cond=validate_training_batch(collate_training_samples([sample]))
    options=dict(method='euler',num_steps=7) if paradigm=='flow' else None
    result=sampling.sample_trained_model(model,cond,value,seed=17,guidance=3.,options=options)
    repeated=sampling.sample_trained_model(model,cond,value,seed=17,guidance=3.,options=options)
    assert result.shape==(1,16,12,8) and torch.isfinite(result).all()
    assert torch.equal(result,repeated)


@pytest.mark.parametrize('kind',['budget','cancel','guidance','shape','nonfinite','method'])
def test_reject_before_model_call(kind,monkeypatch):
    value,cond=inputs()
    model=Predictor()
    kwargs=dict(seed=17,guidance=3.)
    if kind=='budget':
        kwargs['max_workspace_bytes']=1
    elif kind=='cancel':
        def cancel():
            raise InterruptedError('cancelled')
        kwargs['cancel']=cancel
    elif kind=='guidance':
        kwargs['guidance']=True
    elif kind=='shape':
        cond['mean']=cond['mean'][:,:-1]
    elif kind=='nonfinite':
        cond['mean'][0,0,0]=float('nan')
    else:
        kwargs['options']={'method':'unknown'}
    def forbidden(*args,**kwargs):
        raise AssertionError('model called before preflight')
    monkeypatch.setattr(model,'forward',forbidden)
    with pytest.raises(InterruptedError if kind=='cancel' else ValueError):
        sampling.sample_trained_model(model,cond,value,**kwargs)


@pytest.mark.parametrize('paradigm,options',[
    ('flow',{'method':[]}),('flow',{'method':{}}),('flow',{'num_steps':True}),
    ('flow',{'num_steps':1}),('flow',{'atol':0}),('flow',{'rtol':True}),
    ('diffusion',{'clip_denoised':1}),('diffusion',{'eta':.1}),
    ('diffusion',{'method':'ddim','eta':float('nan')}),('diffusion',{'method':'ddim','eta':True})])
def test_invalid_sampling_options_are_value_errors(paradigm,options):
    with pytest.raises(ValueError):
        sampling.sampling_options(paradigm,options)


@pytest.mark.parametrize('paradigm',['flow','diffusion'])
@pytest.mark.parametrize('failure',['cancel','nonfinite'])
def test_mid_sampling_failure_restores_process_rng(paradigm,failure,monkeypatch):
    loss=dict(lambda_geo=0.,lambda_smooth=0.) if paradigm=='flow' else dict(lambda_geo=0.,diffusion_steps=10)
    value,cond=inputs(paradigm,loss)
    model=Predictor()
    calls=[]
    original=model.forward
    def forward(*args,**kwargs):
        calls.append(True)
        result=original(*args,**kwargs)
        if failure=='nonfinite' and len(calls)>1:
            return result*float('nan')
        return result
    monkeypatch.setattr(model,'forward',forward)
    def cancel():
        if failure=='cancel' and len(calls)>1:
            raise InterruptedError('cancelled during sampling')
    before=torch.get_rng_state().clone()
    with pytest.raises(InterruptedError if failure=='cancel' else ValueError):
        sampling.sample_trained_model(model,cond,value,seed=17,guidance=3.,cancel=cancel)
    assert len(calls)>1 and torch.equal(torch.get_rng_state(),before)
