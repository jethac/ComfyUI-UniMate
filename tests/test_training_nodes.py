"""Public training nodes using actual ComfyUI V3 schemas."""
import asyncio
import json

from test_nodes import nodes, extension_module
from test_training_dataset_samples import inputs
from test_training_text import IDENTITY,encoder
from unimate_pack.training_text import build_text_cache


def test_registered_training_schemas():
    extension=asyncio.run(extension_module.comfy_entrypoint())
    ids=[cls.GET_SCHEMA().node_id for cls in asyncio.run(extension.get_node_list())]
    assert len(ids)==len(set(ids))==37
    assert 'UniMateExportInferenceWeights' in ids
    assert 'UniMateLoadInferenceWeights' in ids
    assert 'UniMateSaveTrainingCheckpoint' in ids
    assert 'UniMateLoadTrainingCheckpoint' in ids
    assert 'UniMateTrain' in ids
    assert 'UniMateTrainingJob' in ids
    assert 'UniMateBuildTextCache' in ids
    assert 'UniMatePrepareTrainingSample' in ids
    assert 'UniMateCollateTrainingSamples' in ids
    assert nodes.UniMateBuildTextCache.GET_NODE_INFO_V1()['output']==['UNIMATE_TEXT_CACHE','STRING']
    assert nodes.UniMatePrepareTrainingSample.GET_NODE_INFO_V1()['output']==['UNIMATE_TRAINING_SAMPLE','STRING']
    assert nodes.UniMateCollateTrainingSamples.GET_SCHEMA().is_input_list
    assert nodes.UniMateCollateTrainingSamples.GET_NODE_INFO_V1()['output']==['UNIMATE_TRAINING_BATCH','STRING']


def test_actual_batch_collection_node():
    from unimate_pack.training_batch_contracts import validate_training_batch
    dataset,stats,cache=inputs()
    sample=nodes.UniMatePrepareTrainingSample.execute(dataset,stats,cache,'a',max_motion_length=8,max_joints=16).result[0]
    result=nodes.UniMateCollateTrainingSamples.execute([sample,sample],[512])
    motion,cond=validate_training_batch(result.result[0])
    assert motion.shape==(2,16,12,8)
    assert cond['caption']==[sample['metadata']['caption']]*2
    assert len(json.loads(result.result[1])['samples'])==2


def test_actual_training_job_node():
    from test_training_job import config
    from unimate_pack.training_job import validate_training_job
    dataset,stats,cache=inputs()
    result=nodes.UniMateTrainingJob.execute(dataset,stats,cache,json.dumps(config()))
    job=result.result[0]
    assert validate_training_job(job,dataset,stats,cache)==job
    assert json.loads(result.result[1])['sha256']==job['sha256']
    assert nodes.UniMateTrainingJob.GET_NODE_INFO_V1()['output']==['UNIMATE_TRAINING_JOB','STRING']


def test_actual_managed_training_node_and_portable_resume():
    from test_training_job import config
    from unimate_pack.training_execution import run_training_job
    from comfy import model_management as mm
    dataset,stats,cache=inputs()
    job=nodes.UniMateTrainingJob.execute(dataset,stats,cache,json.dumps(config())).result[0]
    before=set(mm.loaded_models())
    first=nodes.UniMateTrain.execute(job,dataset,stats,cache,updates=1)
    checkpoint=first.result[0]
    assert json.loads(first.result[1])['updates']==1
    assert set(mm.loaded_models())==before
    # Recreate/restore through a different Python package namespace on the same
    # ComfyUI-selected device, as partition installations may rename the pack.
    from contextlib import contextmanager
    @contextmanager
    def selected(model):
        yield model.to(mm.get_torch_device())
    resumed,report=run_training_job(job,dataset,stats,cache,residency=selected,checkpoint=checkpoint)
    full,_=run_training_job(job,dataset,stats,cache,residency=selected,updates=2)
    assert resumed==full
    assert report['updates']==2
    assert nodes.UniMateTrain.GET_NODE_INFO_V1()['output']==['UNIMATE_TRAINING_CHECKPOINT','STRING']


def test_checkpoint_file_nodes_confine_paths_and_declare_input_staging(tmp_path):
    import folder_paths
    import pytest
    from unittest.mock import patch
    from test_training_checkpoint_io import value
    checkpoint=value()
    output=tmp_path/'output'
    input_root=tmp_path/'input'
    output.mkdir()
    input_root.mkdir()
    with patch.object(folder_paths,'get_output_directory',return_value=str(output)),\
         patch.object(folder_paths,'get_input_directory',return_value=str(input_root)):
        result=nodes.UniMateSaveTrainingCheckpoint.execute(checkpoint,'training/state')
        descriptor=result.ui['files'][0]
        payload=(output/descriptor['subfolder']/descriptor['filename']).read_bytes()
        target=input_root/'resume.unimatetrain'
        target.write_bytes(payload)
        loaded=nodes.UniMateLoadTrainingCheckpoint.execute(target.name).result[0]
        assert loaded==checkpoint
        assert nodes.UniMateLoadTrainingCheckpoint.cloud_offload_assets({'archive':target.name})==[
            dict(category='__input__',filename=target.name)]
        assert len(nodes.UniMateLoadTrainingCheckpoint.fingerprint_inputs(target.name))==64
        for name in ('../resume.unimatetrain','resume.pt','missing.unimatetrain'):
            with pytest.raises((ValueError,OSError)):
                nodes.UniMateLoadTrainingCheckpoint.execute(name)
        with pytest.raises(ValueError):
            nodes.UniMateSaveTrainingCheckpoint.execute(checkpoint,'../state')
        assert nodes.UniMateSaveTrainingCheckpoint.GET_SCHEMA().is_output_node


def test_checkpoint_save_cancellation_removes_staged_file(tmp_path):
    import folder_paths
    import pytest
    from unittest.mock import patch
    from comfy import model_management as mm
    from test_training_checkpoint_io import value
    checkpoint=value()
    output=tmp_path/'output'
    output.mkdir()
    cancelled=[]
    def cancel():
        stages=list(output.rglob('.unimate-numeric-*'))
        if stages:
            assert len(stages)==1 and stages[0].read_bytes().startswith(b'UMTRAIN1')
            cancelled.append(True)
            raise InterruptedError('cancelled before publication')
    with patch.object(folder_paths,'get_output_directory',return_value=str(output)),\
         patch.object(mm,'throw_exception_if_processing_interrupted',side_effect=cancel):
        with pytest.raises(InterruptedError,match='before publication'):
            nodes.UniMateSaveTrainingCheckpoint.execute(checkpoint,'cancel/state')
    assert cancelled==[True]
    assert not [path for path in output.rglob('*') if path.is_file()]


def test_inference_weight_nodes_roundtrip_and_staging(tmp_path):
    import folder_paths
    import pytest
    from unittest.mock import patch
    from test_inference_weights import checkpoint
    from unimate_pack.inference_weights import create_inference_model
    output=tmp_path/'output'
    inputs=tmp_path/'input'
    output.mkdir()
    inputs.mkdir()
    with patch.object(folder_paths,'get_output_directory',return_value=str(output)),\
         patch.object(folder_paths,'get_input_directory',return_value=str(inputs)):
        source=checkpoint()
        exported=nodes.UniMateExportInferenceWeights.execute(source,'ema','trained/weights',512)
        value=exported.result[0]
        descriptor=exported.ui['files'][0]
        data=(output/descriptor['subfolder']/descriptor['filename']).read_bytes()
        (inputs/'trained.unimateweights').write_bytes(data)
        loaded=nodes.UniMateLoadInferenceWeights.execute('trained.unimateweights',512).result[0]
        assert loaded==value and not create_inference_model(loaded).training
        assert nodes.UniMateLoadInferenceWeights.cloud_offload_assets({'archive':'trained.unimateweights'})==[
            dict(category='__input__',filename='trained.unimateweights')]
        assert len(nodes.UniMateLoadInferenceWeights.fingerprint_inputs('trained.unimateweights',512))==64
        with pytest.raises(ValueError):
            nodes.UniMateLoadInferenceWeights.execute('../trained.unimateweights',512)
        with pytest.raises(ValueError,match='workspace'):
            nodes.UniMateExportInferenceWeights.execute(source,'ema','trained/weights',1)
        assert nodes.UniMateExportInferenceWeights.GET_SCHEMA().is_output_node


def test_actual_training_sample_node():
    from unimate_pack.training_sample_contracts import validate_training_sample
    dataset,stats,cache=inputs()
    result=nodes.UniMatePrepareTrainingSample.execute(dataset,stats,cache,'a',
        max_motion_length=8,max_joints=16,augmentation='addition',augmentation_seed=17)
    value=result.result[0]
    numeric=validate_training_sample(value)
    assert numeric['motion'].shape==(8,8,12)
    report=json.loads(result.result[1])
    assert report['provenance']['clip_id']=='a'


def test_text_node_delegates_model_and_dataset():
    from test_nodes import NodeTests
    tester=NodeTests()
    cache=build_text_cache(['root'],encoder,IDENTITY)
    captured={}
    def producer(model,dataset,**options):
        captured.update(model=model,dataset=dataset,options=options)
        return cache
    with tester.fake_module('training_text_model',build_dataset_text_cache=producer):
        result=nodes.UniMateBuildTextCache.execute({'model':True},{'dataset':True})
    assert result.result[0]==cache
    assert captured['model']=={'model':True}
    assert captured['options']['cancel'] is not None


def test_actual_codecs_preserve_training_values(tmp_path):
    from pathlib import Path
    from test_dataset_transport import codec
    from unimate_pack.training_sample_contracts import validate_training_sample
    from unimate_pack.training_text import validate_text_cache
    from unimate_pack.training_batch_contracts import validate_training_batch
    dataset,stats,cache=inputs()
    sample=nodes.UniMatePrepareTrainingSample.execute(dataset,stats,cache,'a',max_motion_length=8,max_joints=16).result[0]
    batch=nodes.UniMateCollateTrainingSamples.execute([sample,sample],[512]).result[0]
    from test_training_job import config
    from unimate_pack.training_job import validate_training_job
    job=nodes.UniMateTrainingJob.execute(dataset,stats,cache,json.dumps(config())).result[0]
    root=Path(__file__).resolve().parents[2]
    client=codec(root/'ComfyUI-Cloud-Offload/partition_protocol.py','training_client_codec')
    runner=codec(root/'cloud-offload/cloud_offload/partition_protocol.py','training_runner_codec')
    for kind,value,validate in [('UNIMATE_TEXT_CACHE',cache,validate_text_cache),
                              ('UNIMATE_TRAINING_SAMPLE',sample,validate_training_sample),
                              ('UNIMATE_TRAINING_BATCH',batch,validate_training_batch),
                              ('UNIMATE_TRAINING_JOB',job,lambda value:validate_training_job(value,dataset,stats,cache))]:
        client.validate_boundary_type(kind)
        runner.validate_boundary_type(kind)
        path=tmp_path/(kind+' client.partition')
        client.dump_bundle(client.pack_execution_values([value]),path)
        restored=runner.unpack_execution_values(runner.load_bundle(path))
        assert restored==[value]
        validate(restored[0])
        path=tmp_path/(kind+' worker.partition')
        runner.dump_bundle(runner.pack_execution_values(restored),path)
        assert client.unpack_execution_values(client.load_bundle(path))==[value]
